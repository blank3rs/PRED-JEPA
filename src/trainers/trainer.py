import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel
import logging
import math
from typing import Dict, Any, Optional

from ..utils.losses import energy_loss, masked_mse_loss

class UnifiedTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        device: str = 'cuda',
        distributed: bool = False,
        local_rank: int = 0,
        scheduler: Optional[Any] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.distributed = distributed
        self.local_rank = local_rank
        self.scaler = torch.amp.GradScaler('cuda' if device == 'cuda' else 'cpu')
        
        # Initialize distributed training if needed
        if distributed:
            self.model = DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True
            )
        
        # Metrics tracking
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], task: str = 'pretrain'):
        total_loss = 0
        loss_dict = {}
        
        if task == 'pretrain':
            # Compute masked prediction losses
            transformer_outputs = outputs['transformer_outputs']
            
            if 'image_pred' in transformer_outputs and 'image' in batch:
                image_loss = masked_mse_loss(
                    transformer_outputs['image_pred'],
                    batch['image'].reshape(batch['image'].shape[0], -1, 3),
                    transformer_outputs['image_mask']
                )
                total_loss += image_loss
                loss_dict['image_loss'] = image_loss.item()
            
            if 'text_pred' in transformer_outputs and 'text_ids' in batch:
                text_loss = nn.CrossEntropyLoss()(
                    transformer_outputs['text_pred'].view(-1, transformer_outputs['text_pred'].size(-1)),
                    batch['text_ids'].view(-1)
                )
                total_loss += text_loss
                loss_dict['text_loss'] = text_loss.item()
            
            # Add contrastive loss between modalities
            if 'image' in batch and 'text_embedding' in batch:
                image_embed = outputs['transformer_outputs']['last_hidden_state'][:, 0]
                text_embed = batch['text_embedding']
                contrastive_loss = energy_loss(image_embed, text_embed)
                total_loss += 0.1 * contrastive_loss
                loss_dict['contrastive_loss'] = contrastive_loss.item()
        
        elif task == 'rl':
            # Compute RL losses
            policy_logits = outputs['policy_logits']
            values = outputs['value']
            advantages = batch['advantages']
            returns = batch['returns']
            
            # Policy loss
            policy_loss = -(advantages * torch.log_softmax(policy_logits, dim=-1)).mean()
            
            # Value loss
            value_loss = nn.MSELoss()(values, returns)
            
            # Entropy regularization
            entropy_loss = -torch.mean(torch.sum(
                torch.softmax(policy_logits, dim=-1) * torch.log_softmax(policy_logits, dim=-1),
                dim=-1
            ))
            
            total_loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
            loss_dict.update({
                'policy_loss': policy_loss.item(),
                'value_loss': value_loss.item(),
                'entropy_loss': entropy_loss.item()
            })
        
        return total_loss, loss_dict
    
    def train_epoch(self, train_loader, epoch: int, task: str = 'pretrain'):
        self.model.train()
        total_loss = 0
        num_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            try:
                with autocast():
                    outputs = self.model(batch, task=task)
                    loss, loss_dict = self.compute_loss(outputs, batch, task)
                
                # Backward pass with gradient scaling
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                # Clip gradients
                if self.distributed:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.module.parameters(), 1.0)
                else:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                if self.scheduler is not None:
                    self.scheduler.step()
                
                total_loss += loss.item()
                
                # Logging
                if batch_idx % 100 == 0:
                    lr = self.optimizer.param_groups[0]['lr']
                    loss_info = ' '.join([f'{k}: {v:.4f}' for k, v in loss_dict.items()])
                    logging.info(
                        f'Epoch: {epoch} [{batch_idx}/{num_batches} '
                        f'({100. * batch_idx / num_batches:.0f}%)] '
                        f'Loss: {loss.item():.4f} {loss_info} LR: {lr:.6f}'
                    )
                    
                    self.metrics['train_loss'].append(loss.item())
                    self.metrics['learning_rates'].append(lr)
                
            except Exception as e:
                logging.error(f"Error in training batch: {e}")
                continue
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def validate(self, val_loader, task: str = 'pretrain'):
        self.model.eval()
        total_loss = 0
        num_batches = len(val_loader)
        
        for batch in val_loader:
            try:
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(batch, task=task)
                loss, _ = self.compute_loss(outputs, batch, task)
                total_loss += loss.item()
                
            except Exception as e:
                logging.error(f"Error in validation batch: {e}")
                continue
        
        avg_loss = total_loss / num_batches
        self.metrics['val_loss'].append(avg_loss)
        
        return avg_loss 
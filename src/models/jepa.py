import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .unified_transformer import UnifiedMultiModalTransformer
from .perception import EnhancedPerception
from .world_model import EnhancedWorldModel
from .actor_critic import EnhancedActor, EnhancedCritic
import logging

class JEPAModel(nn.Module):
    """
    Unified JEPA model that combines:
    - Multi-modal transformer (images + text)
    - Enhanced perception module
    - World model with predictive learning
    - Actor-Critic for RL
    - Memory system with FAISS
    """
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        embed_dim=256,
        num_heads=8,
        dropout=0.1,
        device='cuda',
        memory_size=10000,
        use_faiss=True
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.device = device
        
        # Memory settings
        self.memory_size = memory_size
        self.use_faiss = use_faiss
        self.memory_counter = 0
        
        # Initialize memory buffers
        self.register_buffer('memory_keys', torch.zeros(memory_size, embed_dim))
        self.register_buffer('memory_values', torch.zeros(memory_size, embed_dim))
        self.register_buffer('memory_age', torch.zeros(memory_size))
        
        # Initialize components
        self.perception = EnhancedPerception(
            input_channels=3,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            device=device
        )
        
        self.world_model = EnhancedWorldModel(
            input_dim=embed_dim,
            hidden_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            device=device
        )
        
        self.transformer = UnifiedMultiModalTransformer(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=12,
            num_heads=num_heads,
            mlp_ratio=4,
            qkv_bias=True,
            drop_rate=dropout,
            attn_drop_rate=dropout,
            max_text_len=512,
            vocab_size=30522
        )
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.cosine_sim = nn.CosineSimilarity(dim=-1)
        
        # Enhanced Actor-Critic modules
        self.actor = EnhancedActor(
            embed_dim=embed_dim,
            hidden_dim=embed_dim*4,
            num_actions=1000
        )
        
        self.critic = EnhancedCritic(
            embed_dim=embed_dim,
            hidden_dim=embed_dim*4
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Move model to device
        self.to(device)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def _compute_jepa_loss(self, pred_features, target_features, mask):
        """Compute JEPA loss between predicted and target features"""
        # Normalize features
        pred_norm = F.normalize(pred_features, p=2, dim=-1)
        target_norm = F.normalize(target_features, p=2, dim=-1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(pred_norm, target_norm.transpose(-2, -1))
        
        # Temperature-scaled cross entropy loss
        temperature = 0.1
        sim_matrix = sim_matrix / temperature
        
        # Create labels (diagonal is positive pairs)
        labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
        
        # Compute cross entropy loss
        loss = F.cross_entropy(sim_matrix, labels)
        
        # Add MSE loss for masked regions
        mse = self.mse_loss(pred_features * mask.unsqueeze(-1),
                           target_features * mask.unsqueeze(-1))
        
        return loss + mse
    
    def update_memory(self, keys, values):
        """Update memory with new key-value pairs."""
        # Skip memory update if not using memory
        if not hasattr(self, 'memory_keys'):
            return
            
        # Ensure keys and values are 2D
        if keys.dim() == 3:
            keys = keys.mean(dim=1)  # Average over sequence length
        if values.dim() == 3:
            values = values.mean(dim=1)  # Average over sequence length
            
        batch_size = keys.size(0)
        
        # Update memory entries
        for i in range(batch_size):
            if self.memory_counter < self.memory_size:
                idx = self.memory_counter
                self.memory_counter += 1
            else:
                idx = torch.argmax(self.memory_age).item()
            
            self.memory_keys[idx] = keys[i].detach()
            self.memory_values[idx] = values[i].detach()
            self.memory_age[idx] = 0
            self.memory_age += 1
        
        # Update FAISS index
        if self.use_faiss and self.memory_counter > 0:
            if self.memory_counter >= self.memory_size:
                self.index.reset()
                keys_2d = self.memory_keys[:self.memory_counter].cpu().numpy()
                if keys_2d.size > 0:  # Only add if we have valid keys
                    self.index.add(keys_2d)
            else:
                keys_2d = keys.cpu().numpy()
                if keys_2d.size > 0:  # Only add if we have valid keys
                    self.index.add(keys_2d)
    
    def retrieve_from_memory(self, query, top_k=5):
        """
        Retrieve relevant memories given query features
        Args:
            query: Query features to search with
            top_k: Number of nearest neighbors to retrieve
        Returns:
            Dictionary containing retrieved memories
        """
        if self.memory_counter == 0:
            return {
                'keys': None,
                'preds': None,
                'targets': None
            }
        
        if self.use_faiss:
            # Use FAISS for efficient similarity search
            similarities, indices = self.index.search(
                query.cpu().numpy(), 
                min(top_k, self.memory_counter)
            )
            retrieved_keys = self.memory_keys[indices]
            retrieved_values = self.memory_values[indices]
            
            # Split retrieved values into predictions and targets
            retrieved_preds = retrieved_values[..., :self.embed_dim]
            retrieved_targets = retrieved_values[..., self.embed_dim:]
        else:
            # Use cosine similarity for small memory sizes
            similarities = F.cosine_similarity(
                query.unsqueeze(1),
                self.memory_keys[:self.memory_counter].unsqueeze(0),
                dim=2
            )
            top_k = min(top_k, self.memory_counter)
            _, indices = similarities.topk(top_k, dim=1)
            retrieved_keys = self.memory_keys[indices]
            retrieved_values = self.memory_values[indices]
            
            # Split retrieved values into predictions and targets
            retrieved_preds = retrieved_values[..., :self.embed_dim]
            retrieved_targets = retrieved_values[..., self.embed_dim:]
        
        return {
            'keys': retrieved_keys,
            'preds': retrieved_preds,
            'targets': retrieved_targets,
            'similarities': similarities
        }
    
    def forward(self, image, text_input):
        """Unified forward pass with proper output packaging"""
        try:
            # Process inputs through perception module
            perceptual_output = self.perception(image)
            
            # Explicitly unpack all 3 components from perception
            pred_features, target_features, mask = perceptual_output
            
            # Update world model memory with both features
            self.world_model.update_memory(pred_features, target_features)
            
            # Process through transformer
            transformer_outputs = self.transformer(
                visual_features=pred_features,
                text_features=text_input
            )
            
            # Get world model predictions
            world_outputs = self.world_model(pred_features)
            
            # Package all outputs in a dictionary
            return {
                'transformer_outputs': {
                    'text_pred': transformer_outputs['text_pred'],
                    'image_pred': transformer_outputs.get('image_pred', None),
                    'last_hidden_state': transformer_outputs['last_hidden_state']
                },
                'world_outputs': world_outputs,
                'actor_output': self.actor(transformer_outputs['last_hidden_state']),
                'critic_output': self.critic(transformer_outputs['last_hidden_state']),
                'perception_mask': mask
            }
            
        except Exception as e:
            logging.error(f"Forward pass error: {str(e)}")
            raise RuntimeError(f"Model forward pass failed: {str(e)}") from e

    def load_state_dict(self, state_dict, strict=True):
        """Load the model state dict directly."""
        try:
            # Try loading directly
            super().load_state_dict(state_dict, strict=False)
            logging.info("Successfully loaded model weights")
        except Exception as e:
            logging.error(f"Failed to load model weights: {str(e)}")
            raise 

    @classmethod
    def from_pretrained(cls, model_path='models/best_model.pt', device='cuda'):
        """Load the best pretrained model."""
        # Create model instance with device
        model = cls(device=device)
        
        try:
            # Load state dict
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            logging.info(f"Successfully loaded model from {model_path}")
        except Exception as e:
            logging.error(f"Failed to load model from {model_path}: {str(e)}")
            raise
            
        model = model.to(device)
        model.eval()
        return model 
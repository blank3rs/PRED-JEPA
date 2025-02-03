import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .transformer_blocks import TransformerBlock
from ..utils.faiss_utils import create_index

class EnhancedWorldModel(nn.Module):
    def __init__(self, 
                 input_dim,
                 hidden_dim=256,
                 num_layers=3,
                 num_heads=8,
                 dropout=0.1,
                 memory_size=10000,
                 device='cuda'):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.memory_size = memory_size
        self.device = device
        
        # Temporal embedding
        self.temporal_embed = nn.Linear(input_dim, hidden_dim)
        
        # Temporal transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                max_seq_len=16  # Support up to 16 temporal steps
            ) for _ in range(num_layers)
        ])
        
        # Predictive heads
        self.state_pred_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, input_dim)
        )
        
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, 1),
            nn.Sigmoid()
        )
        
        # Memory module
        self.register_buffer('memory_keys', torch.zeros(memory_size, hidden_dim))
        self.register_buffer('memory_values', torch.zeros(memory_size, input_dim * 2))  # Store both pred and target
        self.register_buffer('memory_age', torch.zeros(memory_size))
        self.memory_counter = 0
        
        # Initialize FAISS index
        self.index = create_index(hidden_dim, device)
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
    
    def update_memory(self, pred_features, target_features=None):
        """Update memory with predicted and target features"""
        batch_size = pred_features.size(0)
        device = pred_features.device
        
        # Project features to hidden dim
        features = self.temporal_embed(pred_features)
        
        # Combine pred and target features for values if available
        if target_features is not None:
            values = torch.cat([pred_features, target_features], dim=-1)
        else:
            values = torch.cat([pred_features, pred_features], dim=-1)  # Use pred twice if no target
        
        # Convert to numpy for FAISS
        features_np = features.detach().cpu().numpy()
        
        # Get indices to update
        indices = []
        for i in range(batch_size):
            if self.memory_counter < self.memory_size:
                idx = self.memory_counter
                self.memory_counter += 1
            else:
                # Replace oldest memory
                idx = self.memory_age.argmax().item()
            indices.append(idx)
            
        indices = torch.tensor(indices, device=device)
        
        # Update memories
        self.memory_keys[indices] = features
        self.memory_values[indices] = values
        self.memory_age[indices] = 0
        self.memory_age += 1  # Age existing memories
        
        # Update FAISS index
        if self.index.ntotal > 0:
            self.index.reset()
        self.index.add(features_np)
    
    def retrieve_from_memory(self, query_features, k=5):
        """
        Retrieve relevant memories given query features
        Args:
            query_features: Query features to search with
            k: Number of nearest neighbors to retrieve
        Returns:
            Tuple of (retrieved_keys, retrieved_preds, retrieved_targets)
        """
        if self.memory_counter == 0:
            return None, None, None
            
        # Project query to hidden dim
        query = self.temporal_embed(query_features)
        
        # Convert query to numpy for FAISS
        query_np = query.detach().cpu().numpy()
        
        # Search k-nearest neighbors
        D, I = self.index.search(query_np, min(k, self.memory_counter))
        
        # Get corresponding memories
        retrieved_keys = self.memory_keys[I]
        retrieved_values = self.memory_values[I]
        
        # Split retrieved values back into pred and target
        retrieved_preds = retrieved_values[..., :self.input_dim]
        retrieved_targets = retrieved_values[..., self.input_dim:]
        
        return retrieved_keys, retrieved_preds, retrieved_targets
    
    def forward(self, x, temporal_context=None):
        """
        Forward pass through world model
        Args:
            x: Input features [batch_size, seq_len, input_dim]
            temporal_context: Optional temporal context from previous steps
        Returns:
            Dictionary containing model outputs
        """
        # Project to hidden dim
        x = self.temporal_embed(x)
        
        # Add temporal dimension if not present
        if x.dim() == 3:
            x = x.unsqueeze(1)  # [B, 1, N, D]
        
        # Combine with temporal context if provided
        if temporal_context is not None:
            x = torch.cat([temporal_context, x], dim=1)
        
        # Process through transformer layers with temporal attention
        temporal_features = []
        for layer in self.transformer_layers:
            x = layer(x)
            temporal_features.append(x)
        
        # Combine temporal features
        x = torch.cat(temporal_features, dim=1)
        x = self.feature_fusion(x.reshape(*x.shape[:-2], -1))
        
        # Generate predictions with uncertainty
        state_pred = self.state_pred_head(x)
        uncertainty = self.uncertainty_head(x)
        
        return {
            'state_pred': state_pred,
            'uncertainty': uncertainty,
            'temporal_features': x
        } 
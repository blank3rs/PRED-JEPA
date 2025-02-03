import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer_blocks import TransformerBlock

class Flatten2D(nn.Module):
    def forward(self, x):
        return x.flatten(2).transpose(1, 2)  # B, N, C

class EnhancedPerception(nn.Module):
    def __init__(self, 
                 input_channels=3,
                 patch_size=16,
                 embed_dim=256,
                 num_layers=4,
                 num_heads=8,
                 dropout=0.1,
                 device='cuda'):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.device = device
        
        # Predictor network (processes masked images)
        self.predictor = self._create_encoder_network(input_channels)
        self.predictor_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        # Target network (processes full images)
        self.target = self._create_encoder_network(input_channels)
        self.target_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        # Initialize target with same weights as predictor but stop gradients
        self._init_target_network()
        
    def _create_encoder_network(self, input_channels):
        """Creates a multi-scale patch embedding network with transformer blocks"""
        return nn.ModuleDict({
            'patch_embed': nn.Sequential(
                # First scale: patch_size x patch_size patches
                nn.Conv2d(input_channels, self.embed_dim // 2, 
                         kernel_size=self.patch_size, stride=self.patch_size),
                nn.GELU(),
                # Second scale: patch_size/2 x patch_size/2 patches
                nn.Conv2d(self.embed_dim // 2, self.embed_dim, 
                         kernel_size=self.patch_size // 2, stride=self.patch_size // 2),
                nn.GELU(),
                # Reshape to sequence
                Flatten2D(),  # B, N, C
                nn.LayerNorm(self.embed_dim)  # Apply LayerNorm after reshaping
            ),
            
            'transformer_blocks': nn.ModuleList([
                TransformerBlock(
                    dim=self.embed_dim,
                    num_heads=8,
                    dropout=0.1,
                    max_seq_len=256  # Support up to 16x16 patches
                ) for _ in range(4)
            ])
        })
    
    def _init_target_network(self):
        """Initialize target network with predictor weights and stop gradients"""
        for param_p, param_t in zip(self.predictor.parameters(), 
                                  self.target.parameters()):
            param_t.data.copy_(param_p.data)
            param_t.requires_grad = False
            
        for param_p, param_t in zip(self.predictor_mlp.parameters(),
                                  self.target_mlp.parameters()):
            param_t.data.copy_(param_p.data)
            param_t.requires_grad = False
    
    def update_target_network(self, momentum=0.99):
        """Update target network with EMA of predictor network"""
        with torch.no_grad():
            for param_p, param_t in zip(self.predictor.parameters(),
                                      self.target.parameters()):
                param_t.data = momentum * param_t.data + (1 - momentum) * param_p.data
                
            for param_p, param_t in zip(self.predictor_mlp.parameters(),
                                      self.target_mlp.parameters()):
                param_t.data = momentum * param_t.data + (1 - momentum) * param_p.data
    
    def _process_input(self, x, network):
        """Process input through specified network"""
        # Patch embedding and transformations
        x = network['patch_embed'](x)  # Now returns B, N, C
        
        # Process through transformer blocks
        for block in network['transformer_blocks']:
            x = block(x)
            
        return x
    
    def forward(self, x, mask=None):
        """
        Forward pass through perception module
        Args:
            x: Input tensor [batch_size, channels, height, width]
            mask: Optional binary mask for masking input [batch_size, height, width]
        Returns:
            pred_features: Features from predictor network
            target_features: Features from target network (detached)
            mask: Mask used (if any)
        """
        B, C, H, W = x.shape
        
        # Generate random mask if none provided
        if mask is None:
            mask_size = H // self.patch_size
            mask = torch.rand(B, mask_size, mask_size, device=x.device) > 0.75
            mask = mask.float()
            # Upsample mask to image size
            mask = F.interpolate(mask.unsqueeze(1), size=(H, W), mode='nearest')
            mask = mask.squeeze(1)
        
        # Create masked input
        masked_x = x * mask.unsqueeze(1)
        
        # Process through predictor (masked input)
        pred_features = self._process_input(masked_x, self.predictor)  # B, N, C
        pred_features = self.predictor_mlp(pred_features)  # B, N, C
        
        # Process through target (full input)
        with torch.no_grad():
            target_features = self._process_input(x, self.target)  # B, N, C
            target_features = self.target_mlp(target_features)  # B, N, C
        
        # Ensure all outputs have correct shape
        assert pred_features.ndim == 3, f"Expected 3D tensor (B,N,C), got shape {pred_features.shape}"
        assert target_features.ndim == 3, f"Expected 3D tensor (B,N,C), got shape {target_features.shape}"
        assert pred_features.size(-1) == self.embed_dim, f"Expected last dim to be {self.embed_dim}, got {pred_features.size(-1)}"
        
        return (
            pred_features, 
            target_features.detach(), 
            mask
        ) 
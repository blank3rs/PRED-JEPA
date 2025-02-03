import torch
import torch.nn as nn
from einops import rearrange
from .transformer_blocks import TransformerBlock
import logging

class UnifiedMultiModalTransformer(nn.Module):
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        max_text_len=512,
        vocab_size=30522
    ):
        super().__init__()
        
        # Image embedding
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2)
        )
        self.num_patches = (image_size // patch_size) ** 2
        
        # Text embedding
        self.text_embed = nn.Embedding(vocab_size, embed_dim)
        
        # Position embeddings - separate for image and text
        self.pos_embed_image = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.pos_embed_text = nn.Parameter(torch.zeros(1, max_text_len, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.sep_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Calculate maximum possible sequence length
        max_seq_len = self.num_patches + max_text_len + 2  # patches + text + CLS + SEP
        
        # Transformer blocks with correct max sequence length
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                dropout=drop_rate,
                max_seq_len=max_seq_len
            ) for _ in range(depth)
        ])
        
        # Output heads
        self.image_head = nn.Linear(embed_dim, patch_size * patch_size * 3)
        self.text_head = nn.Linear(embed_dim, vocab_size)
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize weights
        nn.init.normal_(self.pos_embed_image, std=0.02)
        nn.init.normal_(self.pos_embed_text, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.sep_token, std=0.02)
        
        # Save hyperparameters
        self.max_text_len = max_text_len
        self.embed_dim = embed_dim
    
    def random_masking(self, x, mask_ratio=0.75):
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward(self, visual_features=None, text_features=None, memory_features=None):
        """
        Forward pass through unified transformer
        Args:
            visual_features: Optional visual features [batch_size, num_patches, embed_dim]
            text_features: Optional text features [batch_size, seq_len] or [batch_size, seq_len, embed_dim]
            memory_features: Optional memory features [batch_size, mem_len, embed_dim]
        Returns:
            Dictionary containing model outputs
        """
        batch_size = 1  # Default batch size
        device = self.cls_token.device
        
        # Process visual features if provided
        if visual_features is not None:
            batch_size = visual_features.shape[0]
            # Add position embeddings to visual features
            visual_features = visual_features + self.pos_embed_image[:, :visual_features.shape[1]]
        else:
            visual_features = torch.empty((batch_size, 0, self.embed_dim), device=device)
            
        # Process text if provided
        if text_features is not None:
            batch_size = text_features.shape[0]
            # Convert text IDs to embeddings if necessary
            if text_features.dim() == 2:  # [batch_size, seq_len]
                text_features = self.text_embed(text_features)  # [batch_size, seq_len, embed_dim]
            # Add position embeddings to text features
            text_features = text_features + self.pos_embed_text[:, :text_features.shape[1]]
        else:
            text_features = torch.empty((batch_size, 0, self.embed_dim), device=device)
        
        # Expand cls and sep tokens to batch size
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        sep_tokens = self.sep_token.expand(batch_size, -1, -1)
        
        # Combine all features
        x = torch.cat([
            cls_tokens,
            visual_features,
            sep_tokens,
            text_features
        ], dim=1)
        
        # Add memory features if provided
        if memory_features is not None:
            x = torch.cat([x, memory_features], dim=1)
        
        # Process through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Apply final normalization
        x = self.norm(x)
        
        # Extract features for each modality
        outputs = {
            'cls_output': x[:, 0],  # CLS token output
            'visual_output': x[:, 1:visual_features.shape[1]+1],  # Visual features
            'text_output': x[:, visual_features.shape[1]+2:visual_features.shape[1]+2+text_features.shape[1]]  # Text features
        }
        
        if memory_features is not None:
            outputs['memory_output'] = x[:, -memory_features.shape[1]:]
            
        # Add predictions
        outputs['text_pred'] = self.text_head(outputs['text_output'])
        if visual_features.shape[1] > 0:
            outputs['image_pred'] = self.image_head(outputs['visual_output'])
        
        return {
            'text_pred': outputs.get('text_pred', None), 
            'image_pred': outputs.get('image_pred', None),
            'last_hidden_state': x
        }

    def convert_old_state_dict(self, state_dict):
        """Convert old state dict format to new format."""
        new_state_dict = {}
        
        for key, value in state_dict.items():
            if key == 'transformer.pos_embed':
                # Split old combined position embedding into separate image and text embeddings
                num_patches = self.num_patches
                # First part (after CLS) goes to image
                new_state_dict['pos_embed_image'] = value[:, 1:num_patches+1]
                # Last part goes to text (excluding SEP token position)
                new_state_dict['pos_embed_text'] = value[:, num_patches+2:]
                # Save CLS and SEP token embeddings
                new_state_dict['cls_token'] = value[:, :1]
                new_state_dict['sep_token'] = value[:, num_patches+1:num_patches+2]
            elif 'rel_pos_bias' in key:
                # Get block index
                block_idx = int(key.split('.')[2])
                
                # Calculate new size based on max sequence length
                max_seq_len = self.num_patches + self.max_text_len + 2  # patches + text + CLS + SEP
                new_size = 2 * max_seq_len - 1
                
                # Create interpolation coordinates
                old_size = value.size(0)
                old_coords = torch.linspace(-1, 1, old_size)
                new_coords = torch.linspace(-1, 1, new_size)
                
                # Interpolate values
                new_bias = torch.zeros((new_size, value.size(1)), device=value.device)
                for head in range(value.size(1)):
                    new_bias[:, head] = torch.nn.functional.interpolate(
                        value[:, head].unsqueeze(0).unsqueeze(0),
                        size=new_size,
                        mode='linear',
                        align_corners=True
                    ).squeeze()
                
                new_state_dict[key.replace('transformer.', '')] = new_bias
            elif 'rel_pos_indices' in key:
                # Calculate new indices
                max_seq_len = self.num_patches + self.max_text_len + 2
                row_ids = torch.arange(max_seq_len).unsqueeze(1).expand(-1, max_seq_len)
                col_ids = torch.arange(max_seq_len).unsqueeze(0).expand(max_seq_len, -1)
                new_indices = row_ids - col_ids + max_seq_len - 1
                new_state_dict[key.replace('transformer.', '')] = new_indices
            else:
                # Remove transformer prefix from keys
                new_key = key.replace('transformer.', '')
                new_state_dict[new_key] = value
        
        return new_state_dict
    
    def load_state_dict(self, state_dict, strict=True):
        """Load the model state dict directly."""
        try:
            super().load_state_dict(state_dict, strict=False)
            logging.info("Successfully loaded transformer weights")
        except Exception as e:
            logging.error(f"Failed to load transformer weights: {str(e)}")
            raise 
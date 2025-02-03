import torch
import torch.nn as nn
import math

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1, max_seq_len=1024):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttentionWithRelPos(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            max_seq_len=max_seq_len
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout)
        )
        
        # Gating mechanism
        self.gate1 = nn.Parameter(torch.ones(1))
        self.gate2 = nn.Parameter(torch.ones(1))

    def forward(self, x, mask=None):
        # Apply attention with mask
        x = x + self.gate1 * self.attn(self.norm1(x), mask=mask)
        # Apply MLP
        x = x + self.gate2 * self.mlp(self.norm2(x))
        return x

class MultiHeadAttentionWithRelPos(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1, max_seq_len=1024):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize relative position bias table
        self.max_seq_len = max_seq_len
        self.rel_pos_bias = nn.Parameter(torch.zeros(2 * max_seq_len - 1, num_heads))
        nn.init.trunc_normal_(self.rel_pos_bias, std=0.02)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scale query
        q = q * self.scale
        
        # Compute attention scores
        attn = q @ k.transpose(-2, -1)  # (B, num_heads, N, N)
        
        # Generate relative position indices dynamically for current sequence length
        positions = torch.arange(N, device=x.device).unsqueeze(1) - torch.arange(N, device=x.device).unsqueeze(0)
        rel_pos_indices = positions + self.max_seq_len - 1
        
        # Get relative position bias for current sequence length
        rel_pos_bias = self.rel_pos_bias[rel_pos_indices]  # (N, N, num_heads)
        rel_pos_bias = rel_pos_bias.permute(2, 0, 1)  # (num_heads, N, N)
        attn = attn + rel_pos_bias.unsqueeze(0)  # (B, num_heads, N, N)
        
        # Apply attention mask if provided
        if mask is not None:
            mask = mask.float()
            attn_mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
            attn_mask = attn_mask.expand(-1, self.num_heads, N, -1)  # (B, num_heads, N, N)
            attn_mask = attn_mask * attn_mask.transpose(-2, -1)  # (B, num_heads, N, N)
            attn = attn.masked_fill(attn_mask < 0.5, -1e9)
        
        # Apply softmax and dropout
        attn = torch.softmax(attn, dim=-1, dtype=torch.float32)
        attn = self.dropout(attn)
        
        # Compute output
        x = (attn @ v).transpose(1, 2)  # (B, N, num_heads, head_dim)
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x 
# Standard library imports
import os
import io
import math
import time
import json
import pickle
import logging
import threading
import concurrent.futures
from typing import List, Tuple, Dict, Optional, Any
from datetime import datetime
from queue import Queue
from collections import deque
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

# Deep learning imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.parallel import DistributedDataParallel

# Vision and text processing imports
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import numpy as np

# Web crawling and data processing
import requests
from bs4 import BeautifulSoup
import yt_dlp
import arxiv
import aiohttp
import asyncio
import psutil

# Additional utilities
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("FAISS not available, falling back to torch similarity search")

try:
    from einops import rearrange, repeat
    from einops.layers.torch import Rearrange, Reduce
except ImportError:
    print("Installing einops...")
    import subprocess
    subprocess.check_call(["pip", "install", "einops"])
    from einops import rearrange, repeat
    from einops.layers.torch import Rearrange, Reduce

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('jepa.log'),
        logging.StreamHandler()
    ]
)

# Check CUDA availability
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {DEVICE}")

# Constants
MAX_MEMORY_SIZE = 100000
BATCH_SIZE = 64
NUM_WORKERS = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 30
WARMUP_RATIO = 0.1
SAVE_INTERVAL = 300  # 5 minutes

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

@dataclass
class ModelConfig:
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    patch_size: int = 16
    max_position_embeddings: int = 512
    num_modalities: int = 3  # text, image, video
    dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    initializer_range: float = 0.02
    memory_size: int = MAX_MEMORY_SIZE
    memory_dim: int = 768

class TextProcessor:
    def __init__(self, model_name: str = 'bert-base-uncased', device: str = DEVICE):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device
        
    def get_embeddings(self, texts: List[str], max_length: int = 512) -> torch.Tensor:
        embeddings = []
        
        for text in texts:
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                max_length=max_length,
                truncation=True,
                padding='max_length'
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :]  # Use CLS token embedding
                embeddings.append(embedding)
        
        return torch.cat(embeddings, dim=0)

# 🟢 World Model - Predicts Latent Representations
class WorldModel(nn.Module):
    def __init__(self, embed_dim=256, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x):
        return self.net(x)

# 🟡 Actor - Makes Action Decisions
class Actor(nn.Module):
    def __init__(self, embed_dim=256, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# 🔴 Critic - Evaluates Cost Function
class Critic(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, 1)
        )

    def forward(self, x):
        return self.net(x)

class EnhancedPerception(nn.Module):
    def __init__(self, image_size=28, patch_size=4, embed_dim=512, num_heads=8, num_layers=12, dropout=0.1):
        super().__init__()
        # Image processing with advanced vision transformer
        self.patch_embed = nn.Sequential(
            nn.Conv2d(1, embed_dim//2, kernel_size=patch_size, stride=patch_size),
            nn.LayerNorm([embed_dim//2, image_size//patch_size, image_size//patch_size]),
            nn.GELU(),
            nn.Conv2d(embed_dim//2, embed_dim, kernel_size=1),
            nn.LayerNorm([embed_dim, image_size//patch_size, image_size//patch_size]),
        )
        
        num_patches = (image_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Advanced transformer blocks with relative position bias
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        
        # Multi-scale feature fusion
        self.feature_fusion = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(3)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding with multi-scale features
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        # Add position embeddings and CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Multi-scale feature processing
        features = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i % 4 == 0:  # Store features at different scales
                features.append(self.feature_fusion[i//4](x[:, 0]))
            
        x = self.norm(x)
        # Combine multi-scale features
        if features:
            x = torch.cat([x[:, 0]] + features, dim=-1)
        else:
            x = x[:, 0]
        
        return x

# 🏆 Energy-Based Loss Function
def energy_loss(context_embedding, predicted_embedding, temperature=1.0):
    # InfoNCE-style contrastive loss
    similarity = torch.mm(context_embedding, predicted_embedding.t()) / temperature
    labels = torch.arange(similarity.size(0)).to(similarity.device)
    return nn.CrossEntropyLoss()(similarity, labels)

# 🎯 Load Dataset (MNIST for simplicity)
transform_train = transforms.Compose([
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Dataset setup
dataset = datasets.MNIST(root='./data', train=True, transform=transform_train, download=True)
val_dataset = datasets.MNIST(root='./data', train=False, transform=transform_val, download=True)

train_size = int(0.9 * len(dataset))
train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

# ⚡ Initialize Models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
perception = EnhancedPerception().to(device)
world_model = WorldModel().to(device)
actor = Actor().to(device)
critic = Critic().to(device)

# 🚀 Optimizers
optimizer = optim.AdamW([
    {'params': perception.parameters(), 'lr': 1e-4},
    {'params': world_model.parameters(), 'lr': 1e-4},
    {'params': actor.parameters(), 'lr': 1e-4},
    {'params': critic.parameters(), 'lr': 1e-4}
], weight_decay=0.01)

num_epochs = 30
num_training_steps = len(train_loader) * num_epochs
num_warmup_steps = num_training_steps // 10

scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
scaler = GradScaler()

# 🏋️ Training Loop
def train_epoch(epoch):
    perception.train()
    world_model.train()
    actor.train()
    critic.train()
    
    total_loss = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        with autocast():
            # Forward pass
            context_embedding = perception(images)
            predicted_embedding = world_model(context_embedding)
            actions = actor(predicted_embedding)
            costs = critic(predicted_embedding)
            
            # Compute losses
            pred_loss = energy_loss(context_embedding, predicted_embedding)
            class_loss = nn.CrossEntropyLoss()(actions, labels)
            cost_loss = torch.mean(costs)
            
            loss = pred_loss + 0.1 * class_loss + 0.01 * cost_loss

        # Backward pass with gradient scaling
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(perception.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        total_loss += loss.item()
    
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(images)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    return total_loss / len(train_loader)

def validate():
    perception.eval()
    world_model.eval()
    actor.eval()
    critic.eval()
    
    val_loss = 0
    correct = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            context_embedding = perception(images)
            predicted_embedding = world_model(context_embedding)
            actions = actor(predicted_embedding)
            
            pred = actions.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
            
    accuracy = 100. * correct / len(val_loader.dataset)
    print(f'\nValidation set: Accuracy: {correct}/{len(val_loader.dataset)} ({accuracy:.2f}%)\n')
    return accuracy

# Main training loop
best_accuracy = 0
for epoch in range(num_epochs):
    train_loss = train_epoch(epoch)
    accuracy = validate()
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save({
            'perception_state_dict': perception.state_dict(),
            'world_model_state_dict': world_model.state_dict(),
            'actor_state_dict': actor.state_dict(),
            'critic_state_dict': critic.state_dict(),
        }, 'best_model.pt')

print(f"Training complete! Best validation accuracy: {best_accuracy:.2f}%")

# Web Crawler for Wikipedia
class WikiCrawler:
    def __init__(self, start_url: str, max_pages: int = 100):
        self.start_url = start_url
        self.max_pages = max_pages
        self.visited = set()
        self.articles = []
        
    def is_valid_wiki_url(self, url: str) -> bool:
        return url.startswith('https://en.wikipedia.org/wiki/') and not any(x in url for x in [':', 'Main_Page', 'Special:', 'Talk:'])
    
    def extract_text(self, soup: BeautifulSoup) -> str:
        content = soup.find(id='mw-content-text')
        if not content:
            return ""
        
        # Remove unwanted elements
        for element in content.find_all(['table', 'script', 'style', 'sup', 'span']):
            element.decompose()
            
        paragraphs = content.find_all('p')
        return ' '.join(p.get_text().strip() for p in paragraphs)
    
    def crawl(self) -> List[str]:
        queue = [self.start_url]
        
        while queue and len(self.articles) < self.max_pages:
            url = queue.pop(0)
            if url in self.visited:
                continue
                
            try:
                response = requests.get(url, headers={'User-Agent': 'Research Bot (Educational Purpose)'})
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract text content
                text = self.extract_text(soup)
                if len(text.split()) > 50:  # Only keep substantial articles
                    self.articles.append(text)
                    print(f"Crawled {len(self.articles)} articles")
                
                # Find new links
                for link in soup.find_all('a'):
                    href = link.get('href')
                    if href:
                        full_url = urljoin('https://en.wikipedia.org', href)
                        if self.is_valid_wiki_url(full_url):
                            queue.append(full_url)
                
                self.visited.add(url)
                time.sleep(1)  # Be nice to Wikipedia
                
            except Exception as e:
                print(f"Error crawling {url}: {e}")
                
        return self.articles

# Text Processing and Embedding
class TextProcessor:
    def __init__(self, model_name: str = 'bert-base-uncased', device: str = DEVICE):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device
        
    def get_embeddings(self, texts: List[str], max_length: int = 512) -> torch.Tensor:
        embeddings = []
        
        for text in texts:
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                max_length=max_length,
                truncation=True,
                padding='max_length'
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :]  # Use CLS token embedding
                embeddings.append(embedding)
                
        return torch.cat(embeddings, dim=0)

# Combined Dataset for Images and Text
class MultiModalDataset(Dataset):
    def __init__(self, text_embeddings: torch.Tensor, images_dataset: datasets.MNIST):
        self.text_embeddings = text_embeddings
        self.images_dataset = images_dataset
        
    def __len__(self):
        return min(len(self.text_embeddings), len(self.images_dataset))
    
    def __getitem__(self, idx):
        image, label = self.images_dataset[idx]
        text_embedding = self.text_embeddings[idx % len(self.text_embeddings)]
        return image, text_embedding, label

# Enhanced Perception Module


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttentionWithRelPos(dim, num_heads, dropout)
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

    def forward(self, x):
        x = x + self.gate1 * self.attn(self.norm1(x))
        x = x + self.gate2 * self.mlp(self.norm2(x))
        return x

class MultiHeadAttentionWithRelPos(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        # Relative position bias
        self.rel_pos_bias = nn.Parameter(torch.zeros(2 * 7 - 1, num_heads))
        positions = torch.arange(7).unsqueeze(1) - torch.arange(7).unsqueeze(0)
        self.register_buffer('rel_pos_indices', positions + 6)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention with relative position bias
        attn = (q @ k.transpose(-2, -1)) * self.scale
        rel_pos_bias = self.rel_pos_bias[self.rel_pos_indices.view(-1)]
        rel_pos_bias = rel_pos_bias.view(7, 7, -1)
        rel_pos_bias = rel_pos_bias.permute(2, 0, 1)
        attn = attn + rel_pos_bias.unsqueeze(0)
        
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class MaskedAutoencoder(nn.Module):
    def __init__(self, image_size=28, patch_size=4, embed_dim=512, decoder_dim=256, mask_ratio=0.75):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        num_patches = (image_size // patch_size) ** 2
        
        # Encoder (using EnhancedPerception)
        self.encoder = EnhancedPerception(image_size, patch_size, embed_dim)
        
        # Decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_dim)
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(decoder_dim, num_heads=8) for _ in range(4)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, patch_size**2)  # Predict raw pixels
        
        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        
        # Initialize weights
        nn.init.normal_(self.mask_token, std=0.02)
    
    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep the first len_keep tokens
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # Generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward_encoder(self, x, mask_ratio):
        # Get patches
        x = self.encoder.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        # Add position embeddings
        x = x + self.encoder.pos_embed[:, 1:, :]
        
        # Masking
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        # Append cls token
        cls_token = self.encoder.cls_token + self.encoder.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Apply Transformer blocks
        for blk in self.encoder.blocks:
            x = blk(x)
        x = self.encoder.norm(x)
        
        return x, mask, ids_restore
    
    def forward_decoder(self, x, ids_restore):
        # Embed tokens
        x = self.decoder_embed(x)
        
        # Append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # skip cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        
        # Apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        
        # Predictor projection
        x = self.decoder_pred(x)
        
        # Remove cls token
        x = x[:, 1:, :]
        
        return x
    
    def forward(self, imgs, mask_ratio=None):
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        
        return pred, mask

class EnhancedWorldModel(nn.Module):
    def __init__(self, embed_dim=512, hidden_dim=2048, num_layers=8, memory_size=100000):
        super().__init__()
        
        # Multi-layer transformer for world modeling
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads=8) for _ in range(num_layers)
        ])
        
        # Predictive heads for different aspects of the world
        self.state_predictor = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        
        self.uncertainty_predictor = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.Sigmoid()
        )
        
        # Enhanced memory module with FAISS integration
        self.memory_size = memory_size
        self.memory_dim = embed_dim
        self.register_buffer('memory_keys', torch.zeros(memory_size, embed_dim))
        self.register_buffer('memory_values', torch.zeros(memory_size, embed_dim))
        self.register_buffer('memory_age', torch.zeros(memory_size))
        self.memory_counter = 0
        
        # Memory attention
        self.memory_key = nn.Linear(embed_dim, embed_dim)
        self.memory_value = nn.Linear(embed_dim, embed_dim)
        self.memory_query = nn.Linear(embed_dim, embed_dim)
        
        # Initialize FAISS index for fast similarity search
        try:
            import faiss
            self.index = faiss.IndexFlatIP(embed_dim)  # Inner product similarity
            self.use_faiss = True
        except ImportError:
            print("FAISS not available, falling back to torch similarity search")
            self.use_faiss = False
    
    def update_memory(self, keys, values):
        batch_size = keys.size(0)
        
        # Update memory using age-based replacement
        for i in range(batch_size):
            # Find oldest memories to replace
            if self.memory_counter < self.memory_size:
                idx = self.memory_counter
                self.memory_counter += 1
            else:
                idx = torch.argmax(self.memory_age).item()
            
            # Update memory
            self.memory_keys[idx] = keys[i].detach()
            self.memory_values[idx] = values[i].detach()
            self.memory_age[idx] = 0
            
            # Age other memories
            self.memory_age += 1
        
        # Update FAISS index if available
        if self.use_faiss and self.memory_counter > 0:
            if self.memory_counter >= self.memory_size:
                self.index.reset()
                self.index.add(self.memory_keys.cpu().numpy())
            else:
                self.index.add(keys.cpu().numpy())
    
    def retrieve_from_memory(self, query, top_k=5):
        if self.memory_counter == 0:
            return None, None
        
        if self.use_faiss:
            # Use FAISS for fast similarity search
            similarities, indices = self.index.search(query.cpu().numpy(), min(top_k, self.memory_counter))
            retrieved_keys = self.memory_keys[indices]
            retrieved_values = self.memory_values[indices]
        else:
            # Fallback to torch similarity search
            similarities = F.cosine_similarity(query.unsqueeze(1), 
                                            self.memory_keys[:self.memory_counter].unsqueeze(0), dim=2)
            top_k = min(top_k, self.memory_counter)
            _, indices = similarities.topk(top_k, dim=1)
            retrieved_keys = self.memory_keys[indices]
            retrieved_values = self.memory_values[indices]
        
        return retrieved_keys, retrieved_values

    def forward(self, x):
        batch_size = x.size(0)
        
        # Process through transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Generate memory keys and values
        keys = self.memory_key(x)
        values = self.memory_value(x)
        query = self.memory_query(x)
        
        # Retrieve relevant memories
        retrieved_keys, retrieved_values = self.retrieve_from_memory(query)
        
        if retrieved_keys is not None:
            # Compute attention with retrieved memories
            attention = torch.matmul(query, retrieved_keys.transpose(-2, -1)) / math.sqrt(query.size(-1))
            attention = F.softmax(attention, dim=-1)
            memory_output = torch.matmul(attention, retrieved_values)
            
            # Combine current state with memory
            x = x + memory_output
        
        # Update memory with current batch
        self.update_memory(keys, values)
        
        # Generate predictions
        state_pred = self.state_predictor(x)
        uncertainty = self.uncertainty_predictor(x)
        
        return state_pred, uncertainty

def masked_mse_loss(pred, target, mask):
    """
    Compute MSE loss on masked patches only
    """
    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
    loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
    return loss

# Main training function
def train_multimodal_jepa():
    # Initialize web crawler and process text
    crawler = WikiCrawler(start_url='https://en.wikipedia.org/wiki/Artificial_intelligence', max_pages=1000)
    articles = crawler.crawl()
    
    text_processor = TextProcessor()
    text_embeddings = text_processor.get_embeddings(articles)
    
    # Setup datasets
    transform_train = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    mnist_dataset = datasets.MNIST(root='./data', train=True, transform=transform_train, download=True)
    multimodal_dataset = MultiModalDataset(text_embeddings, mnist_dataset)
    
    # Create data loaders
    train_size = int(0.9 * len(multimodal_dataset))
    val_size = len(multimodal_dataset) - train_size
    train_dataset, val_dataset = random_split(multimodal_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    
    # Initialize models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    perception = EnhancedPerception().to(device)
    world_model = EnhancedWorldModel().to(device)
    actor = EnhancedActor(embed_dim=512).to(device)
    critic = EnhancedCritic(embed_dim=512).to(device)
    
    # Training setup
    optimizer = optim.AdamW([
        {'params': perception.parameters(), 'lr': 1e-4},
        {'params': world_model.parameters(), 'lr': 1e-4},
        {'params': actor.parameters(), 'lr': 1e-4},
        {'params': critic.parameters(), 'lr': 1e-4}
    ], weight_decay=0.01)
    
    num_epochs = 30
    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps = num_training_steps // 10
    
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    scaler = GradScaler()
    
    # Training loop
    best_accuracy = 0
    for epoch in range(num_epochs):
        # Training
        perception.train()
        world_model.train()
        actor.train()
        critic.train()
        
        total_loss = 0
        for batch_idx, (images, text_embeddings, labels) in enumerate(train_loader):
            images = images.to(device)
            text_embeddings = text_embeddings.to(device)
            labels = labels.to(device)
            
            with autocast():
                # Forward pass with combined features
                context_embedding = perception(images)
                predicted_embedding = world_model(context_embedding)
                actions = actor(predicted_embedding)
                costs = critic(predicted_embedding)
                
                # Compute losses
                pred_loss = energy_loss(context_embedding, predicted_embedding)
                cost_loss = torch.mean(costs)
                loss = pred_loss + 0.01 * cost_loss
            
            # Backward pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(perception.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(images)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
        # Validation
        accuracy = validate_multimodal(perception, world_model, actor, val_loader, device)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({
                'perception_state_dict': perception.state_dict(),
                'world_model_state_dict': world_model.state_dict(),
                'actor_state_dict': actor.state_dict(),
                'critic_state_dict': critic.state_dict(),
            }, 'best_multimodal_model.pt')
    
    print(f"Training complete! Best validation accuracy: {best_accuracy:.2f}%")

def validate_multimodal(perception, world_model, actor, val_loader, device):
    perception.eval()
    world_model.eval()
    actor.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, text_embeddings, labels in val_loader:
            images = images.to(device)
            text_embeddings = text_embeddings.to(device)
            labels = labels.to(device)
            
            context_embedding = perception(images)
            predicted_embedding = world_model(context_embedding)
            actions = actor(predicted_embedding)
            
            _, predicted = torch.max(actions.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Validation Accuracy: {accuracy:.2f}%')
    return accuracy

class DataSourceConfig:
    WIKIPEDIA_SEEDS = [
        'https://en.wikipedia.org/wiki/Artificial_intelligence',
        'https://en.wikipedia.org/wiki/Machine_learning',
        'https://en.wikipedia.org/wiki/Deep_learning',
        'https://en.wikipedia.org/wiki/Computer_vision',
        'https://en.wikipedia.org/wiki/Natural_language_processing',
        'https://en.wikipedia.org/wiki/Robotics',
        'https://en.wikipedia.org/wiki/Neural_network',
        'https://en.wikipedia.org/wiki/Reinforcement_learning',
        'https://en.wikipedia.org/wiki/Computer_science',
        'https://en.wikipedia.org/wiki/Data_science'
    ]
    
    ARXIV_CATEGORIES = [
        'cs.AI', 'cs.LG', 'cs.CV', 'cs.CL', 'cs.RO', 'cs.NE',
        'stat.ML', 'cs.IR', 'cs.DC', 'cs.SI'
    ]
    
    IMAGE_DATASETS = [
        {'name': 'MNIST', 'loader': datasets.MNIST},
        {'name': 'CIFAR10', 'loader': datasets.CIFAR10},
        {'name': 'CIFAR100', 'loader': datasets.CIFAR100},
        {'name': 'FashionMNIST', 'loader': datasets.FashionMNIST}
    ]
    
    VIDEO_SOURCES = [
        'youtube.com', 'vimeo.com', 'dailymotion.com',
        'ted.com', 'coursera.org', 'udacity.com'
    ]

class EnhancedDataCollector:
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.arxiv_client = arxiv.Client()
        self.youtube_dl_opts = {
            'format': 'best[height<=480]',
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True
        }
        
    async def collect_wikipedia_data(self):
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_wiki_page(session, url) for url in self.config.WIKIPEDIA_SEEDS]
            return await asyncio.gather(*tasks)
    
    async def collect_arxiv_data(self):
        search = arxiv.Search(
            query = "artificial intelligence OR machine learning OR deep learning",
            max_results = 1000,
            sort_by = arxiv.SortCriterion.SubmittedDate
        )
        
        papers = []
        async for result in self.arxiv_client.results(search):
            papers.append({
                'title': result.title,
                'abstract': result.summary,
                'authors': [author.name for author in result.authors],
                'pdf_url': result.pdf_url,
                'categories': result.categories
            })
        return papers
    
    async def collect_image_data(self):
        datasets = []
        for dataset_info in self.config.IMAGE_DATASETS:
            try:
                dataset = dataset_info['loader'](
                    root='./data',
                    train=True,
                    download=True,
                    transform=self.get_transform(dataset_info['name'])
                )
                datasets.append(dataset)
            except Exception as e:
                logging.error(f"Error loading {dataset_info['name']}: {e}")
        return datasets
    
    def get_transform(self, dataset_name):
        if dataset_name in ['CIFAR10', 'CIFAR100']:
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        elif dataset_name in ['MNIST', 'FashionMNIST']:
            return transforms.Compose([
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        return transforms.ToTensor()

class DFSWebCrawler:
    def __init__(self, max_depth=10, max_threads=None):
        self.max_depth = max_depth
        self.visited = set()
        self.data_lock = threading.Lock()
        self.url_stack = []
        self.thread_pool = None
        
        # Auto-detect optimal thread count based on CPU and memory
        if max_threads is None:
            cpu_count = os.cpu_count()
            memory_gb = psutil.virtual_memory().total / (1024 ** 3)
            self.max_threads = min(int(cpu_count * 2), int(memory_gb * 2))
        else:
            self.max_threads = max_threads
            
        # Rate limiting per domain
        self.domain_locks = {}
        self.domain_lock_lock = threading.Lock()
        
        # Metrics
        self.metrics = {
            'pages_crawled': 0,
            'bytes_downloaded': 0,
            'start_time': time.time()
        }
        
        # Content queues with size limits based on available memory
        memory_gb = psutil.virtual_memory().total / (1024 ** 3)
        self.text_queue = Queue(maxsize=int(100000 * memory_gb))
        self.image_queue = Queue(maxsize=int(50000 * memory_gb))
        self.video_queue = Queue(maxsize=int(10000 * memory_gb))
        
        logging.info(f"Initialized DFS crawler with {self.max_threads} threads")

    def get_domain_lock(self, domain):
        with self.domain_lock_lock:
            if domain not in self.domain_locks:
                self.domain_locks[domain] = threading.Lock()
            return self.domain_locks[domain]

    async def crawl_url_dfs(self, url, depth=0):
        if depth > self.max_depth or url in self.visited:
            return

        domain = urlparse(url).netloc
        domain_lock = self.get_domain_lock(domain)

        with domain_lock:
            try:
                with self.data_lock:
                    if url in self.visited:
                        return
                    self.visited.add(url)

                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers={'User-Agent': 'Research Bot (Educational Purpose)'}) as response:
                        if response.status != 200:
                            return

                        content_type = response.headers.get('content-type', '')
                        content = await response.read()
                        
                        self.metrics['bytes_downloaded'] += len(content)
                        self.metrics['pages_crawled'] += 1

                        # Process different content types
                        if 'text/html' in content_type:
                            soup = BeautifulSoup(content, 'html.parser')
                            
                            # Extract and process links first (DFS)
                            links = []
                            for link in soup.find_all('a'):
                                href = link.get('href')
                                if href:
                                    full_url = urljoin(url, href)
                                    if self.is_valid_url(full_url):
                                        links.append(full_url)
                            
                            # Process current page content
                            text = self.extract_text(soup)
                            if text:
                                if not self.text_queue.full():
                                    self.text_queue.put({'url': url, 'text': text, 'depth': depth})
                            
                            # Extract media
                            for img in soup.find_all('img'):
                                img_url = img.get('src')
                                if img_url and self.is_valid_image_url(img_url):
                                    if not self.image_queue.full():
                                        self.image_queue.put({'url': img_url, 'depth': depth})
                            
                            for video in soup.find_all(['video', 'iframe']):
                                video_url = video.get('src')
                                if video_url and self.is_valid_video_url(video_url):
                                    if not self.video_queue.full():
                                        self.video_queue.put({'url': video_url, 'depth': depth})
                            
                            # DFS: Process links immediately
                            for link in links:
                                await self.crawl_url_dfs(link, depth + 1)
                        
                        elif 'image' in content_type:
                            if not self.image_queue.full():
                                self.image_queue.put({'url': url, 'content': content, 'depth': depth})
                        
                        elif self.is_video_url(url):
                            if not self.video_queue.full():
                                self.video_queue.put({'url': url, 'depth': depth})

            except Exception as e:
                logging.error(f"Error crawling {url}: {e}")

    def start_crawling(self, seed_urls):
        async def main():
            tasks = []
            for url in seed_urls:
                task = asyncio.create_task(self.crawl_url_dfs(url))
                tasks.append(task)
            await asyncio.gather(*tasks)

        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_threads)
        asyncio.run(main())

    @staticmethod
    def is_valid_url(url):
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False

    @staticmethod
    def is_valid_image_url(url):
        return any(url.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp'])

    @staticmethod
    def is_valid_video_url(url):
        video_platforms = ['youtube.com', 'vimeo.com', 'dailymotion.com']
        return any(platform in url.lower() for platform in video_platforms)

    def get_metrics(self):
        elapsed_time = time.time() - self.metrics['start_time']
        return {
            'pages_crawled': self.metrics['pages_crawled'],
            'bytes_downloaded': self.metrics['bytes_downloaded'],
            'elapsed_time': elapsed_time,
            'crawl_rate': self.metrics['pages_crawled'] / elapsed_time if elapsed_time > 0 else 0,
            'memory_usage': psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        }

# Add this class definition before EnhancedContinuousLearningSystem
class ContinuousLearningSystem:
    def __init__(self, model_save_dir='./models', data_cache_dir='./data_cache'):
        self.model_save_dir = model_save_dir
        self.data_cache_dir = data_cache_dir
        os.makedirs(model_save_dir, exist_ok=True)
        os.makedirs(data_cache_dir, exist_ok=True)
        
        # Initialize queues for data collection and processing
        self.text_queue = Queue(maxsize=10000)
        self.image_queue = Queue(maxsize=10000)
        self.processed_queue = Queue(maxsize=5000)
        
        # Data buffers for continuous learning
        self.text_buffer = deque(maxlen=10000)
        self.image_buffer = deque(maxlen=10000)
        self.embedding_buffer = deque(maxlen=10000)
        
        # Threading control
        self.is_running = True
        self.threads = []
        
        # Initialize models and processors
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.initialize_models()
        
        # Learning state
        self.current_epoch = 0
        self.samples_processed = 0
        self.last_save_time = time.time()
        
        # Metrics tracking
        self.metrics = {
            'training_loss': [],
            'validation_accuracy': [],
            'samples_processed': [],
            'learning_rate': []
        }

    def initialize_models(self):
        self.perception = EnhancedPerception().to(self.device)
        self.world_model = EnhancedWorldModel().to(self.device)
        self.actor = EnhancedActor(embed_dim=512).to(self.device)
        self.critic = EnhancedCritic(embed_dim=512).to(self.device)
        
        self.text_processor = TextProcessor()
        self.optimizer = optim.AdamW([
            {'params': self.perception.parameters(), 'lr': 1e-4},
            {'params': self.world_model.parameters(), 'lr': 1e-4},
            {'params': self.actor.parameters(), 'lr': 1e-4},
            {'params': self.critic.parameters(), 'lr': 1e-4}
        ], weight_decay=0.01)
        
        self.scaler = GradScaler()

    def text_processor_worker(self):
        while self.is_running:
            try:
                if not self.text_queue.empty():
                    text_data = self.text_queue.get()
                    text = text_data['text']
                    embedding = self.text_processor.get_embeddings([text])
                    self.text_buffer.append(text)
                    self.embedding_buffer.append(embedding)
            except Exception as e:
                logging.error(f"Error in text processor: {e}")
            time.sleep(0.01)

    def image_processor_worker(self):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        while self.is_running:
            try:
                if not self.image_queue.empty():
                    image_data = self.image_queue.get()
                    if 'content' in image_data:
                        image = Image.open(io.BytesIO(image_data['content'])).convert('RGB')
                        processed_image = transform(image)
                        self.image_buffer.append(processed_image)
            except Exception as e:
                logging.error(f"Error in image processor: {e}")
            time.sleep(0.01)

    def video_processor_worker(self):
        while self.is_running:
            try:
                if not self.video_queue.empty():
                    video_data = self.video_queue.get()
                    video_url = video_data['url']
                    with yt_dlp.YoutubeDL(self.youtube_dl_opts) as ydl:
                        info = ydl.extract_info(video_url, download=False)
                        if info.get('frames'):
                            for frame in info['frames']:
                                if not self.image_queue.full():
                                    self.image_queue.put({'content': frame, 'depth': video_data['depth']})
            except Exception as e:
                logging.error(f"Error in video processor: {e}")
            time.sleep(0.1)

    def continuous_learning_worker(self):
        while self.is_running:
            try:
                if len(self.embedding_buffer) >= 32 and len(self.image_buffer) >= 32:
                    self.train_mini_batch()
                    
                    if time.time() - self.last_save_time > 3600:  # Save every hour
                        self.save_checkpoint()
                        self.last_save_time = time.time()
            except Exception as e:
                logging.error(f"Error in learning worker: {e}")
            time.sleep(0.1)

    def train_mini_batch(self):
        self.perception.train()
        self.world_model.train()
        self.actor.train()
        self.critic.train()
        
        try:
            # Prepare batch
            images = torch.stack(list(self.image_buffer)[:32]).to(self.device)
            text_embeddings = torch.cat(list(self.embedding_buffer)[:32]).to(self.device)
            
            with autocast():
                # Forward pass
                context_embedding = self.perception(images)
                predicted_embedding = self.world_model(context_embedding)
                actions = self.actor(predicted_embedding)
                costs = self.critic(predicted_embedding)
                
                # Compute losses
                pred_loss = energy_loss(context_embedding, predicted_embedding)
                cost_loss = torch.mean(costs)
                loss = pred_loss + 0.01 * cost_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.perception.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            self.samples_processed += 32
            self.metrics['training_loss'].append(loss.item())
            
            if self.samples_processed % 1000 == 0:
                logging.info(f"Processed {self.samples_processed} samples, Current loss: {loss.item():.4f}")
        
        except Exception as e:
            logging.error(f"Error in training batch: {e}")

    def save_checkpoint(self):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = os.path.join(self.model_save_dir, f'checkpoint_{timestamp}.pt')
            
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'samples_processed': self.samples_processed,
                'metrics': self.metrics,
                'current_epoch': self.current_epoch
            }, checkpoint_path)
            
            logging.info(f"Saved checkpoint to {checkpoint_path}")
        except Exception as e:
            logging.error(f"Error saving checkpoint: {e}")

    def stop(self):
        self.is_running = False
        for thread in self.threads:
            thread.join()
        self.save_checkpoint()
        logging.info("System stopped gracefully")

class EnhancedContinuousLearningSystem(ContinuousLearningSystem):
    def __init__(
        self,
        model_config: ModelConfig,
        device: str = DEVICE,
        distributed: bool = False,
        local_rank: int = 0,
        model_save_dir: str = './models',
        data_cache_dir: str = './data_cache'
    ):
        super().__init__(model_save_dir=model_save_dir, data_cache_dir=data_cache_dir)
        self.model_config = model_config
        self.device = device
        self.distributed = distributed
        self.local_rank = local_rank
        self.dfs_crawler = DFSWebCrawler()
        
        # Add save/load paths
        self.knowledge_save_path = os.path.join(self.model_save_dir, 'knowledge_buffer.pkl')
        self.conversation_save_path = os.path.join(self.model_save_dir, 'conversation_history.json')
        
        # Initialize the unified model
        self.model = JEPAModel(
            image_size=224,
            patch_size=model_config.patch_size,
            embed_dim=model_config.hidden_size,
            depth=model_config.num_hidden_layers,
            num_heads=model_config.num_attention_heads,
            memory_size=model_config.memory_size
        ).to(device)
        
        if distributed:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[local_rank],
                output_device=local_rank
            )
        
        # Initialize optimizer and trainer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )
        
        self.trainer = UnifiedTrainer(
            model=self.model,
            optimizer=self.optimizer,
            device=device,
            distributed=distributed,
            local_rank=local_rank
        )
        
        # Try to load previous knowledge
        self.load_previous_state()
        
    def start(self):
        # Start DFS crawler with multiple seed points
        crawler_thread = threading.Thread(
            target=self.dfs_crawler.start_crawling,
            args=(DataSourceConfig.WIKIPEDIA_SEEDS,)
        )
        
        # Start processing threads
        process_thread = threading.Thread(target=self.process_data)
        train_thread = threading.Thread(target=self.train_continuously)
        
        # Start all threads
        self.threads.extend([crawler_thread, process_thread, train_thread])
        for thread in self.threads:
            thread.start()
        
        logging.info(f"Started enhanced system with {len(self.threads)} threads")
    
    def process_data(self):
        while self.is_running:
            try:
                # Process queued data
                if not self.text_queue.empty():
                    text_batch = []
                    while not self.text_queue.empty() and len(text_batch) < BATCH_SIZE:
                        text_batch.append(self.text_queue.get())
                    
                    # Process text batch
                    embeddings = self.text_processor.get_embeddings(text_batch)
                    for text, embedding in zip(text_batch, embeddings):
                        self.text_buffer.append(text)
                        self.embedding_buffer.append(embedding)
                
                if not self.image_queue.empty():
                    image_batch = []
                    while not self.image_queue.empty() and len(image_batch) < BATCH_SIZE:
                        image_batch.append(self.image_queue.get())
                    
                    # Process image batch
                    processed_images = self.process_image_batch(image_batch)
                    for image in processed_images:
                        self.image_buffer.append(image)
                
                time.sleep(0.1)
                
            except Exception as e:
                logging.error(f"Error in data processing: {e}")
    
    def process_image_batch(self, images: List[Image.Image]) -> torch.Tensor:
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        processed = []
        for img in images:
            try:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                processed.append(transform(img))
            except Exception as e:
                logging.error(f"Error processing image: {e}")
                continue
        
        if processed:
            return torch.stack(processed)
        return torch.tensor([])
    
    def train_continuously(self):
        while self.is_running:
            try:
                if len(self.text_buffer) >= BATCH_SIZE and len(self.image_buffer) >= BATCH_SIZE:
                    # Prepare batch
                    batch = {
                        'text': list(self.text_buffer)[:BATCH_SIZE],
                        'image': torch.stack(list(self.image_buffer)[:BATCH_SIZE]),
                        'text_embeddings': torch.stack(list(self.embedding_buffer)[:BATCH_SIZE])
                    }
                    
                    # Train on batch
                    self.trainer.train_epoch([batch], self.current_epoch)
                    self.current_epoch += 1
                    
                    # Save periodically
                    if time.time() - self.last_save_time > SAVE_INTERVAL:
                        self.save_current_state()
                        self.last_save_time = time.time()
                
                time.sleep(0.1)
                
            except Exception as e:
                logging.error(f"Error in training: {e}")

    def save_current_state(self):
        try:
            # Save model checkpoint
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = os.path.join(self.model_save_dir, f'checkpoint_{timestamp}.pt')
            
            model_state = self.model.module.state_dict() if self.distributed else self.model.state_dict()
            
            torch.save({
                'model_state_dict': model_state,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'samples_processed': self.samples_processed,
                'metrics': self.metrics,
                'current_epoch': self.current_epoch
            }, checkpoint_path)
            
            # Save knowledge buffers
            with open(self.knowledge_save_path, 'wb') as f:
                pickle.dump({
                    'text_buffer': self.text_buffer,
                    'embedding_buffer': self.embedding_buffer,
                    'image_buffer': self.image_buffer
                }, f)
            
            # Clean up old checkpoints (keep last 5)
            checkpoints = sorted([f for f in os.listdir(self.model_save_dir) if f.startswith('checkpoint_')])
            if len(checkpoints) > 5:
                for old_checkpoint in checkpoints[:-5]:
                    os.remove(os.path.join(self.model_save_dir, old_checkpoint))
            
            logging.info(f"Saved current state to {checkpoint_path}")
            
        except Exception as e:
            logging.error(f"Error saving current state: {e}")
    
    def load_previous_state(self):
        try:
            # Load the latest model checkpoint
            checkpoints = [f for f in os.listdir(self.model_save_dir) 
                         if f.startswith('checkpoint_') and f.endswith('.pt')]
            
            if checkpoints:
                latest_checkpoint = max(checkpoints)
                checkpoint_path = os.path.join(self.model_save_dir, latest_checkpoint)
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                
                if self.distributed:
                    self.model.module.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.samples_processed = checkpoint['samples_processed']
                self.metrics = checkpoint['metrics']
                self.current_epoch = checkpoint.get('current_epoch', 0)
                
                logging.info(f"Loaded previous model state from {checkpoint_path}")
            
            # Load knowledge buffers
            if os.path.exists(self.knowledge_save_path):
                with open(self.knowledge_save_path, 'rb') as f:
                    saved_buffers = pickle.load(f)
                    self.text_buffer = saved_buffers['text_buffer']
                    self.embedding_buffer = saved_buffers['embedding_buffer']
                    self.image_buffer = saved_buffers.get('image_buffer', deque(maxlen=10000))
                    logging.info(f"Loaded {len(self.text_buffer)} previous knowledge entries")
                    
        except Exception as e:
            logging.warning(f"Could not load previous state: {e}")
            logging.info("Starting with fresh model state")
    
    def stop(self):
        logging.info("Saving final state before stopping...")
        self.save_current_state()
        self.is_running = False
        for thread in self.threads:
            thread.join()
        logging.info("System stopped gracefully")

class ChatInterface:
    def __init__(self, learning_system):
        self.learning_system = learning_system
        self.conversation_history = []
        self.load_conversation_history()
        
    def load_conversation_history(self):
        try:
            if os.path.exists(self.learning_system.conversation_save_path):
                with open(self.learning_system.conversation_save_path, 'r') as f:
                    self.conversation_history = json.load(f)
                logging.info(f"Loaded {len(self.conversation_history)} previous conversations")
        except Exception as e:
            logging.warning(f"Could not load conversation history: {e}")
    
    def save_conversation_history(self):
        try:
            with open(self.learning_system.conversation_save_path, 'w') as f:
                json.dump(self.conversation_history, f)
        except Exception as e:
            logging.error(f"Error saving conversation history: {e}")
    
    def process_query(self, user_input: str) -> str:
        response = super().process_query(user_input)
        self.save_conversation_history()  # Save after each interaction
        return response

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
            Rearrange('b c h w -> b (h w) c')
        )
        num_patches = (image_size // patch_size) ** 2
        
        # Text embedding
        self.text_embed = nn.Embedding(vocab_size, embed_dim)
        
        # Position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + max_text_len + 2, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.sep_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, drop_rate) for _ in range(depth)
        ])
        
        # Output heads
        self.image_head = nn.Linear(embed_dim, patch_size * patch_size * 3)
        self.text_head = nn.Linear(embed_dim, vocab_size)
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize weights
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.sep_token, std=0.02)
    
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
    
    def forward(self, image=None, text_ids=None, text_mask=None):
        # Process image if provided
        if image is not None:
            image_tokens = self.patch_embed(image)
            image_tokens, image_mask, image_ids_restore = self.random_masking(image_tokens)
        else:
            image_tokens = torch.tensor([]).to(self.cls_token.device)
            image_mask = None
            
        # Process text if provided
        if text_ids is not None:
            text_tokens = self.text_embed(text_ids)
            if text_mask is None:
                text_tokens, text_mask, text_ids_restore = self.random_masking(text_tokens)
        else:
            text_tokens = torch.tensor([]).to(self.cls_token.device)
            text_mask = None
        
        # Combine modalities with special tokens
        cls_tokens = self.cls_token.expand(image_tokens.shape[0], -1, -1)
        sep_tokens = self.sep_token.expand(image_tokens.shape[0], -1, -1)
        
        x = torch.cat([cls_tokens, image_tokens, sep_tokens, text_tokens], dim=1)
        x = x + self.pos_embed[:, :x.size(1)]
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        
        # Get predictions for each modality
        image_pred = self.image_head(x[:, 1:image_tokens.shape[1]+1])
        text_pred = self.text_head(x[:, -text_tokens.shape[1]:]) if text_tokens.shape[1] > 0 else None
        
        return {
            'last_hidden_state': x,
            'cls_output': x[:, 0],
            'image_pred': image_pred,
            'text_pred': text_pred,
            'image_mask': image_mask,
            'text_mask': text_mask
        }

class JEPAModel(nn.Module):
    """
    Unified JEPA model that combines:
    - Multi-modal transformer (images + text)
    - Memory system with FAISS
    - Actor-Critic heads for RL
    - Masked autoencoding for self-supervised learning
    """
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
        vocab_size=30522,
        memory_size=100000,
        num_actions=1000  # Adjustable based on task
    ):
        super().__init__()
        
        # Core transformer for multi-modal processing
        self.transformer = UnifiedMultiModalTransformer(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            max_text_len=max_text_len,
            vocab_size=vocab_size
        )
        
        # Memory system
        self.memory_size = memory_size
        self.memory_dim = embed_dim
        self.register_buffer('memory_keys', torch.zeros(memory_size, embed_dim))
        self.register_buffer('memory_values', torch.zeros(memory_size, embed_dim))
        self.register_buffer('memory_age', torch.zeros(memory_size))
        self.memory_counter = 0
        
        # Initialize FAISS index
        try:
            import faiss
            self.index = faiss.IndexFlatIP(embed_dim)
            self.use_faiss = True
        except ImportError:
            print("FAISS not available, falling back to torch similarity search")
            self.use_faiss = False
        
        # Actor-Critic heads
        self.actor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, num_actions)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def update_memory(self, keys, values):
        batch_size = keys.size(0)
        
        # Update memory using age-based replacement
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
                self.index.add(self.memory_keys.cpu().numpy())
            else:
                self.index.add(keys.cpu().numpy())
    
    def retrieve_from_memory(self, query, top_k=5):
        if self.memory_counter == 0:
            return None, None
        
        if self.use_faiss:
            similarities, indices = self.index.search(
                query.cpu().numpy(), 
                min(top_k, self.memory_counter)
            )
            retrieved_keys = self.memory_keys[indices]
            retrieved_values = self.memory_values[indices]
        else:
            similarities = F.cosine_similarity(
                query.unsqueeze(1),
                self.memory_keys[:self.memory_counter].unsqueeze(0),
                dim=2
            )
            top_k = min(top_k, self.memory_counter)
            _, indices = similarities.topk(top_k, dim=1)
            retrieved_keys = self.memory_keys[indices]
            retrieved_values = self.memory_values[indices]
        
        return retrieved_keys, retrieved_values
    
    def forward(self, batch, task='pretrain'):
        """
        Unified forward pass supporting multiple tasks:
        - pretrain: Masked autoencoding for both modalities
        - rl: Actor-critic outputs for RL
        - generate: Generate responses based on input
        """
        # Get transformer outputs
        transformer_outputs = self.transformer(
            image=batch.get('image'),
            text_ids=batch.get('text_ids'),
            text_mask=batch.get('text_mask')
        )
        
        # Get CLS token embedding
        cls_embedding = transformer_outputs['cls_output']
        
        # Memory operations
        retrieved_keys, retrieved_values = self.retrieve_from_memory(cls_embedding)
        if retrieved_values is not None:
            # Combine current embedding with memory
            attention = torch.matmul(
                cls_embedding, 
                retrieved_keys.transpose(-2, -1)
            ) / math.sqrt(cls_embedding.size(-1))
            attention = F.softmax(attention, dim=-1)
            memory_output = torch.matmul(attention, retrieved_values)
            cls_embedding = cls_embedding + memory_output
        
        # Update memory
        self.update_memory(cls_embedding, cls_embedding)
        
        # Task-specific outputs
        outputs = {'transformer_outputs': transformer_outputs}
        
        if task == 'pretrain':
            # Return masked prediction losses
            return outputs
        
        elif task == 'rl':
            # Add actor-critic outputs
            policy_logits = self.actor(cls_embedding)
            value = self.critic(cls_embedding)
            outputs.update({
                'policy_logits': policy_logits,
                'value': value
            })
            return outputs
        
        elif task == 'generate':
            # Use policy network for generation
            policy_logits = self.actor(cls_embedding)
            outputs['policy_logits'] = policy_logits
            return outputs
        
        else:
            raise ValueError(f"Unknown task: {task}")

class UnifiedTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[Any] = None,
        device: str = 'cuda',
        distributed: bool = False,
        local_rank: int = 0
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.distributed = distributed
        self.local_rank = local_rank
        self.scaler = GradScaler()
        
        # Metrics
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'rl_reward': [],
            'learning_rates': []
        }
        
        # Initialize distributed training if needed
        if distributed:
            self.model = DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True  # Important for training stability
            )
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], 
                    batch: Dict[str, torch.Tensor], 
                    task: str = 'pretrain') -> Tuple[torch.Tensor, Dict[str, float]]:
        total_loss = 0
        loss_dict = {}
        
        if task == 'pretrain':
            # Compute masked prediction losses
            transformer_outputs = outputs['transformer_outputs']
            if 'image_pred' in transformer_outputs:
                image_loss = masked_mse_loss(
                    transformer_outputs['image_pred'],
                    batch['image'].reshape(batch['image'].shape[0], -1, 3),
                    transformer_outputs['image_mask']
                )
                total_loss += image_loss
                loss_dict['image_loss'] = image_loss.item()
            
            if 'text_pred' in transformer_outputs:
                text_loss = F.cross_entropy(
                    transformer_outputs['text_pred'].view(-1, transformer_outputs['text_pred'].size(-1)),
                    batch['text_ids'].view(-1),
                    ignore_index=-100
                )
                total_loss += text_loss
                loss_dict['text_loss'] = text_loss.item()
            
            # Add contrastive loss between modalities
            if 'image_pred' in transformer_outputs and 'text_pred' in transformer_outputs:
                image_embed = outputs['transformer_outputs']['last_hidden_state'][:, 0]  # CLS token
                text_embed = outputs['transformer_outputs']['last_hidden_state'][:, -1]  # Last token
                contrastive_loss = energy_loss(image_embed, text_embed, temperature=0.07)
                total_loss += 0.1 * contrastive_loss
                loss_dict['contrastive_loss'] = contrastive_loss.item()
        
        elif task == 'rl':
            # Compute RL losses with PPO-style clipping
            policy_logits = outputs['policy_logits']
            values = outputs['value']
            old_policy_logits = batch.get('old_policy_logits')
            old_values = batch.get('old_values')
            advantages = batch['advantages']
            returns = batch['returns']
            
            # Policy loss with clipping
            ratio = torch.exp(policy_logits - old_policy_logits.detach())
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss with clipping
            value_pred_clipped = old_values + (values - old_values).clamp(-0.2, 0.2)
            value_losses = (values - returns).pow(2)
            value_losses_clipped = (value_pred_clipped - returns).pow(2)
            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
            
            # Entropy bonus for exploration
            dist = torch.distributions.Categorical(logits=policy_logits)
            entropy_loss = -dist.entropy().mean()
            
            total_loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
            loss_dict.update({
                'policy_loss': policy_loss.item(),
                'value_loss': value_loss.item(),
                'entropy_loss': entropy_loss.item()
            })
        
        return total_loss, loss_dict
    
    def train_epoch(self, train_loader: DataLoader, epoch: int, task: str = 'pretrain') -> float:
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
                
                # Backward pass with gradient clipping
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
                    
                    # Update metrics
                    self.metrics['train_loss'].append(loss.item())
                    self.metrics['learning_rates'].append(lr)
                
            except Exception as e:
                logging.error(f"Error in training batch: {e}")
                continue
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader, task: str = 'pretrain') -> float:
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

class EnhancedCritic(nn.Module):
    def __init__(self, embed_dim=512, hidden_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        return self.net(x)

class EnhancedActor(nn.Module):
    def __init__(self, embed_dim=512, hidden_dim=1024, num_actions=1000):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_actions)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    # Set up distributed training if available
    distributed = torch.cuda.device_count() > 1
    if distributed:
        torch.distributed.init_process_group(backend='nccl')
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = DEVICE
        local_rank = 0
    
    # Initialize system
    config = ModelConfig()
    system = EnhancedContinuousLearningSystem(
        model_config=config,
        device=device,
        distributed=distributed,
        local_rank=local_rank
    )
    
    try:
        system.start()
        
        # Main loop
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logging.info("Shutting down...")
        system.save_state()
        if distributed:
            torch.distributed.destroy_process_group()
        logging.info("System stopped and state saved.")


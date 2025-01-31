import torch
import torch.nn.functional as F

def energy_loss(x: torch.Tensor, y: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """
    Compute energy-based contrastive loss between two embeddings.
    """
    # Normalize embeddings
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    
    # Compute similarity matrix
    sim = torch.matmul(x, y.transpose(-2, -1)) / temperature
    
    # Compute positive and negative pairs
    labels = torch.arange(sim.size(0), device=sim.device)
    loss = F.cross_entropy(sim, labels) + F.cross_entropy(sim.transpose(-2, -1), labels)
    
    return loss / 2

def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute MSE loss only on masked positions.
    """
    # Ensure mask is binary
    mask = mask.bool()
    
    # Compute MSE loss only on masked positions
    loss = F.mse_loss(pred[mask], target[mask], reduction='mean')
    
    return loss 
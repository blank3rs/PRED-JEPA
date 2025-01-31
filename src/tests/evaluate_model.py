import torch
import logging
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.jepa import JEPAModel
from src.config.config import ModelConfig

# Configure logging to write only to file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log')
    ]
)

# Prevent logging from propagating to the root logger
logger = logging.getLogger(__name__)
logger.propagate = False

def load_checkpoint(model, checkpoint_path):
    """Load model checkpoint"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint.get('epoch', 0), checkpoint.get('best_val_loss', float('inf'))
    except Exception as e:
        logging.error(f"Error loading checkpoint: {e}")
        return 0, float('inf')

def prepare_image(image_path, image_size=224):
    """Prepare image for model input"""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

def visualize_attention(image, attention_weights, save_path=None):
    """Visualize attention weights overlaid on image"""
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    # Attention heatmap
    plt.subplot(1, 3, 2)
    plt.imshow(attention_weights, cmap='hot')
    plt.title('Attention Weights')
    plt.colorbar()
    plt.axis('off')
    
    # Overlay
    plt.subplot(1, 3, 3)
    plt.imshow(image)
    plt.imshow(attention_weights, alpha=0.6, cmap='hot')
    plt.title('Attention Overlay')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def test_model_understanding(model, device, test_image_path, test_text, save_dir='./test_results'):
    """Test what the model has learned"""
    model.eval()
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Prepare inputs
    image = prepare_image(test_image_path).to(device)
    
    with torch.no_grad():
        try:
            # Get model outputs
            outputs = model(image, [test_text])
            
            # Get attention weights from last layer
            attention_weights = model.get_attention_weights()
            
            # Reshape attention weights for visualization
            attention_map = attention_weights.mean(1)[0].reshape(
                int(np.sqrt(attention_weights.size(-1))),
                int(np.sqrt(attention_weights.size(-1)))
            ).cpu().numpy()
            
            # Visualize results
            original_image = Image.open(test_image_path).convert('RGB')
            visualize_attention(
                original_image,
                attention_map,
                save_path=str(Path(save_dir) / 'attention_visualization.png')
            )
            
            # Calculate and log similarity scores
            if hasattr(outputs, 'similarity_scores'):
                sim_scores = outputs.similarity_scores.cpu().numpy()
                logging.info(f"\nSimilarity scores between image and text: {sim_scores[0]:.4f}")
            
            # Log model's focus areas
            focus_regions = np.unravel_index(
                np.argmax(attention_map),
                attention_map.shape
            )
            logging.info(f"Model's primary focus area: {focus_regions}")
            
            return {
                'attention_map': attention_map,
                'similarity_scores': sim_scores if 'sim_scores' in locals() else None,
                'focus_regions': focus_regions
            }
            
        except Exception as e:
            logging.error(f"Error during model evaluation: {e}")
            return None

def evaluate_text_understanding(model, device, test_texts):
    """Evaluate model's text understanding"""
    model.eval()
    results = []
    
    with torch.no_grad():
        try:
            # Create text embeddings
            text_embeddings = model.encode_text(test_texts)
            
            # Calculate similarity between different texts
            similarity_matrix = F.cosine_similarity(
                text_embeddings.unsqueeze(1),
                text_embeddings.unsqueeze(0),
                dim=-1
            )
            
            # Log results
            logging.info("\nText Understanding Results:")
            for i, text1 in enumerate(test_texts):
                for j, text2 in enumerate(test_texts):
                    if i < j:
                        sim_score = similarity_matrix[i, j].item()
                        logging.info(f"\nSimilarity between:")
                        logging.info(f"Text 1: {text1}")
                        logging.info(f"Text 2: {text2}")
                        logging.info(f"Score: {sim_score:.4f}")
                        
                        results.append({
                            'text1': text1,
                            'text2': text2,
                            'similarity': sim_score
                        })
            
            return results
            
        except Exception as e:
            logging.error(f"Error evaluating text understanding: {e}")
            return None

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Initialize model
    model_config = ModelConfig()
    model = JEPAModel(
        image_size=224,
        patch_size=model_config.patch_size,
        embed_dim=model_config.hidden_size,
        depth=model_config.num_hidden_layers,
        num_heads=model_config.num_attention_heads,
        memory_size=model_config.memory_size
    ).to(device)
    
    # Load latest checkpoint
    checkpoint_dir = Path('./checkpoints')
    if checkpoint_dir.exists():
        latest_checkpoint = max(checkpoint_dir.glob('*.pt'), key=lambda x: x.stat().st_mtime)
        epoch, best_val_loss = load_checkpoint(model, latest_checkpoint)
        logging.info(f"Loaded model from epoch {epoch} with validation loss {best_val_loss:.4f}")
    
    # Test image understanding
    test_image_path = './test_data/test_image.jpg'
    test_text = "A sample image for testing the model's understanding"
    
    if Path(test_image_path).exists():
        results = test_model_understanding(model, device, test_image_path, test_text)
        if results:
            logging.info("\nImage Understanding Results:")
            logging.info(f"Focus regions: {results['focus_regions']}")
            if results['similarity_scores'] is not None:
                logging.info(f"Image-Text similarity: {results['similarity_scores']:.4f}")
    
    # Test text understanding
    test_texts = [
        "A beautiful landscape with mountains",
        "Mountains and nature scenery",
        "A busy city street at night",
        "Urban nightlife and city lights"
    ]
    
    text_results = evaluate_text_understanding(model, device, test_texts)
    if text_results:
        logging.info("\nMost similar text pairs:")
        sorted_results = sorted(text_results, key=lambda x: x['similarity'], reverse=True)
        for result in sorted_results[:3]:
            logging.info(f"\nScore: {result['similarity']:.4f}")
            logging.info(f"Text 1: {result['text1']}")
            logging.info(f"Text 2: {result['text2']}")

if __name__ == "__main__":
    main() 
import os
import torch
import torch.optim as optim
import logging
import threading
import queue
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel
import time

from .config.config import ModelConfig, DataSourceConfig
from .models.jepa import JEPAModel
from .data.dataset import DataProcessor, MultiModalDataset
from .trainers.trainer import UnifiedTrainer
from .crawlers.web_crawler import DFSWebCrawler
from .interfaces.chat import ChatInterface

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('jepa.log'),
        logging.StreamHandler()
    ]
)

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + torch.cos(torch.pi * progress)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def chat_worker(model, device, stop_event):
    """Worker function for chat interface"""
    chat_interface = ChatInterface(model, device=device)
    
    print("\nChat interface is ready! Type 'quit' to exit.")
    while not stop_event.is_set():
        try:
            user_input = input("\nYou: ")
            if user_input.lower() == 'quit':
                break
            
            response = chat_interface.process_query(user_input)
            print(f"Assistant: {response}")
            
        except Exception as e:
            logging.error(f"Error in chat interface: {e}")
            continue

def training_worker(model, optimizer, device, distributed, local_rank, stop_event):
    """Worker function for training loop"""
    try:
        # Initialize data processor and crawler
        data_processor = DataProcessor()
        crawler = DFSWebCrawler(cache_dir='./crawler_cache')
        
        # Create checkpoint and model directories
        checkpoint_dir = './checkpoints'
        model_dir = './models'
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        
        # Load latest checkpoint if exists
        start_epoch = 0
        best_val_loss = float('inf')
        checkpoint_files = sorted(os.listdir(checkpoint_dir)) if os.path.exists(checkpoint_dir) else []
        
        if checkpoint_files:
            latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[-1])
            logging.info(f"Loading checkpoint: {latest_checkpoint}")
            try:
                checkpoint = torch.load(latest_checkpoint, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch']
                best_val_loss = checkpoint['best_val_loss']
                crawler.visited = set(checkpoint.get('visited_urls', []))
                logging.info(f"Resumed from epoch {start_epoch} with validation loss {best_val_loss}")
            except Exception as e:
                logging.error(f"Error loading checkpoint: {e}")
                logging.info("Starting from scratch")
        
        # Start crawling in background
        logging.info("Starting crawler in training worker")
        crawler.start_crawling(DataSourceConfig.WIKIPEDIA_SEEDS)
        
        # Initialize trainer
        trainer = UnifiedTrainer(
            model=model,
            optimizer=optimizer,
            device=device,
            distributed=distributed,
            local_rank=local_rank
        )
        
        # Training loop
        num_epochs = 30
        min_batch_size = 8
        last_save_time = time.time()
        save_interval = 300  # Save every 5 minutes
        
        for epoch in range(start_epoch, num_epochs):
            if stop_event.is_set():
                break
            
            # Get data from crawler queues and process
            text_data = []
            image_data = []
            
            # Try to collect enough data for a batch
            while len(text_data) < min_batch_size and not crawler.text_queue.empty():
                try:
                    item = crawler.text_queue.get(timeout=1)
                    if item and item['text'] and len(item['text'].split()) > 50:
                        text_data.append(item['text'])
                except queue.Empty:
                    break
                except Exception as e:
                    logging.error(f"Error processing text data: {e}")
                    continue
            
            while len(image_data) < min_batch_size and not crawler.image_queue.empty():
                try:
                    item = crawler.image_queue.get(timeout=1)
                    if item and item.get('image'):
                        image_data.append(item['image'])
                except queue.Empty:
                    break
                except Exception as e:
                    logging.error(f"Error processing image data: {e}")
                    continue
            
            # Create dataset and loaders if we have enough data
            if len(text_data) >= min_batch_size and len(image_data) >= min_batch_size:
                # Ensure equal lengths by truncating to shorter list
                min_len = min(len(text_data), len(image_data))
                text_data = text_data[:min_len]
                image_data = image_data[:min_len]
                
                try:
                    dataset = MultiModalDataset(text_data, image_data)
                    train_loader = data_processor.get_train_loader(dataset, batch_size=min_batch_size)
                    val_loader = data_processor.get_val_loader(dataset, batch_size=min_batch_size)
                    
                    # Train epoch
                    train_loss = trainer.train_epoch(train_loader, epoch)
                    val_loss = trainer.validate(val_loader)
                    
                    logging.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
                    
                    # Save checkpoint periodically
                    current_time = time.time()
                    if current_time - last_save_time >= save_interval:
                        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}_{current_time:.0f}.pt')
                        checkpoint = {
                            'epoch': epoch + 1,  # Save next epoch to resume from
                            'model_state_dict': model.state_dict() if not distributed else model.module.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'train_loss': train_loss,
                            'val_loss': val_loss,
                            'best_val_loss': best_val_loss,
                            'visited_urls': list(crawler.visited),
                            'timestamp': current_time
                        }
                        torch.save(checkpoint, checkpoint_path)
                        logging.info(f"Saved checkpoint to {checkpoint_path}")
                        last_save_time = current_time
                        
                        # Clean up old checkpoints (keep last 5)
                        checkpoint_files = sorted(os.listdir(checkpoint_dir))
                        if len(checkpoint_files) > 5:
                            for old_checkpoint in checkpoint_files[:-5]:
                                os.remove(os.path.join(checkpoint_dir, old_checkpoint))
                    
                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        model_path = os.path.join(model_dir, 'best_model.pt')
                        best_checkpoint = {
                            'epoch': epoch + 1,
                            'model_state_dict': model.state_dict() if not distributed else model.module.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'train_loss': train_loss,
                            'val_loss': val_loss,
                            'best_val_loss': best_val_loss,
                            'visited_urls': list(crawler.visited),
                            'timestamp': current_time
                        }
                        torch.save(best_checkpoint, model_path)
                        logging.info(f"Saved best model with validation loss {val_loss:.4f}")
                
                except Exception as e:
                    logging.error(f"Error during training: {e}")
                    continue
            
            else:
                # Get crawler metrics to help diagnose issues
                metrics = crawler.get_enhanced_metrics()
                logging.info(
                    f"Epoch {epoch}: Waiting for data from crawler... "
                    f"(text: {len(text_data)}, images: {len(image_data)}) "
                    f"Pages crawled: {metrics['pages_crawled']}, "
                    f"Crawl rate: {metrics['crawl_rate']:.2f} pages/sec, "
                    f"Cache hits: {metrics['cache_hits']}, "
                    f"Success rate: {metrics['success_rate']:.2%}"
                )
                time.sleep(1)  # Wait a bit before checking again
    
    except Exception as e:
        logging.error(f"Error in training loop: {e}")
        raise
    
    finally:
        # Save final checkpoint
        try:
            final_checkpoint_path = os.path.join(checkpoint_dir, f'final_checkpoint_{time.time():.0f}.pt')
            final_checkpoint = {
                'epoch': epoch + 1 if 'epoch' in locals() else 0,
                'model_state_dict': model.state_dict() if not distributed else model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss if 'train_loss' in locals() else None,
                'val_loss': val_loss if 'val_loss' in locals() else None,
                'best_val_loss': best_val_loss,
                'visited_urls': list(crawler.visited),
                'timestamp': time.time()
            }
            torch.save(final_checkpoint, final_checkpoint_path)
            logging.info(f"Saved final checkpoint to {final_checkpoint_path}")
        except Exception as e:
            logging.error(f"Error saving final checkpoint: {e}")
        
        logging.info("Stopping crawler from training worker")
        crawler.stop()

def main():
    # Check CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    distributed = torch.cuda.device_count() > 1
    local_rank = 0
    
    if distributed:
        torch.distributed.init_process_group(backend='nccl')
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    
    # Initialize configuration
    model_config = ModelConfig()
    
    # Initialize model
    model = JEPAModel(
        image_size=224,
        patch_size=model_config.patch_size,
        embed_dim=model_config.hidden_size,
        depth=model_config.num_hidden_layers,
        num_heads=model_config.num_attention_heads,
        memory_size=model_config.memory_size
    ).to(device)
    
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # Initialize optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # Create stop event for graceful shutdown
    stop_event = threading.Event()
    
    try:
        # Start training thread
        training_thread = threading.Thread(
            target=training_worker,
            args=(model, optimizer, device, distributed, local_rank, stop_event)
        )
        training_thread.start()
        
        # Start chat interface thread
        chat_thread = threading.Thread(
            target=chat_worker,
            args=(model, device, stop_event)
        )
        chat_thread.start()
        
        # Wait for threads to complete
        training_thread.join()
        chat_thread.join()
    
    except KeyboardInterrupt:
        logging.info("Received interrupt, shutting down gracefully...")
        stop_event.set()
        
        # Wait for threads to complete
        training_thread.join()
        chat_thread.join()
    
    finally:
        # Clean up
        if distributed:
            torch.distributed.destroy_process_group()
        logging.info("Training and chat interface stopped")

if __name__ == "__main__":
    main() 
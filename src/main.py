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
    crawler = None
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
                # Only resume from checkpoint if we haven't completed all epochs
                if checkpoint['epoch'] < 30:  # num_epochs
                    start_epoch = checkpoint['epoch']
                    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
                    if 'visited_urls' in checkpoint:
                        crawler.visited = set(checkpoint['visited_urls'])
                    logging.info(f"Resumed from epoch {start_epoch} with validation loss {best_val_loss}")
                else:
                    logging.info("Starting fresh training run (previous run completed)")
            except Exception as e:
                logging.error(f"Error loading checkpoint: {e}")
                logging.info("Starting from scratch")
        
        # Start crawling in background
        logging.info("Starting crawler in training worker")
        crawler.start_crawling(DataSourceConfig.WIKIPEDIA_SEEDS)
        
        # Wait a bit for crawler to initialize and start collecting data
        time.sleep(2)
        
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
        max_batch_size = 32
        last_save_time = time.time()
        save_interval = 300  # Save every 5 minutes
        data_collection_timeout = 60  # Wait up to 60 seconds for data
        max_collection_retries = 10  # Maximum number of times to retry collecting data
        
        # Data buffers for batching
        text_buffer = []
        image_buffer = []
        
        for epoch in range(start_epoch, num_epochs):
            if stop_event.is_set():
                logging.info("Stop event received, saving checkpoint and stopping...")
                break
            
            epoch_start_time = time.time()
            batches_processed = 0
            total_loss = 0
            collection_retries = 0
            
            # Collect and process data throughout the epoch
            while time.time() - epoch_start_time < 3600:  # Run epoch for max 1 hour
                if stop_event.is_set():
                    break
                
                # Check if crawler is still running
                if not crawler.is_running or not crawler.thread_pool.is_alive():
                    logging.error("Crawler stopped unexpectedly, restarting...")
                    crawler.stop()
                    crawler = DFSWebCrawler(cache_dir='./crawler_cache')
                    crawler.visited = set()  # Start fresh
                    crawler.start_crawling(DataSourceConfig.WIKIPEDIA_SEEDS)
                    time.sleep(2)  # Wait for crawler to initialize
                    collection_retries = 0
                    continue
                
                # Try to collect data for a batch
                collection_start = time.time()
                data_collected = False
                
                while len(text_buffer) < max_batch_size and len(image_buffer) < max_batch_size:
                    if time.time() - collection_start > data_collection_timeout:
                        break
                    
                    # Get text data
                    try:
                        while not crawler.text_queue.empty() and len(text_buffer) < max_batch_size:
                            item = crawler.text_queue.get_nowait()
                            if item and item['text'] and len(item['text'].split()) > 50:
                                text_buffer.append(item['text'])
                                data_collected = True
                    except queue.Empty:
                        pass
                    
                    # Get image data
                    try:
                        while not crawler.image_queue.empty() and len(image_buffer) < max_batch_size:
                            item = crawler.image_queue.get_nowait()
                            if item and item.get('image'):
                                image_buffer.append(item['image'])
                                data_collected = True
                    except queue.Empty:
                        pass
                    
                    if len(text_buffer) < min_batch_size or len(image_buffer) < min_batch_size:
                        time.sleep(0.1)  # Short sleep to prevent CPU spinning
                
                if not data_collected:
                    collection_retries += 1
                    if collection_retries >= max_collection_retries:
                        logging.warning("No data collected after maximum retries, restarting crawler...")
                        crawler.stop()
                        crawler = DFSWebCrawler(cache_dir='./crawler_cache')
                        crawler.visited = set()  # Start fresh
                        crawler.start_crawling(DataSourceConfig.WIKIPEDIA_SEEDS)
                        time.sleep(2)  # Wait for crawler to initialize
                        collection_retries = 0
                        continue
                else:
                    collection_retries = 0
                
                # Process batch if we have enough data
                if len(text_buffer) >= min_batch_size and len(image_buffer) >= min_batch_size:
                    try:
                        # Take batch_size items from buffers
                        batch_size = min(len(text_buffer), len(image_buffer), max_batch_size)
                        batch_texts = text_buffer[:batch_size]
                        batch_images = image_buffer[:batch_size]
                        
                        # Remove used items from buffers
                        text_buffer = text_buffer[batch_size:]
                        image_buffer = image_buffer[batch_size:]
                        
                        # Create dataset and loaders
                        dataset = MultiModalDataset(batch_texts, batch_images)
                        train_loader = data_processor.get_train_loader(dataset, batch_size=batch_size)
                        val_loader = data_processor.get_val_loader(dataset, batch_size=batch_size)
                        
                        # Train on batch
                        train_loss = trainer.train_epoch(train_loader, epoch)
                        val_loss = trainer.validate(val_loader)
                        
                        total_loss += train_loss
                        batches_processed += 1
                        
                        # Log progress
                        if batches_processed % 10 == 0:
                            avg_loss = total_loss / batches_processed
                            logging.info(
                                f"Epoch {epoch} - Batch {batches_processed}: "
                                f"Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, "
                                f"Avg Loss = {avg_loss:.4f}"
                            )
                        
                        # Save checkpoint periodically
                        current_time = time.time()
                        if current_time - last_save_time >= save_interval:
                            save_checkpoint(
                                checkpoint_dir, epoch, model, optimizer,
                                train_loss, val_loss, best_val_loss,
                                crawler, current_time, distributed
                            )
                            last_save_time = current_time
                        
                        # Update best model if needed
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            save_best_model(
                                model_dir, epoch, model, optimizer,
                                train_loss, val_loss, best_val_loss,
                                crawler, current_time, distributed
                            )
                    
                    except Exception as e:
                        logging.error(f"Error processing batch: {e}")
                        continue
                
                else:
                    # Get crawler metrics to help diagnose issues
                    metrics = crawler.get_enhanced_metrics()
                    logging.info(
                        f"Epoch {epoch}: Waiting for data... "
                        f"(text buffer: {len(text_buffer)}, image buffer: {len(image_buffer)}) "
                        f"Pages crawled: {metrics['pages_crawled']}, "
                        f"Crawl rate: {metrics['crawl_rate']:.2f} pages/sec, "
                        f"Cache hits: {metrics['cache_hits']}, "
                        f"Success rate: {metrics['success_rate']:.2%}, "
                        f"Collection retries: {collection_retries}"
                    )
                    time.sleep(1)  # Wait a bit before checking again
            
            # End of epoch logging
            if batches_processed > 0:
                epoch_loss = total_loss / batches_processed
                logging.info(f"Epoch {epoch} completed - Average loss: {epoch_loss:.4f}")
    
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
                'visited_urls': list(crawler.visited) if crawler else [],
                'timestamp': time.time()
            }
            torch.save(final_checkpoint, final_checkpoint_path)
            logging.info(f"Saved final checkpoint to {final_checkpoint_path}")
        except Exception as e:
            logging.error(f"Error saving final checkpoint: {e}")
        
        # Stop crawler gracefully
        if crawler:
            try:
                logging.info("Stopping crawler from training worker")
                crawler.stop()
            except Exception as e:
                logging.error(f"Error stopping crawler: {e}")

def save_checkpoint(checkpoint_dir, epoch, model, optimizer, train_loss, val_loss, best_val_loss, crawler, timestamp, distributed):
    """Save a training checkpoint"""
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}_{timestamp:.0f}.pt')
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict() if not distributed else model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'best_val_loss': best_val_loss,
        'visited_urls': list(crawler.visited),
        'timestamp': timestamp
    }
    torch.save(checkpoint, checkpoint_path)
    logging.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Clean up old checkpoints (keep last 5)
    checkpoint_files = sorted(os.listdir(checkpoint_dir))
    if len(checkpoint_files) > 5:
        for old_checkpoint in checkpoint_files[:-5]:
            os.remove(os.path.join(checkpoint_dir, old_checkpoint))

def save_best_model(model_dir, epoch, model, optimizer, train_loss, val_loss, best_val_loss, crawler, timestamp, distributed):
    """Save the best model"""
    model_path = os.path.join(model_dir, 'best_model.pt')
    best_checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict() if not distributed else model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'best_val_loss': best_val_loss,
        'visited_urls': list(crawler.visited),
        'timestamp': timestamp
    }
    torch.save(best_checkpoint, model_path)
    logging.info(f"Saved best model with validation loss {val_loss:.4f}")

def main():
    # Optimize CUDA settings for RTX 4070
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for faster training
        torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner
        torch.backends.cudnn.deterministic = False  # Disable deterministic mode for speed
        device = torch.device('cuda')
        torch.cuda.set_device(0)  # Explicitly set to first GPU
        
        # Print GPU info
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device('cpu')
        logging.warning("CUDA not available, using CPU")
    
    distributed = False  # Only enable distributed if multiple GPUs
    local_rank = 0
    
    logging.info(f"Using device: {device} (Distributed: {distributed})")
    
    # Initialize configuration
    model_config = ModelConfig()
    
    # Initialize model with optimized settings
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
    
    # Optimize learning rate and optimizer settings for RTX 4070
    if torch.cuda.is_available():
        lr = 2e-4  # Higher learning rate for GPU
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    else:
        lr = 5e-5  # Lower learning rate for CPU
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
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
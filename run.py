"""
Entry point script for running the JEPA project.
"""
import os
import logging
import threading
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# Configure logging - separate handlers for console and file
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(message)s'))

file_handler = logging.FileHandler('jepa.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler]  # Only log to file by default
)

# Create console logger for chat interface only
chat_logger = logging.getLogger('chat')
chat_logger.addHandler(console_handler)
chat_logger.propagate = False  # Don't send chat logs to root logger

# First check GPU
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    chat_logger.info(f"\nUsing GPU: {torch.cuda.get_device_name()}")
else:
    chat_logger.info("\nNo GPU found, using CPU")

# Initialize FAISS
from src.utils.faiss_utils import initialize_faiss
gpu_support = initialize_faiss(device)

# Import our modules
from src.interfaces.chat import ChatInterface
from src.models.jepa import JEPAModel
from src.main import training_worker, chat_worker
import torch.optim as optim

# Load model and create optimizer
chat_logger.info("\nLoading model...")
model = JEPAModel.from_pretrained(device=device)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# Create stop event for graceful shutdown
stop_event = threading.Event()

# Create and start training thread
training_thread = threading.Thread(
    target=training_worker,
    args=(model, optimizer, device, False, 0, stop_event)
)
training_thread.start()

# Create and start chat thread
chat_thread = threading.Thread(
    target=chat_worker,
    args=(model, device, stop_event)
)
chat_thread.start()

try:
    # Wait for threads to complete
    training_thread.join()
    chat_thread.join()
except KeyboardInterrupt:
    chat_logger.info("\nShutting down gracefully...")
    stop_event.set()
    training_thread.join()
    chat_thread.join()
except Exception as e:
    chat_logger.error(f"\nError: {str(e)}")
finally:
    stop_event.set() 
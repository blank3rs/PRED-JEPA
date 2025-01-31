"""
Entry point script for running the JEPA project.
"""
import os
import logging
import threading
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'  # Simplified format
)

# First check GPU
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    print(f"\nUsing GPU: {torch.cuda.get_device_name()}")
else:
    print("\nNo GPU found, using CPU")

# Then initialize FAISS
import faiss
try:
    logging.info("\nInitializing FAISS...")
    if hasattr(faiss, 'swigfaiss_avx512'):
        logging.info("FAISS using AVX512")
    elif hasattr(faiss, 'swigfaiss_avx2'):
        logging.info("FAISS using AVX2")
    else:
        logging.info("FAISS using basic CPU instructions")
    
    if device == 'cuda':
        logging.info("Note: FAISS running in CPU mode - GPU FAISS not available on Windows")
except Exception as e:
    logging.error(f"FAISS initialization error: {e}")

# Import our modules
from src.interfaces.chat import ChatInterface
from src.models.jepa import JEPAModel
from src.main import training_worker, chat_worker
import torch.optim as optim

# Load model and create optimizer
print("\nLoading model...")
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
    print("\nShutting down gracefully...")
    stop_event.set()
    training_thread.join()
    chat_thread.join()
except Exception as e:
    print(f"\nError: {str(e)}")
finally:
    stop_event.set() 
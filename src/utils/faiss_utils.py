"""Utility functions for FAISS initialization and management."""
import logging
import faiss
import torch

def initialize_faiss(device='cuda'):
    """Initialize FAISS with the best available configuration.
    
    Args:
        device (str): Device to use ('cuda' or 'cpu')
        
    Returns:
        bool: Whether GPU support is available
    """
    gpu_support = False
    
    # Try to initialize GPU FAISS if CUDA is available
    if device == 'cuda' and torch.cuda.is_available():
        try:
            # Try to enable GPU support
            res = faiss.StandardGpuResources()
            gpu_support = True
            logging.info("FAISS GPU support enabled")
        except Exception:
            logging.info("FAISS running in CPU mode (GPU support not available)")
            # Fall back to CPU mode
            try_cpu_faiss()
    else:
        # CPU-only mode
        try_cpu_faiss()
    
    return gpu_support

def try_cpu_faiss():
    """Try to initialize CPU FAISS with best available instruction set."""
    try:
        # Try AVX512 first
        import faiss.swigfaiss_avx512
        logging.info("FAISS using AVX512 instructions")
    except ImportError:
        try:
            # Fall back to AVX2
            import faiss.swigfaiss_avx2
            logging.info("FAISS using AVX2 instructions")
        except ImportError:
            # Fall back to basic CPU
            logging.info("FAISS using basic CPU instructions")

def create_index(dim, device='cuda', use_gpu=False):
    """Create a FAISS index with the appropriate configuration.
    
    Args:
        dim (int): Dimension of vectors to index
        device (str): Device to use ('cuda' or 'cpu')
        use_gpu (bool): Whether GPU support is available
        
    Returns:
        faiss.Index: Configured FAISS index
    """
    if use_gpu and device == 'cuda' and torch.cuda.is_available():
        try:
            # Create GPU index
            res = faiss.StandardGpuResources()
            index = faiss.GpuIndexFlatL2(res, dim)
            logging.info("Created GPU FAISS index")
            return index
        except Exception:
            logging.info("Failed to create GPU index, falling back to CPU")
    
    # Create CPU index
    index = faiss.IndexFlatL2(dim)
    logging.info("Created CPU FAISS index")
    return index 
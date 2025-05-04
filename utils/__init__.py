# utils package
from .logger import setup_logger
from .pdf_utils import pdf_to_text

from gpu_utils import GPUVerifier

# Initialize GPU verification
gpu_verifier = GPUVerifier(require_gpu=True)

__all__ = ['setup_logger', 'pdf_to_text']
# gpu_utils.py

import os
import torch
import tensorflow as tf
import subprocess
import platform
import logging

class GPUVerifier:
    """Utility class to verify and enforce GPU usage for Python scripts."""
    
    def __init__(self, require_gpu=True, log_level=logging.INFO):
        self.require_gpu = require_gpu
        
        # Set up logging
        logging.basicConfig(level=log_level, 
                           format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger("GPUVerifier")
        
        # Check GPU availability
        self.gpu_available = self._check_gpu_available()
        self.gpu_info = self._get_gpu_info()
        
        if self.require_gpu and not self.gpu_available:
            self.logger.error("GPU is required but not available. Check your installation.")
            raise RuntimeError("GPU required but not available")
        
        self.logger.info(f"GPU Available: {self.gpu_available}")
        if self.gpu_available:
            self.logger.info(f"GPU Info: {self.gpu_info}")
    

    def _check_gpu_available(self):
        """Check if GPU is available using multiple frameworks."""
        # PyTorch CUDA check
        pytorch_cuda = torch.cuda.is_available()
        
        # PyTorch MPS check (Apple Silicon GPU)
        pytorch_mps = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        
        # TensorFlow checkS
        tf_gpus = len(tf.config.list_physical_devices('GPU')) > 0

        # NVIDIA-SMI check
        nvidia_smi_available = False
        try:
            result = subprocess.run(['which', 'nvidia-smi'], capture_output=True)
            nvidia_smi_available = result.returncode == 0
        except:
            nvidia_smi_available = False

        self.logger.debug(f"PyTorch CUDA: {pytorch_cuda}, PyTorch MPS: {pytorch_mps}, TensorFlow GPU: {tf_gpus}, NVIDIA-SMI: {nvidia_smi_available}")
        
        return pytorch_cuda or pytorch_mps or tf_gpus or nvidia_smi_available


    def _get_gpu_info(self):
        """Get detailed GPU information."""
        info = {}
        
        # PyTorch info
        if torch.cuda.is_available():
            info["pytorch"] = {
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(0),
                "memory_allocated": f"{torch.cuda.memory_allocated(0) / 1024**2:.2f} MB",
                "memory_reserved": f"{torch.cuda.memory_reserved(0) / 1024**2:.2f} MB"
            }
        
        # TensorFlow info
        tf_gpus = tf.config.list_physical_devices('GPU')
        if tf_gpus:
            info["tensorflow"] = {
                "device_count": len(tf_gpus),
                "devices": [gpu.name for gpu in tf_gpus]
            }

        # In _get_gpu_info()
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            info["pytorch_mps"] = {
                "device": "Apple Silicon MPS",
                "note": "Running on Apple GPU (Metal)"
            }

            
        return info
    
    def enable_gpu_for_pytorch(self, model=None):
        """Move a PyTorch model to GPU if available."""
        if not torch.cuda.is_available():
            self.logger.warning("PyTorch GPU not available, using CPU")
            return model
        
        if model is not None:
            model = model.cuda()
            self.logger.info(f"Model moved to GPU: {torch.cuda.get_device_name(0)}")
        
        # Set default tensor type
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.logger.info("PyTorch default tensor type set to CUDA")
        
        return model
    
    def enable_gpu_for_tensorflow(self):
        """Configure TensorFlow to use GPU."""
        tf_gpus = tf.config.list_physical_devices('GPU')
        
        if not tf_gpus:
            self.logger.warning("TensorFlow GPU not available, using CPU")
            return
        
        # Configure TensorFlow to use the first GPU and allow memory growth
        try:
            for gpu in tf_gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.set_visible_devices(tf_gpus[0], 'GPU')
            self.logger.info(f"TensorFlow configured to use GPU: {tf_gpus[0].name}")
        except RuntimeError as e:
            self.logger.error(f"Error configuring TensorFlow GPU: {e}")
    
    def monitor_gpu_usage(self):
        """Display current GPU memory usage."""
        if not torch.cuda.is_available():
            self.logger.warning("GPU monitoring not available")
            return
        
        allocated = torch.cuda.memory_allocated(0) / 1024**2
        reserved = torch.cuda.memory_reserved(0) / 1024**2
        
        self.logger.info(f"GPU Memory: Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")
        
        # You can also call nvidia-smi as an alternative
        try:
            if platform.system() == "Windows":
                subprocess.run(['nvidia-smi'], check=True)
            else:
                subprocess.run(['nvidia-smi'], check=True)
        except:
            self.logger.debug("nvidia-smi command failed")

# Simple usage example when this file is run directly
if __name__ == "__main__":
    gpu = GPUVerifier(require_gpu=False)  # Don't raise error if GPU not found
    
    if gpu.gpu_available:
        print("\n✅ GPU IS AVAILABLE AND READY TO USE!")
        print(f"GPU Info: {gpu.gpu_info}")
    else:
        print("\n⚠️ GPU NOT DETECTED. Code will run on CPU only.")
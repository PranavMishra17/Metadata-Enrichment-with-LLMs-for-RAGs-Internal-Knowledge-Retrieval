from abc import ABC, abstractmethod
import json
import os
import glob
from typing import Dict, List, Any
import uuid

from utils.logger import setup_logger

from gpu_utils import GPUVerifier

# Initialize GPU verification
gpu_verifier = GPUVerifier(require_gpu=True)

class BaseMetadataGenerator(ABC):
    """Base class for metadata generation."""
    
    def __init__(self, output_dir="metadata_gen_output"):
        """Initialize the metadata generator."""
        self.output_dir = output_dir
        self.logger = setup_logger(self.__class__.__name__)
        
        # Create output directory structure
        self._create_output_dirs()
    
    def _create_output_dirs(self):
        """Create the output directory structure."""
        # Create main output dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            self.logger.info(f"Created output directory: {self.output_dir}")
        
        # Create subdirectories for each chunking method
        for method in ["semantic_chunks_metadata", "naive_chunks_metadata", "recursive_chunks_metadata"]:
            method_dir = os.path.join(self.output_dir, method)
            if not os.path.exists(method_dir):
                os.makedirs(method_dir)
                self.logger.info(f"Created subdirectory: {method_dir}")
    
    def process_chunks(self, chunks_dir="chunk_output"):
        """Process all chunks in the specified directory."""
        self.logger.info(f"Processing chunks from {chunks_dir}")
        
        # Find all JSON files in the chunks directory
        chunk_files = glob.glob(os.path.join(chunks_dir, "*.json"))
        self.logger.info(f"Found {len(chunk_files)} chunk files")
        
        # Group files by chunking method
        semantic_files = [f for f in chunk_files if "semantic" in f.lower()]
        naive_files = [f for f in chunk_files if "naive" in f.lower()]
        recursive_files = [f for f in chunk_files if "recursive" in f.lower()]
        
        # Skip evaluation files
        semantic_files = [f for f in semantic_files if "evaluation" not in f.lower()]
        naive_files = [f for f in naive_files if "evaluation" not in f.lower()]
        recursive_files = [f for f in recursive_files if "evaluation" not in f.lower()]
        
        self.logger.info(f"Semantic files: {len(semantic_files)}")
        self.logger.info(f"Naive files: {len(naive_files)}")
        self.logger.info(f"Recursive files: {len(recursive_files)}")
        
        # Process each group separately
        for method, files in [
            ("semantic", semantic_files), 
            ("naive", naive_files), 
            ("recursive", recursive_files)
        ]:
            self.logger.info(f"Processing {method} chunks")
            for file_path in files:
                try:
                    self._process_chunk_file(file_path, method)
                except Exception as e:
                    self.logger.error(f"Error processing {file_path}: {str(e)}")
    
    def _process_chunk_file(self, file_path, method):
        """Process a single chunk file."""
        self.logger.info(f"Processing file: {file_path}")
        
        # Read chunk file
        with open(file_path, 'r', encoding='utf-8') as f:
            chunk_data = json.load(f)
        
        # Determine output directory
        output_subdir = f"{method}_chunks_metadata"
        output_dir = os.path.join(self.output_dir, output_subdir)
        
        # Extract document name for output filename
        doc_name = chunk_data.get("document_name", os.path.basename(file_path))
        doc_base = os.path.splitext(doc_name)[0]
        output_path = os.path.join(output_dir, f"{doc_base}_enriched_chunks.json")
        
        # Process chunks
        enriched_chunks = self._enrich_chunks(chunk_data, method)
        
        # Save enriched chunks
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(enriched_chunks, f, indent=2)
        
        self.logger.info(f"Saved enriched chunks to {output_path}")
        
        return output_path
    
    @abstractmethod
    def _enrich_chunks(self, chunk_data, method):
        """Abstract method to enrich chunks with metadata."""
        pass
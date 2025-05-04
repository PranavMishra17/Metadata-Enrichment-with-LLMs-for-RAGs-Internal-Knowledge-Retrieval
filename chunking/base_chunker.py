from abc import ABC, abstractmethod
import json
import os
import uuid
from datetime import datetime
from typing import List, Dict, Any, Tuple

from utils.logger import setup_logger

from gpu_utils import GPUVerifier

# Initialize GPU verification
gpu_verifier = GPUVerifier(require_gpu=True)

class BaseChunker(ABC):
    """Abstract base class for text chunking strategies."""


    
    def __init__(
        self,
        min_chunk_size: int = 2,
        max_chunk_size: int = 10,
        overlap: int = 1,
        output_dir: str = "chunk_output"
    ):
        """Initialize the base chunker."""
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        self.output_dir = output_dir
        self.logger = setup_logger(self.__class__.__name__)
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            self.logger.info(f"Created output directory: {output_dir}")
    
    def process_document(self, document_path: str) -> Dict[str, Any]:
        """Process a document and create chunks."""
        self.logger.info(f"Processing document: {document_path}")
        
        try:
            # Generate a unique document ID
            document_id = str(uuid.uuid4())
            document_name = os.path.basename(document_path)
            
            # Read document content
            with open(document_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Create chunks
            self.logger.info("Creating chunks...")
            chunks, document_metadata = self._create_chunks(text, document_id)
            
            # Remove invalid chunks
            valid_chunks = self._filter_valid_chunks(chunks)
            self.logger.info(f"Created {len(valid_chunks)} valid chunks out of {len(chunks)} total chunks")
            
            # Create result document
            result = {
                "document_id": document_id,
                "document_name": document_name,
                "chunks": valid_chunks,
                "metadata": {
                    "total_chunks": len(valid_chunks),
                    "avg_chunk_size": sum(len(chunk["text"].split()) for chunk in valid_chunks) / max(1, len(valid_chunks)),
                    "chunking_method": self.__class__.__name__,
                    "processed_at": datetime.now().isoformat(),
                    **document_metadata
                }
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing document {document_path}: {str(e)}")
            raise
    
    
    def save_chunks(self, result: Dict[str, Any]) -> str:
            """Save chunks to a JSON file."""
            try:
                document_id = result["document_id"]
                document_name = result["document_name"]
                chunking_method = result["metadata"]["chunking_method"]
                
                # Create a more descriptive filename with chunking parameters
                base_name = os.path.splitext(document_name)[0]
                method_name = self.__class__.__name__.lower().replace('chunker', '')
                
                # Add specific parameters based on chunker type
                params = ""
                if method_name == "semantic":
                    params = f"_p{getattr(self, 'percentile_threshold', 95)}"
                elif method_name == "recursive":
                    params = f"_{getattr(self, 'split_method', 'length')}{getattr(self, 'max_chunk_length', 1000)}"
                elif method_name == "naive":
                    params = f"_{getattr(self, 'chunk_by', 'paragraph')}"
                
                # Construct the filename with meaningful parameters
                filename = f"{base_name}_{method_name}{params}_chunks.json"
                output_path = os.path.join(self.output_dir, filename)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2)
                
                self.logger.info(f"Saved chunks to {output_path}")
                return output_path
            
            except Exception as e:
                self.logger.error(f"Error saving chunks: {str(e)}")
                raise


    def _filter_valid_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out invalid chunks (null or incoherent)."""
        valid_chunks = []
        for chunk in chunks:
            # Skip empty chunks
            if not chunk["text"] or len(chunk["text"].strip()) == 0:
                self.logger.warning(f"Skipping empty chunk: {chunk['chunk_id']}")
                continue
            
            # Skip chunks with too few sentences
            sentences = chunk["text"].split('. ')
            if len(sentences) < self.min_chunk_size:
                self.logger.warning(f"Skipping too small chunk: {chunk['chunk_id']} (only {len(sentences)} sentences)")
                continue
            
            valid_chunks.append(chunk)
            
        return valid_chunks
    
    @abstractmethod
    def _create_chunks(self, text: str, document_id: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Abstract method to create chunks from text."""
        pass
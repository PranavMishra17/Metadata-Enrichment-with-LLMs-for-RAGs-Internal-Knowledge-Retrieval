import os
import json
import glob
import numpy as np
import faiss
import pickle
from typing import List, Dict, Any, Tuple
from abc import ABC, abstractmethod
import time
import shutil

from utils.logger import setup_logger

from gpu_utils import GPUVerifier

# Initialize GPU verification
gpu_verifier = GPUVerifier(require_gpu=True)

class BaseEmbedder(ABC):
    """Base class for all embedding strategies."""
    
    def __init__(
        self,
        input_dir: str = "metadata_gen_output",
        output_dir: str = "embeddings_output",
        chunking_type: str = "semantic",
        model_name: str = "Snowflake/arctic-embed-s"
    ):
        """Initialize the base embedder.
        
        Args:
            input_dir: Input directory containing enriched chunks
            output_dir: Output directory for embeddings
            chunking_type: Type of chunking (semantic, naive, recursive)
            model_name: Name of the embedding model to use
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.chunking_type = chunking_type
        self.model_name = model_name
        self.logger = setup_logger(self.__class__.__name__)
        
        # Create necessary directories
        self.input_chunking_dir = os.path.join(input_dir, f"{chunking_type}_chunks_metadata")
        self.output_chunking_dir = os.path.join(output_dir, chunking_type)
        self.embedding_type_dir = self._get_embedding_type_dir()
        
        self._create_directories()
        
        # Will be set during initialization
        self.model = None
        self.vectorizer = None
    
    @abstractmethod
    def _get_embedding_type_dir(self) -> str:
        """Get the directory name for this embedding type."""
        pass
    
    def _create_directories(self):
        """Create necessary directories."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.output_chunking_dir, exist_ok=True)
        os.makedirs(self.embedding_type_dir, exist_ok=True)
        self.logger.info(f"Created output directory: {self.embedding_type_dir}")
    
    def process_all_chunks(self):
        """Process all enriched chunks and create embeddings."""
        self.logger.info(f"Processing chunks from {self.input_chunking_dir}")
        
        # Find all JSON files
        json_files = glob.glob(os.path.join(self.input_chunking_dir, "*_enriched_chunks.json"))
        if not json_files:
            self.logger.warning(f"No enriched chunks found in {self.input_chunking_dir}")
            return
        
        self.logger.info(f"Found {len(json_files)} files to process")
        
        # Initialize model
        self._initialize_model()
        
        # Collect all chunks from all files
        all_chunks = []
        document_names = []
        
        for json_file in json_files:
            try:
                doc_name = os.path.basename(json_file).replace("_enriched_chunks.json", "")
                document_names.append(doc_name)
                
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                chunks = data.get("chunks", [])
                if not chunks:
                    self.logger.warning(f"No chunks found in {json_file}")
                    continue
                
                all_chunks.extend(chunks)
                self.logger.info(f"Loaded {len(chunks)} chunks from {doc_name}")
                
            except Exception as e:
                self.logger.error(f"Error processing {json_file}: {str(e)}")
        
        if not all_chunks:
            self.logger.warning("No chunks to process")
            return
        
        self.logger.info(f"Processing {len(all_chunks)} chunks total")
        
        # Generate embeddings
        embeddings, chunk_ids, chunk_metadata = self._generate_embeddings(all_chunks)
        
        # Create FAISS index
        index_info = self._create_faiss_index(embeddings, chunk_ids, chunk_metadata)
        
        # Save document list
        doc_list_path = os.path.join(self.embedding_type_dir, "document_list.json")
        with open(doc_list_path, 'w', encoding='utf-8') as f:
            json.dump({
                "documents": document_names,
                "total_chunks": len(all_chunks),
                "chunking_type": self.chunking_type,
                "embedding_type": self._get_embedding_type(),
                "model_name": self.model_name,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }, f, indent=2)
        
        self.logger.info(f"Successfully created embeddings for {len(all_chunks)} chunks")
        return index_info
    
    @abstractmethod
    def _initialize_model(self):
        """Initialize the embedding model."""
        pass
    
    @abstractmethod
    def _generate_embeddings(self, chunks: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[str], List[Dict[str, Any]]]:
        """Generate embeddings for chunks.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Tuple of (embeddings, chunk_ids, chunk_metadata)
        """
        pass
    
    @abstractmethod
    def _get_embedding_type(self) -> str:
        """Get a string representing the embedding type."""
        pass
    
    def _create_faiss_index(self, embeddings: np.ndarray, chunk_ids: List[str], chunk_metadata: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a FAISS index for the embeddings.
        
        Args:
            embeddings: Numpy array of embeddings
            chunk_ids: List of chunk IDs
            chunk_metadata: List of chunk metadata
            
        Returns:
            Dictionary with index information
        """
        self.logger.info(f"Creating FAISS index with {len(embeddings)} embeddings")
        
        # Normalize embeddings for cosine similarity
        normalized_embeddings = self._normalize_embeddings(embeddings)
        
        # Create FAISS index
        dimension = normalized_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        index.add(normalized_embeddings)
        
        # Create ID mappings
        id_to_index = {chunk_id: i for i, chunk_id in enumerate(chunk_ids)}
        index_to_id = {i: chunk_id for i, chunk_id in enumerate(chunk_ids)}
        
        # Store metadata mapping
        id_to_metadata = {chunk_id: meta for chunk_id, meta in zip(chunk_ids, chunk_metadata)}
        
        # Save artifacts
        index_path = os.path.join(self.embedding_type_dir, "index.faiss")
        mapping_path = os.path.join(self.embedding_type_dir, "id_mapping.pkl")
        metadata_path = os.path.join(self.embedding_type_dir, "metadata.json")
        
        faiss.write_index(index, index_path)
        
        with open(mapping_path, "wb") as f:
            pickle.dump({"id_to_index": id_to_index, "index_to_id": index_to_id}, f)
        
        with open(metadata_path, "w", encoding='utf-8') as f:
            json.dump(id_to_metadata, f, indent=2)
        
        self.logger.info(f"Saved FAISS index to {index_path}")
        self.logger.info(f"Saved ID mappings to {mapping_path}")
        self.logger.info(f"Saved metadata to {metadata_path}")
        
        index_info = {
            "index": index,
            "id_to_index": id_to_index,
            "index_to_id": index_to_id,
            "id_to_metadata": id_to_metadata,
            "dimension": dimension,
            "num_vectors": len(embeddings)
        }
        
        return index_info
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings for cosine similarity.
        
        Args:
            embeddings: Numpy array of embeddings
            
        Returns:
            Normalized embeddings
        """
        # Calculate norms
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Handle zero norms to avoid division by zero
        norms[norms == 0] = 1.0
        
        # Normalize
        normalized = embeddings / norms
        
        return normalized.astype(np.float32)  # FAISS requires float32
import os
import json
import pickle
import faiss
import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from abc import ABC, abstractmethod

from utils.logger import setup_logger
from gpu_utils import GPUVerifier
# Initialize GPU verification
gpu_verifier = GPUVerifier(require_gpu=True)

class BaseRetriever(ABC):
    """Base class for all retrievers."""
    
    def __init__(
        self,
        embedding_dir: str = "embeddings_output",
        chunking_type: str = "semantic",
        embedding_type: str = "naive_embedding",
        model_name: str = "Snowflake/arctic-embed-s",
        top_k: int = 5
    ):
        """Initialize the base retriever.
        
        Args:
            embedding_dir: Base directory containing embeddings
            chunking_type: Type of chunking (semantic, naive, recursive)
            embedding_type: Type of embedding to use
            model_name: Name of the embedding model to use
            top_k: Number of results to retrieve
        """
        self.embedding_dir = embedding_dir
        self.chunking_type = chunking_type
        self.embedding_type = embedding_type
        self.model_name = model_name
        self.top_k = top_k
        self.logger = setup_logger(self.__class__.__name__)
        
        # Will be loaded when needed
        self.index = None
        self.id_to_index = None
        self.index_to_id = None
        self.id_to_metadata = None
        self.embedding_model = None
        
        # Load index and mappings
        self._load_resources()
    
    def _get_embedding_path(self) -> str:
        """Get the path to the embedding directory."""
        return os.path.join(self.embedding_dir, self.chunking_type, self.embedding_type)
    
    def _load_resources(self):
        """Load FAISS index and ID mappings."""
        embedding_path = self._get_embedding_path()
        
        # Load FAISS index
        index_path = os.path.join(embedding_path, "index.faiss")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        
        try:
            self.index = faiss.read_index(index_path)
            self.logger.info(f"Loaded FAISS index from {index_path}")
        except Exception as e:
            self.logger.error(f"Error loading FAISS index: {str(e)}")
            raise
        
        # Load ID mappings
        mapping_path = os.path.join(embedding_path, "id_mapping.pkl")
        if not os.path.exists(mapping_path):
            raise FileNotFoundError(f"ID mapping not found: {mapping_path}")
        
        try:
            with open(mapping_path, "rb") as f:
                mappings = pickle.load(f)
                self.id_to_index = mappings["id_to_index"]
                self.index_to_id = mappings["index_to_id"]
            self.logger.info(f"Loaded ID mappings from {mapping_path}")
        except Exception as e:
            self.logger.error(f"Error loading ID mappings: {str(e)}")
            raise
        
        # Test FAISS index size and capacity
        self.logger.info(f"FAISS index contains {self.index.ntotal} vectors")
        self.logger.info(f"FAISS index dimension: {self.index.d}")

        # Load metadata
        metadata_path = os.path.join(embedding_path, "metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.id_to_metadata = json.load(f)
            self.logger.info(f"Loaded metadata from {metadata_path}")
        except Exception as e:
            self.logger.error(f"Error loading metadata: {str(e)}")
            raise
    
    def _load_embedding_model(self):
        """Load the embedding model."""
        if self.embedding_model is None:
            try:
                self.embedding_model = SentenceTransformer(self.model_name)
                self.logger.info(f"Loaded embedding model: {self.model_name}")
            except Exception as e:
                self.logger.error(f"Error loading embedding model: {str(e)}")
                raise
    
    @abstractmethod
    def _prepare_query(self, query: str) -> np.ndarray:
        """Prepare a query for searching.
        
        Args:
            query: The query string
            
        Returns:
            A query vector
        """
        pass
    
    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve top-k relevant chunks for a query."""
        # Prepare query
        query_vector = self._prepare_query(query)
        
        # Log configured top_k
        self.logger.info(f"Searching for query with top_k={self.top_k}")
        
        # Search
        scores, indices = self.index.search(query_vector, self.top_k)
        
        # Log search results
        self.logger.info(f"FAISS search returned {len(indices[0])} results")
        
        # Prepare results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0 or idx >= len(self.index_to_id):
                self.logger.warning(f"Invalid index {idx} - skipping")
                continue  # Skip invalid indices
                
            chunk_id = self.index_to_id[idx]
            metadata = self.id_to_metadata.get(chunk_id, {})
            
            # Add retrieval info
            result = metadata.copy()
            result["score"] = float(score)
            result["rank"] = i + 1
            result["chunk_id"] = chunk_id
            
            # Make sure text field is included
            if "text" not in result or not result["text"]:
                # Try to get text from metadata
                result["text"] = metadata.get("text", "")
                if not result["text"]:
                    self.logger.warning(f"No text found for chunk {chunk_id}")
            
            results.append(result)
        
        self.logger.info(f"Returning {len(results)} results")
        
        return results
    
    def retrieve_batch(self, queries: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Retrieve top-k relevant chunks for a batch of queries.
        
        Args:
            queries: List of query strings
            
        Returns:
            Dictionary mapping queries to results
        """
        results = {}
        for query in queries:
            results[query] = self.retrieve(query)
        
        return results

    def normalize_vector(self, vec: np.ndarray) -> np.ndarray:
        """Normalize a vector to unit length.
        
        Args:
            vec: The vector to normalize
            
        Returns:
            Normalized vector
        """
        norm = np.linalg.norm(vec)
        if norm > 0:
            return vec / norm
        return vec
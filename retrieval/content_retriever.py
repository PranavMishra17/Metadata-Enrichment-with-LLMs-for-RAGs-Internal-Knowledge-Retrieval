import numpy as np
from typing import List, Dict, Any

from retrieval.base_retriever import BaseRetriever
from gpu_utils import GPUVerifier
# Initialize GPU verification
gpu_verifier = GPUVerifier(require_gpu=True)

class ContentRetriever(BaseRetriever):
    """Retriever that uses only content embeddings."""
    
    def __init__(
        self,
        embedding_dir: str = "embeddings_output",
        chunking_type: str = "semantic",
        model_name: str = "Snowflake/arctic-embed-s",
        top_k: int = 5
    ):
        """Initialize the content retriever.
        
        Args:
            embedding_dir: Base directory containing embeddings
            chunking_type: Type of chunking (semantic, naive, recursive)
            model_name: Name of the embedding model to use
            top_k: Number of results to retrieve
        """
        # Use naive embeddings for content-only retrieval
        super().__init__(
            embedding_dir=embedding_dir,
            chunking_type=chunking_type,
            embedding_type="naive_embedding",
            model_name=model_name,
            top_k=top_k
        )
    
    def _prepare_query(self, query: str) -> np.ndarray:
        """Prepare a query for searching using content embedding only.
        
        Args:
            query: The query string
            
        Returns:
            A query vector
        """
        # Load embedding model if needed
        self._load_embedding_model()
        
        # Embed query
        query_vector = self.embedding_model.encode([query])
        
        # Normalize
        query_vector = self.normalize_vector(query_vector)
        
        return query_vector
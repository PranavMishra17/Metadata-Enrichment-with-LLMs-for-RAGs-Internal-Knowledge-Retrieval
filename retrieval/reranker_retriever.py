import numpy as np
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

from retrieval.base_retriever import BaseRetriever
from gpu_utils import GPUVerifier
# Initialize GPU verification
gpu_verifier = GPUVerifier(require_gpu=True)


class RerankerRetriever:
    """Retriever that adds a reranking step to any base retriever."""
    
    def __init__(
        self,
        base_retriever: BaseRetriever,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        initial_k: int = 50
    ):
        """Initialize the reranker retriever.
        
        Args:
            base_retriever: Base retriever to use for initial retrieval
            reranker_model: Name of the reranker model to use
            initial_k: Number of initial results to retrieve before reranking
        """
        self.base_retriever = base_retriever
        self.reranker_model_name = reranker_model
        self.initial_k = initial_k
        self.logger = base_retriever.logger
        
        # Save original top_k
        self.final_k = base_retriever.top_k
        
        # Set base retriever to retrieve more results initially
        self.base_retriever.top_k = self.initial_k
        
        # Initialize reranker
        self.reranker = None
        self._load_reranker()
    
    def _load_reranker(self):
        """Load the reranker model."""
        try:
            self.reranker = CrossEncoder(self.reranker_model_name)
            self.logger.info(f"Loaded reranker model: {self.reranker_model_name}")
        except Exception as e:
            self.logger.error(f"Error loading reranker model: {str(e)}")
            raise
    
    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve and rerank chunks for a query.
        
        Args:
            query: The query string
            
        Returns:
            List of retrieved metadata with scores
        """
        # Get initial results from base retriever
        initial_results = self.base_retriever.retrieve(query)
        
        if not initial_results:
            return []
        
        # Prepare passage pairs for reranking
        passage_pairs = []
        for result in initial_results:
            text = result.get("text", "")
            if not text and "chunk_id" in result:
                # Try to get text from original metadata
                chunk_id = result["chunk_id"]
                text = self.base_retriever.id_to_metadata.get(chunk_id, {}).get("text", "")
            
            if text:
                passage_pairs.append([query, text])
        
        # Rerank with cross-encoder
        if passage_pairs:
            scores = self.reranker.predict(passage_pairs)
            
            # Add scores to results
            for i, score in enumerate(scores):
                if i < len(initial_results):
                    initial_results[i]["rerank_score"] = float(score)
            
            # Sort by reranker score
            reranked_results = sorted(initial_results, key=lambda x: x.get("rerank_score", 0), reverse=True)
            
            # Limit to final_k results
            reranked_results = reranked_results[:self.final_k]
            
            # Update ranks
            for i, result in enumerate(reranked_results):
                result["rank"] = i + 1
                
            return reranked_results
        
        # Fallback if reranking failed
        return initial_results[:self.final_k]
    
    def retrieve_batch(self, queries: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Retrieve and rerank chunks for a batch of queries.
        
        Args:
            queries: List of query strings
            
        Returns:
            Dictionary mapping queries to results
        """
        results = {}
        for query in queries:
            results[query] = self.retrieve(query)
        
        return results
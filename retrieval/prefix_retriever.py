import numpy as np
import re
from typing import List, Dict, Any

from retrieval.base_retriever import BaseRetriever
from gpu_utils import GPUVerifier
# Initialize GPU verification
gpu_verifier = GPUVerifier(require_gpu=True)


class PrefixFusionRetriever(BaseRetriever):
    """Retriever that uses prefix-fusion embeddings."""
    
    def __init__(
        self,
        embedding_dir: str = "embeddings_output",
        chunking_type: str = "semantic",
        model_name: str = "Snowflake/arctic-embed-s",
        top_k: int = 5
    ):
        """Initialize the prefix-fusion retriever.
        
        Args:
            embedding_dir: Base directory containing embeddings
            chunking_type: Type of chunking (semantic, naive, recursive)
            model_name: Name of the embedding model to use
            top_k: Number of results to retrieve
        """
        # Use prefix-fusion embeddings
        super().__init__(
            embedding_dir=embedding_dir,
            chunking_type=chunking_type,
            embedding_type="prefix_fusion_embedding",
            model_name=model_name,
            top_k=top_k
        )
    
    def _format_intent_prefix(self, query: str) -> str:
        """Extract and format intent prefix from query.
        
        Args:
            query: The query string
            
        Returns:
            Intent prefix
        """
        # Detect likely intent from query
        query_lower = query.lower()
        
        if any(q in query_lower for q in ["how to", "how do i", "steps to", "guide for"]):
            return "[Intent:HowTo]"
        elif any(q in query_lower for q in ["debug", "error", "fix", "issue", "problem", "troubleshoot"]):
            return "[Intent:Debug]"
        elif any(q in query_lower for q in ["compare", "difference", "vs", "versus", "better"]):
            return "[Intent:Compare]"
        elif any(q in query_lower for q in ["what is", "explain", "definition", "mean"]):
            return "[Intent:Reference]"
        else:
            return "[Intent:General]"
    
    def _format_service_prefix(self, query: str) -> str:
        """Extract and format service context prefix from query.
        
        Args:
            query: The query string
            
        Returns:
            Service context prefix
        """
        # Look for AWS services in query
        services = []
        aws_services = ["S3", "Glacier", "IAM", "EC2", "Lambda", "DynamoDB", "SQS", "SNS"]
        
        for service in aws_services:
            if service.lower() in query.lower() or service in query:
                services.append(service)
        
        if not services:
            return ""
        
        # Limit to 2 services
        service_str = "|".join(services[:2])
        return f"[Service:{service_str}]"
    
    def _format_content_type_prefix(self, query: str) -> str:
        """Extract and format content type prefix from query.
        
        Args:
            query: The query string
            
        Returns:
            Content type prefix
        """
        query_lower = query.lower()
        
        if any(q in query_lower for q in ["how to", "steps", "guide", "tutorial"]):
            return "[Procedural]"
        elif any(q in query_lower for q in ["what is", "explain", "concept", "definition"]):
            return "[Conceptual]"
        elif any(q in query_lower for q in ["error", "debug", "fix", "issue", "problem"]):
            return "[Troubleshooting]"
        elif any(q in query_lower for q in ["example", "sample", "code"]):
            return "[Example]"
        else:
            return "[Reference]"
    
    def _format_query_prefixes(self, query: str) -> str:
        """Format query with prefixes for embedding.
        
        Args:
            query: The query string
            
        Returns:
            Query with prefixes
        """
        # Intent prefix
        intent_prefix = self._format_intent_prefix(query)
        
        # Service context prefix
        service_prefix = self._format_service_prefix(query)
        
        # Content type prefix
        content_type_prefix = self._format_content_type_prefix(query)
        
        # Question prefix
        # Normalize: remove spaces, question marks, convert to CamelCase
        q_text = query.replace("?", "").strip()
        q_text = re.sub(r'[^a-zA-Z0-9]', ' ', q_text)  # Replace non-alphanumeric with space
        words = q_text.split()
        if words:
            q_text = words[0].lower() + ''.join(w.capitalize() for w in words[1:])
        
        # Truncate if too long
        if len(q_text) > 50:
            q_text = q_text[:50]
            
        question_prefix = f"[Q:{q_text}]"
        
        # Combine all prefixes
        all_prefixes = [intent_prefix, service_prefix, content_type_prefix, question_prefix]
        
        return " ".join([p for p in all_prefixes if p])
    
    def _prepare_query(self, query: str) -> np.ndarray:
        """Prepare a query for searching using prefix-fusion embedding.
        
        Args:
            query: The query string
            
        Returns:
            A query vector
        """
        # Load embedding model if needed
        self._load_embedding_model()
        
        # Format prefixes
        prefixed_query = self._format_query_prefixes(query) + " " + query
        
        # Embed query
        query_vector = self.embedding_model.encode([prefixed_query])
        
        # Normalize
        query_vector = self.normalize_vector(query_vector)
        
        return query_vector
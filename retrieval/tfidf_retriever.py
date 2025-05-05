import numpy as np
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer

from retrieval.base_retriever import BaseRetriever
from gpu_utils import GPUVerifier
# Initialize GPU verification
gpu_verifier = GPUVerifier(require_gpu=True)


class TFIDFRetriever(BaseRetriever):
    """Retriever that uses TF-IDF weighted embeddings with content-metadata mix."""
    
    def __init__(
        self,
        embedding_dir: str = "embeddings_output",
        chunking_type: str = "semantic",
        model_name: str = "Snowflake/arctic-embed-s",
        top_k: int = 5,
        content_weight: float = 0.7,
        tfidf_weight: float = 0.3
    ):
        """Initialize the TF-IDF retriever.
        
        Args:
            embedding_dir: Base directory containing embeddings
            chunking_type: Type of chunking (semantic, naive, recursive)
            model_name: Name of the embedding model to use
            top_k: Number of results to retrieve
            content_weight: Weight for content embeddings (default: 0.7)
            tfidf_weight: Weight for TF-IDF embeddings (default: 0.3)
        """
        # Use TF-IDF embeddings
        super().__init__(
            embedding_dir=embedding_dir,
            chunking_type=chunking_type,
            embedding_type="tfidf_embedding",
            model_name=model_name,
            top_k=top_k
        )
        
        self.content_weight = content_weight
        self.tfidf_weight = tfidf_weight
        self.vectorizer = None
        
        # Initialize TF-IDF vectorizer
        self._init_vectorizer()
    
    def _init_vectorizer(self):
        """Initialize TF-IDF vectorizer."""
        self.vectorizer = TfidfVectorizer(
            max_features=384,  # Match embedding dimensions approximately
            stop_words='english',
            ngram_range=(1, 2)  # Use both unigrams and bigrams
        )
        
        # Extract some keywords from metadata to fit the vectorizer
        sample_texts = []
        
        # Take a sample of metadata keywords to fit vectorizer
        for chunk_id, metadata in list(self.id_to_metadata.items())[:100]:
            keywords = []
            
            # Extract keywords from metadata if available
            if "keywords" in metadata:
                keywords.extend(metadata.get("keywords", []))
            
            if "entities" in metadata:
                keywords.extend(metadata.get("entities", []))
                
            if "primary_category" in metadata:
                keywords.append(metadata.get("primary_category", ""))
                
            sample_texts.append(" ".join(keywords))
        
        # Fit vectorizer if we have samples
        if sample_texts:
            self.vectorizer.fit(sample_texts)
            self.logger.info("Initialized and fitted TF-IDF vectorizer")
        else:
            self.logger.warning("No metadata samples available for fitting TF-IDF vectorizer")
    
    def _extract_query_keywords(self, query: str) -> str:
        """Extract keywords from query for TF-IDF vectorization.
        
        Args:
            query: The query string
            
        Returns:
            String of keywords
        """
        # Simple keyword extraction (could be improved with NLP)
        words = query.lower().replace("?", "").replace(",", " ").replace(".", " ").split()
        
        # Remove common words
        stopwords = {'what', 'why', 'how', 'when', 'where', 'which', 'who', 
                   'does', 'is', 'are', 'can', 'could', 'would', 'should',
                   'will', 'shall', 'may', 'might', 'must', 'have', 'has',
                   'had', 'been', 'was', 'were', 'being', 'do', 'does', 
                   'did', 'doing', 'the', 'and', 'but', 'or', 'if', 'then',
                   'that', 'this', 'these', 'those', 'for', 'with', 'about',
                   'to', 'in', 'on', 'at', 'by', 'as'}
        
        keywords = [w for w in words if len(w) > 3 and w not in stopwords]
        
        return " ".join(keywords)
    
    def _prepare_query(self, query: str) -> np.ndarray:
        """Prepare a query for searching using weighted content and TF-IDF.
        
        Args:
            query: The query string
            
        Returns:
            A query vector
        """
        # Load embedding model if needed
        self._load_embedding_model()
        
        # Generate content embedding for query
        content_embedding = self.embedding_model.encode([query])
        content_embedding = self.normalize_vector(content_embedding)
        
        # Generate TF-IDF vector for query keywords
        query_keywords = self._extract_query_keywords(query)
        if not query_keywords:
            query_keywords = query  # Fallback to full query
            
        # Transform query to TF-IDF vector
        tfidf_vector = self.vectorizer.transform([query_keywords])
        tfidf_dense = tfidf_vector.toarray()
        
        # Match dimensions if needed
        content_dim = content_embedding.shape[1]
        tfidf_dim = tfidf_dense.shape[1]
        
        if tfidf_dim > content_dim:
            # Truncate TF-IDF vector
            tfidf_resized = tfidf_dense[:, :content_dim]
        else:
            # Pad TF-IDF vector with zeros
            padding = np.zeros((tfidf_dense.shape[0], content_dim - tfidf_dim))
            tfidf_resized = np.concatenate([tfidf_dense, padding], axis=1)
        
        # Normalize TF-IDF vector
        tfidf_resized = self.normalize_vector(tfidf_resized)
        
        # Combine with weights
        combined_vector = (self.content_weight * content_embedding) + (self.tfidf_weight * tfidf_resized)
        
        # Final normalization
        combined_vector = self.normalize_vector(combined_vector)
        
        return combined_vector
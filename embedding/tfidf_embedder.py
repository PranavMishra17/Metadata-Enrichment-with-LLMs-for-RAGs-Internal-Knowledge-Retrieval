import os
import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from embedding.base_embedder import BaseEmbedder

from gpu_utils import GPUVerifier

# Initialize GPU verification
gpu_verifier = GPUVerifier(require_gpu=True)

class TfidfEmbedder(BaseEmbedder):
    """TF-IDF weighted embedding strategy that combines content and metadata keywords."""
    
    def __init__(
        self,
        input_dir: str = "metadata_gen_output",
        output_dir: str = "embeddings_output",
        chunking_type: str = "semantic",
        model_name: str = "Snowflake/arctic-embed-s",
        content_weight: float = 0.7,
        tfidf_weight: float = 0.3
    ):
        """Initialize the TF-IDF embedder.
        
        Args:
            input_dir: Input directory containing enriched chunks
            output_dir: Output directory for embeddings
            chunking_type: Type of chunking (semantic, naive, recursive)
            model_name: Name of the embedding model to use
            content_weight: Weight for content embeddings (default: 0.7)
            tfidf_weight: Weight for TF-IDF embeddings (default: 0.3)
        """
        super().__init__(input_dir, output_dir, chunking_type, model_name)
        self.content_weight = content_weight
        self.tfidf_weight = tfidf_weight
        self.vectorizer = None
    
    def _get_embedding_type_dir(self) -> str:
        """Get the directory name for TF-IDF embeddings."""
        return os.path.join(self.output_chunking_dir, "tfidf_embedding")
    
    def _get_embedding_type(self) -> str:
        """Get a string representing the embedding type."""
        return "tfidf"
    
    def _initialize_model(self):
        """Initialize the embedding model and TF-IDF vectorizer."""
        self.logger.info(f"Initializing SentenceTransformer model: {self.model_name}")
        try:
            self.model = SentenceTransformer(self.model_name)
            self.logger.info(f"Successfully loaded model: {self.model_name}")
            
            # Initialize TF-IDF vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=384,  # Match embedding dimensions approximately
                stop_words='english',
                ngram_range=(1, 2)  # Use both unigrams and bigrams
            )
            self.logger.info("Initialized TF-IDF vectorizer")
            
        except Exception as e:
            self.logger.error(f"Error initializing models: {str(e)}")
            raise
    
    def _extract_metadata_keywords(self, chunk: Dict[str, Any]) -> str:
        """Extract keywords from chunk metadata for TF-IDF vectorization.
        
        Args:
            chunk: Chunk dictionary with metadata
            
        Returns:
            String of keywords
        """
        keywords = []
        metadata = chunk.get("metadata", {})
        
        # Technical keywords (40%)
        if "content" in metadata and "keywords" in metadata["content"]:
            keywords.extend(metadata["content"]["keywords"])
        
        # Named entities (25%)
        if "content" in metadata and "entities" in metadata["content"]:
            keywords.extend(metadata["content"]["entities"])
        
        # Technical categories (20%)
        if "technical" in metadata:
            if "primary_category" in metadata["technical"]:
                keywords.append(metadata["technical"]["primary_category"])
            
            if "secondary_categories" in metadata["technical"]:
                keywords.extend(metadata["technical"]["secondary_categories"])
            
            if "mentioned_services" in metadata["technical"]:
                keywords.extend(metadata["technical"]["mentioned_services"])
        
        # Question keywords (15%)
        if "semantic" in metadata and "potential_questions" in metadata["semantic"]:
            for question in metadata["semantic"]["potential_questions"]:
                # Simple keyword extraction by removing common words
                words = question.lower().replace("?", "").replace(",", " ").replace(".", " ").split()
                question_keywords = [w for w in words if len(w) > 3 and w not in 
                                    {'what', 'why', 'how', 'when', 'where', 'which', 'who', 
                                     'does', 'is', 'are', 'can', 'could', 'would', 'should',
                                     'will', 'shall', 'may', 'might', 'must', 'have', 'has',
                                     'had', 'been', 'was', 'were', 'being', 'do', 'does', 
                                     'did', 'doing', 'the', 'and', 'but', 'or', 'if', 'then',
                                     'that', 'this', 'these', 'those', 'for', 'with', 'about'}]
                keywords.extend(question_keywords)
        
        # Get TF-IDF keywords if available (fallback)
        if "embedding_enhancement" in chunk and "tf_idf_keywords" in chunk["embedding_enhancement"]:
            keywords.extend(chunk["embedding_enhancement"]["tf_idf_keywords"])
        
        # Remove duplicates while preserving order
        unique_keywords = []
        for k in keywords:
            if k and k not in unique_keywords:
                unique_keywords.append(k)
        
        return " ".join(unique_keywords)
    
    def _generate_embeddings(self, chunks: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[str], List[Dict[str, Any]]]:
        """Generate TF-IDF weighted embeddings for chunks.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Tuple of (embeddings, chunk_ids, chunk_metadata)
        """
        self.logger.info(f"Generating TF-IDF weighted embeddings for {len(chunks)} chunks")
        
        # Extract content and IDs
        texts = [chunk.get("text", "") for chunk in chunks]
        chunk_ids = [chunk.get("chunk_id", "") for chunk in chunks]
        
        # Extract metadata keywords for each chunk
        metadata_keywords = [self._extract_metadata_keywords(chunk) for chunk in chunks]
        
        # Prepare simplified metadata for storage
        chunk_metadata = []
        for chunk in chunks:
            # Create simplified metadata to save storage space
            metadata = {
                "chunk_id": chunk.get("chunk_id", ""),
                "text": chunk.get("text", "")[:200] + "..." if len(chunk.get("text", "")) > 200 else chunk.get("text", ""),
                "document_id": chunk.get("document_id", ""),
            }
            
            # Add only essential metadata fields if available
            if "metadata" in chunk:
                if "content" in chunk["metadata"]:
                    if "content_type" in chunk["metadata"]["content"]:
                        metadata["content_type"] = chunk["metadata"]["content"].get("content_type", {}).get("primary", "Unknown")
                
                if "technical" in chunk["metadata"]:
                    metadata["primary_category"] = chunk["metadata"]["technical"].get("primary_category", "Unknown")
                
                if "semantic" in chunk["metadata"]:
                    metadata["intents"] = chunk["metadata"]["semantic"].get("intents", [])
                    metadata["summary"] = chunk["metadata"]["semantic"].get("summary", "")
            
            chunk_metadata.append(metadata)
        
        try:
            # Generate content embeddings
            self.logger.info("Generating content embeddings with SentenceTransformer...")
            content_embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32)
            self.logger.info(f"Generated content embeddings with shape {content_embeddings.shape}")
            
            # Generate TF-IDF vectors for metadata keywords
            self.logger.info("Generating TF-IDF vectors for metadata keywords...")
            tfidf_matrix = self.vectorizer.fit_transform(metadata_keywords)
            tfidf_dense = tfidf_matrix.toarray()
            self.logger.info(f"Generated TF-IDF vectors with shape {tfidf_dense.shape}")
            
            # Match dimensions between content embeddings and TF-IDF vectors
            content_dim = content_embeddings.shape[1]
            tfidf_dim = tfidf_dense.shape[1]
            
            if tfidf_dim > content_dim:
                # Truncate TF-IDF vectors
                self.logger.info(f"Truncating TF-IDF vectors from {tfidf_dim} to {content_dim} dimensions")
                tfidf_resized = tfidf_dense[:, :content_dim]
            else:
                # Pad TF-IDF vectors with zeros
                self.logger.info(f"Padding TF-IDF vectors from {tfidf_dim} to {content_dim} dimensions")
                padding = np.zeros((tfidf_dense.shape[0], content_dim - tfidf_dim))
                tfidf_resized = np.concatenate([tfidf_dense, padding], axis=1)
            
            # Normalize before combining
            content_norm = self._normalize_embeddings(content_embeddings)
            tfidf_norm = self._normalize_embeddings(tfidf_resized)
            
            # Combine content and TF-IDF embeddings with weights
            combined_embeddings = (self.content_weight * content_norm) + (self.tfidf_weight * tfidf_norm)
            
            # Final normalization
            final_embeddings = self._normalize_embeddings(combined_embeddings)
            
            self.logger.info(f"Successfully generated {len(final_embeddings)} weighted embeddings")
            
            return final_embeddings, chunk_ids, chunk_metadata
            
        except Exception as e:
            self.logger.error(f"Error generating TF-IDF weighted embeddings: {str(e)}")
            raise
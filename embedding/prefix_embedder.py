import os
import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import re

from embedding.base_embedder import BaseEmbedder

from gpu_utils import GPUVerifier

# Initialize GPU verification
gpu_verifier = GPUVerifier(require_gpu=True)

class PrefixEmbedder(BaseEmbedder):
    """Prefix-injection embedding strategy that prepends metadata to content."""
    
    def _get_embedding_type_dir(self) -> str:
        """Get the directory name for prefix embeddings."""
        return os.path.join(self.output_chunking_dir, "prefix_fusion_embedding")
    
    def _get_embedding_type(self) -> str:
        """Get a string representing the embedding type."""
        return "prefix-fusion"
    
    def _initialize_model(self):
        """Initialize the embedding model."""
        self.logger.info(f"Initializing SentenceTransformer model: {self.model_name}")
        try:
            self.model = SentenceTransformer(self.model_name)
            self.logger.info(f"Successfully loaded model: {self.model_name}")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
    
    def _format_intent(self, metadata: Dict[str, Any]) -> str:
        """Extract and format intent prefix (25% weight)."""
        intents = metadata.get("semantic", {}).get("intents", [])
        if not intents:
            return "[Intent:General]"
        
        # Get first intent and normalize
        primary_intent = intents[0].replace(" ", "").replace("-", "")
        return f"[Intent:{primary_intent}]"
    
    def _format_service_context(self, metadata: Dict[str, Any]) -> str:
        """Extract and format service context prefix (20% weight)."""
        services = metadata.get("technical", {}).get("mentioned_services", [])
        if not services:
            # Default to primary category if no services mentioned
            primary_category = metadata.get("technical", {}).get("primary_category", "General")
            return f"[Service:{primary_category}]"
        
        # Limit to 2 most important services
        service_str = "|".join(services[:2])
        return f"[Service:{service_str}]"
    
    def _format_content_type(self, metadata: Dict[str, Any]) -> str:
        """Extract and format content type prefix (15% weight)."""
        content_type = metadata.get("content", {}).get("content_type", {}).get("primary", "General")
        return f"[{content_type}]"
    
    def _format_technical_category(self, metadata: Dict[str, Any]) -> str:
        """Extract and format technical category prefix (10% weight)."""
        category = metadata.get("technical", {}).get("primary_category", "General")
        # Remove spaces for tokenization efficiency
        category = category.replace(" ", "")
        return f"[{category}]"
    
    def _format_code_presence(self, metadata: Dict[str, Any]) -> str:
        """Extract and format code presence prefix (10% weight)."""
        has_code = metadata.get("content", {}).get("has_code", False)
        if not has_code:
            return "[Code:None]"
        
        # Try to detect language from metadata
        language = "Generic"
        return f"[Code:{language}]"
    
    def _format_potential_question(self, metadata: Dict[str, Any]) -> str:
        """Extract and format potential question prefix (20% weight)."""
        questions = metadata.get("semantic", {}).get("potential_questions", [])
        if not questions:
            return ""
        
        # Use first question
        question = questions[0]
        
        # Normalize: remove spaces, question marks, convert to CamelCase
        q_text = question.replace("?", "").strip()
        
        # Convert to CamelCase
        q_text = re.sub(r'[^a-zA-Z0-9]', ' ', q_text)  # Replace non-alphanumeric with space
        words = q_text.split()
        if words:
            q_text = words[0].lower() + ''.join(w.capitalize() for w in words[1:])
        
        # Truncate if too long
        if len(q_text) > 50:
            q_text = q_text[:50]
            
        return f"[Q:{q_text}]"
    
    def _format_prefixes(self, metadata: Dict[str, Any]) -> str:
        """Format metadata as prefixes for embedding."""
        # Intent prefix (25%)
        intent_prefix = self._format_intent(metadata)
        
        # Service context prefix (20%)
        service_prefix = self._format_service_context(metadata)
        
        # Content type prefix (15%)
        content_type_prefix = self._format_content_type(metadata)
        
        # Technical category prefix (10%)
        category_prefix = self._format_technical_category(metadata)
        
        # Code presence prefix (10%)
        code_prefix = self._format_code_presence(metadata)
        
        # Potential question prefix (20%)
        question_prefix = self._format_potential_question(metadata)
        
        # Combine all prefixes
        all_prefixes = [intent_prefix, service_prefix, content_type_prefix, 
                       category_prefix, code_prefix, question_prefix]
        
        return " ".join([p for p in all_prefixes if p])
    
    def _generate_embeddings(self, chunks: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[str], List[Dict[str, Any]]]:
        """Generate embeddings for chunks using prefix injection.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Tuple of (embeddings, chunk_ids, chunk_metadata)
        """
        self.logger.info(f"Generating prefix-injected embeddings for {len(chunks)} chunks")
        
        # Extract content, IDs, and prepare augmented texts
        augmented_texts = []
        chunk_ids = []
        
        for chunk in chunks:
            # Format prefixes from metadata
            prefixes = self._format_prefixes(chunk.get("metadata", {}))
            
            # Combine prefixes with chunk content
            text = chunk.get("text", "")
            augmented_text = f"{prefixes} {text}"
            
            augmented_texts.append(augmented_text)
            chunk_ids.append(chunk.get("chunk_id", ""))
        
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
        
        # Generate embeddings
        try:
            self.logger.info("Generating embeddings with SentenceTransformer...")
            embeddings = self.model.encode(augmented_texts, show_progress_bar=True, batch_size=32)
            self.logger.info(f"Successfully generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
            
            return embeddings, chunk_ids, chunk_metadata
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {str(e)}")
            raise
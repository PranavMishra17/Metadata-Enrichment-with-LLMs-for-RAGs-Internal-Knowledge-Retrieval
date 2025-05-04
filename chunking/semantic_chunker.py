import nltk
from nltk.tokenize import sent_tokenize
from typing import List, Dict, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import uuid
import os
from datetime import datetime
import json

from chunking.base_chunker import BaseChunker

from gpu_utils import GPUVerifier
# Initialize GPU verification
gpu_verifier = GPUVerifier(require_gpu=True)

class SemanticChunker(BaseChunker):
    """Semantic chunking strategy that identifies natural semantic breakpoints in text."""


    
    def __init__(
        self,
        sentence_model: str = "all-MiniLM-L6-v2",
        percentile_threshold: float = 95,
        context_window: int = 1,
        min_chunk_size: int = 3,
        max_chunk_size: int = 10,
        overlap: int = 0,  # Not used in semantic chunking the same way
        output_dir: str = "chunk_output"
    ):
        """Initialize semantic chunker."""
        super().__init__(min_chunk_size, max_chunk_size, overlap, output_dir)
        self.sentence_model_name = sentence_model
        self.percentile_threshold = percentile_threshold
        self.context_window = context_window
        
        # Download NLTK resources if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            self.logger.info("Downloading NLTK punkt tokenizer")
            nltk.download('punkt')
        
        # Load sentence transformer model
        try:
            self.logger.info(f"Loading sentence transformer model: {sentence_model}")
            self.model = SentenceTransformer(sentence_model)
        except Exception as e:
            self.logger.error(f"Failed to load sentence transformer model: {str(e)}")
            raise
    
    def _create_chunks(self, text: str, document_id: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Create chunks from text using semantic chunking with contextual embeddings."""
        # Process text to identify semantic chunks
        self.logger.info("Processing text using semantic chunking")
        
        # Split text into sentences
        sentences = sent_tokenize(text)
        total_sentences = len(sentences)
        
        if total_sentences <= self.min_chunk_size:
            # If too few sentences, return as a single chunk
            self.logger.info(f"Text has only {total_sentences} sentences. Returning as a single chunk.")
            chunk = {
                "chunk_id": f"{document_id}_chunk_0",
                "text": text,
                "start_idx": 0,
                "end_idx": len(text),
                "metadata": {
                    "num_sentences": total_sentences,
                    "num_words": len(text.split()),
                    "chunk_type": "semantic_single"
                }
            }
            return [chunk], {"total_sentences": total_sentences, "chunking_strategy": "semantic"}
        
        # Add context to sentences for better embeddings
        contextualized_sentences = self._add_context(sentences, self.context_window)
        
        # Get sentence embeddings
        self.logger.info(f"Generating embeddings for {len(contextualized_sentences)} sentences")
        embeddings = self.model.encode(contextualized_sentences)
        
        # Calculate distances between consecutive sentences
        distances = self._calculate_distances(embeddings)
        
        # Identify breakpoints based on semantic distances
        breakpoints = self._identify_breakpoints(distances, self.percentile_threshold)
        self.logger.info(f"Identified {len(breakpoints)} semantic breakpoints")
        
        # Create initial chunks based on breakpoints
        initial_chunks = []
        
        # Track sentence positions in original text
        sentence_positions = self._find_sentence_positions(text, sentences)
        
        # Create chunks based on breakpoints
        start_idx = 0
        for breakpoint in breakpoints:
            chunk_sentences = sentences[start_idx:breakpoint + 1]
            
            if len(chunk_sentences) >= self.min_chunk_size:
                chunk_text = ' '.join(chunk_sentences)
                
                # Get text positions
                text_start_idx = sentence_positions[start_idx][0]
                text_end_idx = sentence_positions[breakpoint][1]
                
                chunk = {
                    "chunk_id": f"{document_id}_chunk_{len(initial_chunks)}",
                    "text": chunk_text,
                    "start_idx": text_start_idx,
                    "end_idx": text_end_idx,
                    "metadata": {
                        "num_sentences": len(chunk_sentences),
                        "num_words": len(chunk_text.split()),
                        "breakpoint_idx": breakpoint,
                        "chunk_type": "semantic_breakpoint"
                    }
                }
                initial_chunks.append(chunk)
            else:
                self.logger.debug(f"Skipping small chunk with {len(chunk_sentences)} sentences")
            
            start_idx = breakpoint + 1
        
        # Add the final chunk if not empty
        if start_idx < len(sentences):
            chunk_sentences = sentences[start_idx:]
            
            if len(chunk_sentences) >= self.min_chunk_size:
                chunk_text = ' '.join(chunk_sentences)
                
                # Get text positions
                text_start_idx = sentence_positions[start_idx][0]
                text_end_idx = sentence_positions[-1][1]
                
                chunk = {
                    "chunk_id": f"{document_id}_chunk_{len(initial_chunks)}",
                    "text": chunk_text,
                    "start_idx": text_start_idx,
                    "end_idx": text_end_idx,
                    "metadata": {
                        "num_sentences": len(chunk_sentences),
                        "num_words": len(chunk_text.split()),
                        "chunk_type": "semantic_final"
                    }
                }
                initial_chunks.append(chunk)
        
        # Merge small chunks with their most similar neighbors
        if len(initial_chunks) > 1:
            self.logger.info("Merging small chunks with their most similar neighbors")
            final_chunks = self._merge_small_chunks(initial_chunks)
        else:
            final_chunks = initial_chunks
        
        # Document metadata
        document_metadata = {
            "total_sentences": total_sentences,
            "total_words": len(text.split()),
            "chunking_strategy": "semantic",
            "embedding_model": self.sentence_model_name,
            "percentile_threshold": self.percentile_threshold,
            "original_breakpoints": len(breakpoints),
            "final_chunks": len(final_chunks)
        }
        
        return final_chunks, document_metadata
    
    def _add_context(self, sentences: List[str], window_size: int) -> List[str]:
        """Combine sentences with their neighbors for better context."""
        contextualized = []
        for i in range(len(sentences)):
            start = max(0, i - window_size)
            end = min(len(sentences), i + window_size + 1)
            context = ' '.join(sentences[start:end])
            contextualized.append(context)
        return contextualized
    
    def _calculate_distances(self, embeddings: np.ndarray) -> List[float]:
        """Calculate cosine distances between consecutive embeddings."""
        distances = []
        for i in range(len(embeddings) - 1):
            # Calculate cosine similarity
            similarity = np.dot(embeddings[i], embeddings[i + 1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1])
            )
            # Convert to distance (1 - similarity)
            distance = 1 - similarity
            distances.append(float(distance))
        return distances
    
    def _identify_breakpoints(self, distances: List[float], threshold_percentile: float) -> List[int]:
        """Find natural breaking points in the text based on semantic distances."""
        if not distances:
            return []
            
        threshold = np.percentile(distances, threshold_percentile)
        return [i for i, dist in enumerate(distances) if dist > threshold]
    
    def _find_sentence_positions(self, text: str, sentences: List[str]) -> List[Tuple[int, int]]:
        """Find the start and end positions of each sentence in the original text."""
        positions = []
        pos = 0
        for s in sentences:
            start = text.find(s, pos)
            end = start + len(s)
            positions.append((start, end))
            pos = end
        return positions
    
    def _merge_small_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge small chunks with their most similar neighbor."""
        if len(chunks) <= 1:
            return chunks
            
        # Extract chunk texts
        chunk_texts = [chunk["text"] for chunk in chunks]
        
        # Get embeddings for chunks
        chunk_embeddings = self.model.encode(chunk_texts)
        
        # Find chunks smaller than max_chunk_size
        small_chunk_indices = [
            i for i, chunk in enumerate(chunks) 
            if chunk["metadata"]["num_sentences"] < self.max_chunk_size and i > 0
        ]
        
        # Process small chunks
        final_chunks = chunks.copy()
        merged_indices = set()
        
        for i in small_chunk_indices:
            if i in merged_indices:
                continue
                
            # Calculate similarity with previous and next chunk
            prev_idx = i - 1
            next_idx = i + 1 if i + 1 < len(chunks) else None
            
            prev_similarity = None
            next_similarity = None
            
            if prev_idx >= 0 and prev_idx not in merged_indices:
                prev_similarity = np.dot(chunk_embeddings[i], chunk_embeddings[prev_idx]) / (
                    np.linalg.norm(chunk_embeddings[i]) * np.linalg.norm(chunk_embeddings[prev_idx])
                )
            
            if next_idx is not None and next_idx < len(chunks) and next_idx not in merged_indices:
                next_similarity = np.dot(chunk_embeddings[i], chunk_embeddings[next_idx]) / (
                    np.linalg.norm(chunk_embeddings[i]) * np.linalg.norm(chunk_embeddings[next_idx])
                )
            
            # Determine which neighbor to merge with
            if prev_similarity is not None and (next_similarity is None or prev_similarity > next_similarity):
                # Merge with previous chunk
                merged_chunk = {
                    "chunk_id": final_chunks[prev_idx]["chunk_id"],
                    "text": final_chunks[prev_idx]["text"] + " " + final_chunks[i]["text"],
                    "start_idx": final_chunks[prev_idx]["start_idx"],
                    "end_idx": final_chunks[i]["end_idx"],
                    "metadata": {
                        "num_sentences": final_chunks[prev_idx]["metadata"]["num_sentences"] + 
                                        final_chunks[i]["metadata"]["num_sentences"],
                        "num_words": len((final_chunks[prev_idx]["text"] + " " + final_chunks[i]["text"]).split()),
                        "chunk_type": "semantic_merged"
                    }
                }
                final_chunks[prev_idx] = merged_chunk
                merged_indices.add(i)
                
            elif next_similarity is not None:
                # Merge with next chunk
                merged_chunk = {
                    "chunk_id": final_chunks[i]["chunk_id"],
                    "text": final_chunks[i]["text"] + " " + final_chunks[next_idx]["text"],
                    "start_idx": final_chunks[i]["start_idx"],
                    "end_idx": final_chunks[next_idx]["end_idx"],
                    "metadata": {
                        "num_sentences": final_chunks[i]["metadata"]["num_sentences"] + 
                                        final_chunks[next_idx]["metadata"]["num_sentences"],
                        "num_words": len((final_chunks[i]["text"] + " " + final_chunks[next_idx]["text"]).split()),
                        "chunk_type": "semantic_merged"
                    }
                }
                final_chunks[i] = merged_chunk
                merged_indices.add(next_idx)
        
        # Return non-merged chunks
        result = [chunk for i, chunk in enumerate(final_chunks) if i not in merged_indices]
        
        # Re-number chunk IDs
        document_id = result[0]["chunk_id"].split("_chunk_")[0]
        for i, chunk in enumerate(result):
            chunk["chunk_id"] = f"{document_id}_chunk_{i}"
            
        return result
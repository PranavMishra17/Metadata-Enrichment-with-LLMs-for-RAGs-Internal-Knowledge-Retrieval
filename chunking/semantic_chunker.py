from gpu_utils import GPUVerifier
# Initialize GPU verification
gpu_verifier = GPUVerifier(require_gpu=True)

import nltk
from nltk.tokenize import sent_tokenize
from typing import List, Dict, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import uuid
import os
from datetime import datetime
import json

from chunking.base_chunker import BaseChunker

class SemanticChunker(BaseChunker):
    """Semantic chunking strategy that identifies natural semantic breakpoints in text.
    
    This implements the approach described in the Hugging Face model 
    "Raubachm/sentence-transformers-semantic-chunker" which:
    1. Uses sentence embeddings to represent sentence meanings
    2. Detects shifts in meaning to identify breakpoints
    3. Merges smaller chunks with semantically similar neighbors
    4. Provides adjustable parameters for tuning granularity
    """
    
    def __init__(
        self,
        sentence_model: str = "sentence-transformers/all-mpnet-base-v1",
        percentile_threshold: float = 85,  # Lowered from 95 to create more breakpoints
        context_window: int = 1,
        min_chunk_size: int = 3,
        max_chunk_size: int = 15,  # Increased to allow more reasonable chunk sizes
        overlap: int = 0,  # Not used in semantic chunking the same way
        output_dir: str = "chunk_output",
        normalize_chunk_sizes: bool = True  # Added parameter to control chunk size normalization
    ):
        """Initialize semantic chunker.
        
        Args:
            sentence_model: Name of the sentence transformer model to use
            percentile_threshold: Percentile threshold for identifying semantic breakpoints
            context_window: Number of sentences to consider on either side for context
            min_chunk_size: Minimum number of sentences in a chunk
            max_chunk_size: Maximum number of sentences in a chunk
            overlap: Not used in semantic chunking
            output_dir: Directory to save chunks
            normalize_chunk_sizes: Whether to normalize chunk sizes
        """
        super().__init__(min_chunk_size, max_chunk_size, overlap, output_dir)
        self.sentence_model_name = sentence_model
        self.percentile_threshold = percentile_threshold
        self.context_window = context_window
        self.normalize_chunk_sizes = normalize_chunk_sizes
        
        # Download NLTK resources if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            self.logger.info("Downloading NLTK punkt tokenizer")
            nltk.download('punkt')
        
        # Load sentence transformer model
        try:
            self.logger.info(f"Loading sentence transformer model: {sentence_model}")
            # Try direct model first, fall back to simpler models if needed
            try:
                self.model = SentenceTransformer(sentence_model)
            except Exception as e:
                self.logger.warning(f"Failed to load {sentence_model}, falling back to simpler model: {str(e)}")
                self.model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
        except Exception as e:
            self.logger.error(f"Failed to load sentence transformer model: {str(e)}")
            raise
    
    def _create_chunks(self, text: str, document_id: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Create chunks from text using semantic chunking with contextual embeddings."""
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
                    "chunk_type": "semantic_single",
                    "coherence_score": 1.0
                }
            }
            return [chunk], {"total_sentences": total_sentences, "chunking_strategy": "semantic"}
        
        # Find sentence positions in original text
        sentence_positions = self._find_sentence_positions(text, sentences)
        
        # Add context to sentences for better embeddings
        self.logger.info(f"Adding context to {len(sentences)} sentences with window size {self.context_window}")
        contextualized_sentences = self._add_context(sentences, self.context_window)
        
        # Get sentence embeddings
        self.logger.info(f"Generating embeddings for {len(contextualized_sentences)} sentences")
        embeddings = self.model.encode(contextualized_sentences)
        
        # Calculate distances between consecutive sentences
        distances = self._calculate_distances(embeddings)
        
        # Identify breakpoints based on semantic distances
        breakpoints = self._identify_breakpoints(distances, self.percentile_threshold)
        self.logger.info(f"Identified {len(breakpoints)} semantic breakpoints at percentile {self.percentile_threshold}")
        
        # If too few breakpoints, adjust threshold and retry
        if len(breakpoints) < 3 and total_sentences > 30:
            adjusted_threshold = max(50, self.percentile_threshold - 15)
            self.logger.info(f"Too few breakpoints. Adjusting threshold to {adjusted_threshold}")
            breakpoints = self._identify_breakpoints(distances, adjusted_threshold)
            self.logger.info(f"After adjustment: {len(breakpoints)} breakpoints")
        
        # Create initial chunks based on breakpoints
        initial_chunks = self._create_initial_chunks(sentences, breakpoints, sentence_positions, document_id)
        self.logger.info(f"Created {len(initial_chunks)} initial chunks")
        
        # If normalize_chunk_sizes is enabled, merge small chunks and split large ones
        if self.normalize_chunk_sizes and len(initial_chunks) > 1:
            final_chunks = self._normalize_chunks(initial_chunks, sentences, document_id)
            self.logger.info(f"After normalization: {len(final_chunks)} chunks")
        else:
            final_chunks = initial_chunks
        
        # Calculate coherence scores for final chunks
        final_chunks = self._calculate_chunk_coherence(final_chunks)
        
        # Document metadata
        document_metadata = {
            "total_sentences": total_sentences,
            "total_words": len(text.split()),
            "chunking_strategy": "semantic",
            "embedding_model": self.sentence_model_name,
            "percentile_threshold": self.percentile_threshold,
            "original_breakpoints": len(breakpoints),
            "final_chunks": len(final_chunks),
            "avg_chunk_size_sentences": sum(c["metadata"]["num_sentences"] for c in final_chunks) / len(final_chunks)
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
            # Calculate cosine similarity using dot product
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
        self.logger.debug(f"Distance threshold: {threshold} at percentile {threshold_percentile}")
        
        # Find all points above threshold
        breakpoint_candidates = [i for i, dist in enumerate(distances) if dist > threshold]
        
        # Ensure the breakpoints aren't too close to each other (minimum 3 sentences apart)
        filtered_breakpoints = []
        last_breakpoint = -3  # Start with a negative value to ensure first breakpoint is considered
        
        for bp in breakpoint_candidates:
            if bp - last_breakpoint >= 3:  # Minimum distance between breakpoints
                filtered_breakpoints.append(bp)
                last_breakpoint = bp
        
        return filtered_breakpoints
    
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
    
    def _create_initial_chunks(self, sentences: List[str], breakpoints: List[int], 
                               positions: List[Tuple[int, int]], document_id: str) -> List[Dict[str, Any]]:
        """Create initial chunks based on identified breakpoints."""
        chunks = []
        start_idx = 0
        
        for bp in breakpoints:
            # Create chunk from start_idx to current breakpoint
            chunk_sentences = sentences[start_idx:bp + 1]
            if len(chunk_sentences) >= self.min_chunk_size:
                chunk_text = ' '.join(chunk_sentences)
                chunk = {
                    "chunk_id": f"{document_id}_chunk_{len(chunks)}",
                    "text": chunk_text,
                    "start_idx": positions[start_idx][0],
                    "end_idx": positions[bp][1],
                    "metadata": {
                        "num_sentences": len(chunk_sentences),
                        "num_words": len(chunk_text.split()),
                        "breakpoint_idx": bp,
                        "chunk_type": "semantic_breakpoint"
                    }
                }
                chunks.append(chunk)
            
            # Update start index for next chunk
            start_idx = bp + 1
        
        # Add final chunk if needed
        if start_idx < len(sentences):
            final_sentences = sentences[start_idx:]
            if len(final_sentences) >= self.min_chunk_size:
                final_text = ' '.join(final_sentences)
                chunk = {
                    "chunk_id": f"{document_id}_chunk_{len(chunks)}",
                    "text": final_text,
                    "start_idx": positions[start_idx][0],
                    "end_idx": positions[-1][1],
                    "metadata": {
                        "num_sentences": len(final_sentences),
                        "num_words": len(final_text.split()),
                        "chunk_type": "semantic_final"
                    }
                }
                chunks.append(chunk)
        
        return chunks
    
    def _normalize_chunks(self, chunks: List[Dict[str, Any]], sentences: List[str], 
                          document_id: str) -> List[Dict[str, Any]]:
        """Normalize chunk sizes by merging small chunks and splitting large ones."""
        # Step 1: Merge small chunks with their most semantically similar neighbor
        chunk_texts = [chunk["text"] for chunk in chunks]
        chunk_embeddings = self.model.encode(chunk_texts)
        
        # Calculate pairwise similarities between all chunks
        similarities = cosine_similarity(chunk_embeddings)
        
        # Identify small chunks (less than min_chunk_size in sentences)
        small_chunks = [
            i for i, chunk in enumerate(chunks) 
            if chunk["metadata"]["num_sentences"] < self.min_chunk_size
        ]
        
        # Merge small chunks with their most similar neighbor
        merged_chunks = chunks.copy()
        merged_indices = set()
        
        for i in small_chunks:
            if i in merged_indices:
                continue
            
            # Find most similar chunk (excluding self and already merged chunks)
            sim_scores = similarities[i]
            sim_scores[i] = -1  # Exclude self
            for idx in merged_indices:
                sim_scores[idx] = -1  # Exclude already merged
            
            most_similar = np.argmax(sim_scores)
            
            # Merge with most similar chunk
            if sim_scores[most_similar] > 0:
                # Which one comes first in the text?
                first_idx = min(i, most_similar)
                second_idx = max(i, most_similar)
                
                merged_chunk = {
                    "chunk_id": merged_chunks[first_idx]["chunk_id"],
                    "text": merged_chunks[first_idx]["text"] + " " + merged_chunks[second_idx]["text"],
                    "start_idx": merged_chunks[first_idx]["start_idx"],
                    "end_idx": merged_chunks[second_idx]["end_idx"],
                    "metadata": {
                        "num_sentences": merged_chunks[first_idx]["metadata"]["num_sentences"] + 
                                        merged_chunks[second_idx]["metadata"]["num_sentences"],
                        "num_words": len((merged_chunks[first_idx]["text"] + " " + 
                                        merged_chunks[second_idx]["text"]).split()),
                        "chunk_type": "semantic_merged"
                    }
                }
                
                merged_chunks[first_idx] = merged_chunk
                merged_indices.add(second_idx)
        
        # Remove merged chunks
        normalized_chunks = [chunk for i, chunk in enumerate(merged_chunks) if i not in merged_indices]
        
        # Step 2: Split any overly large chunks (more than max_chunk_size sentences)
        final_chunks = []
        
        for chunk in normalized_chunks:
            if chunk["metadata"]["num_sentences"] > self.max_chunk_size * 1.5:  # Only split if significantly over
                # Split chunk into roughly equal parts
                chunk_sentences = sent_tokenize(chunk["text"])
                num_parts = max(2, chunk["metadata"]["num_sentences"] // self.max_chunk_size)
                part_size = len(chunk_sentences) // num_parts
                
                start = 0
                for part in range(num_parts):
                    # For last part, include all remaining sentences
                    end = len(chunk_sentences) if part == num_parts - 1 else start + part_size
                    
                    part_text = ' '.join(chunk_sentences[start:end])
                    part_chunk = {
                        "chunk_id": f"{document_id}_chunk_{len(final_chunks)}",
                        "text": part_text,
                        "start_idx": chunk["start_idx"],  # Approximate, not exact
                        "end_idx": chunk["end_idx"],      # Approximate, not exact
                        "metadata": {
                            "num_sentences": end - start,
                            "num_words": len(part_text.split()),
                            "chunk_type": "semantic_split"
                        }
                    }
                    final_chunks.append(part_chunk)
                    start = end
            else:
                # Keep chunk as is
                final_chunks.append(chunk)
        
        # Renumber chunk IDs
        for i, chunk in enumerate(final_chunks):
            chunk["chunk_id"] = f"{document_id}_chunk_{i}"
        
        return final_chunks
    
    def _calculate_chunk_coherence(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate coherence scores for each chunk."""
        for i, chunk in enumerate(chunks):
            chunk_sentences = sent_tokenize(chunk["text"])
            
            if len(chunk_sentences) < 2:
                # Single sentence is perfectly coherent with itself
                chunk["metadata"]["coherence_score"] = 1.0
                continue
            
            # Get embeddings for sentences in this chunk
            sentence_embeddings = self.model.encode(chunk_sentences)
            
            # Calculate pairwise similarities
            similarity_matrix = cosine_similarity(sentence_embeddings)
            
            # Calculate average similarity (excluding self-similarity)
            n = len(chunk_sentences)
            coherence = (similarity_matrix.sum() - n) / (n * (n - 1)) if n > 1 else 1.0
            
            # Store coherence score
            chunk["metadata"]["coherence_score"] = float(coherence)
            
        return chunks
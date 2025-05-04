from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from collections import Counter
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import os
from sentence_transformers import SentenceTransformer

from utils.logger import setup_logger

from gpu_utils import GPUVerifier

# Initialize GPU verification
gpu_verifier = GPUVerifier(require_gpu=True)

class ChunkEvaluator:
    """Evaluator for assessing and comparing chunk quality."""


    
    def __init__(self, model_name: str = "paraphrase-MiniLM-L3-v2"):
            """Initialize the chunk evaluator."""
            self.logger = setup_logger("ChunkEvaluator")
            self.model_name = model_name
            
            # Download NLTK resources if needed
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                self.logger.info("Downloading NLTK punkt tokenizer")
                nltk.download('punkt')
                
            # Load sentence transformer model for similarity calculations
            self.model = None
            try:
                self.logger.info(f"Loading sentence transformer model: {model_name}")
                self.model = SentenceTransformer(model_name)
            except Exception as e:
                self.logger.error(f"Failed to load sentence transformer model: {str(e)}")
                self.logger.warning("Will use basic TF-IDF similarity for evaluations instead")
                # Fallback - will use basic TF-IDF similarity

    def evaluate_chunks(self, chunks: List[Dict[str, Any]], plot: bool = False) -> Dict[str, Any]:
        """Evaluate chunks and return quality metrics."""
        self.logger.info(f"Evaluating {len(chunks)} chunks")
        
        # Extract chunk texts
        texts = [chunk["text"] for chunk in chunks]
        
        # Calculate basic metrics
        size_metrics = self._calculate_size_metrics(chunks)
        overlap_metrics = self._calculate_overlap_metrics(chunks, texts)
        
        # Calculate coherence metrics only if model is available
        if self.model is not None:
            coherence_metrics = self._calculate_coherence_metrics(texts, chunks)
            similarity_metrics = self._calculate_similarity_metrics(texts, plot=plot)
        else:
            coherence_metrics = {}
            similarity_metrics = {}
        
        # Combine all metrics
        metrics = {
            "total_chunks": len(chunks),
            **size_metrics,
            **overlap_metrics,
            **coherence_metrics,
            **similarity_metrics
        }
        
        return metrics
    
    def _calculate_size_metrics(self, chunks: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate size-related metrics for chunks."""
        # Get sizes
        word_counts = [len(chunk["text"].split()) for chunk in chunks]
        sentence_counts = [len(sent_tokenize(chunk["text"])) for chunk in chunks]
        char_counts = [len(chunk["text"]) for chunk in chunks]
        
        # Calculate metrics
        metrics = {
            "avg_chunk_size_words": np.mean(word_counts),
            "std_chunk_size_words": np.std(word_counts),
            "min_chunk_size_words": min(word_counts),
            "max_chunk_size_words": max(word_counts),
            "avg_chunk_size_sentences": np.mean(sentence_counts),
            "std_chunk_size_sentences": np.std(sentence_counts),
            "avg_chunk_size_chars": np.mean(char_counts)
        }
        
        return metrics
    
    def _calculate_overlap_metrics(self, chunks: List[Dict[str, Any]], texts: List[str]) -> Dict[str, float]:
        """Calculate overlap metrics between chunks."""
        # If fewer than 2 chunks, no overlap to calculate
        if len(chunks) < 2:
            return {
                "avg_jaccard_overlap": 0.0,
                "max_jaccard_overlap": 0.0,
                "total_duplicate_sentences": 0,
                "sentence_coverage": 1.0
            }
        
        # Calculate Jaccard similarity between consecutive chunks
        jaccard_similarities = []
        
        for i in range(len(texts) - 1):
            # Tokenize to words
            words1 = set(texts[i].lower().split())
            words2 = set(texts[i + 1].lower().split())
            
            # Calculate Jaccard similarity
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            if union > 0:
                jaccard_similarities.append(intersection / union)
            else:
                jaccard_similarities.append(0.0)
        
        # Count duplicate sentences
        all_sentences = []
        for text in texts:
            all_sentences.extend(sent_tokenize(text))
        
        sentence_counter = Counter(all_sentences)
        duplicate_sentences = sum(count - 1 for count in sentence_counter.values() if count > 1)
        
        # Calculate sentence coverage (how many unique sentences / total sentences in document)
        unique_sentences = len(sentence_counter)
        total_sentences = len(all_sentences)
        sentence_coverage = unique_sentences / total_sentences if total_sentences > 0 else 1.0
        
        # Calculate metrics
        metrics = {
            "avg_jaccard_overlap": np.mean(jaccard_similarities) if jaccard_similarities else 0.0,
            "max_jaccard_overlap": max(jaccard_similarities) if jaccard_similarities else 0.0,
            "total_duplicate_sentences": duplicate_sentences,
            "sentence_coverage": sentence_coverage
        }
        
        return metrics
    
    def _calculate_coherence_metrics(self, texts: List[str], chunks: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate coherence metrics for chunks."""
        # If fewer than 2 chunks, coherence is perfect
        if len(texts) < 2:
            return {
                "avg_internal_coherence": 1.0,
                "avg_external_coherence": 0.0,
                "coherence_score": 1.0
            }
        
        # Calculate internal coherence (within chunks)
        internal_coherence = []
        
        for text in texts:
            sentences = sent_tokenize(text)
            if len(sentences) < 2:
                # Perfect coherence for a single sentence
                internal_coherence.append(1.0)
                continue
            
            # Calculate TF-IDF vectors for sentences
            try:
                tfidf = TfidfVectorizer().fit_transform(sentences)
                # Calculate similarity matrix
                sim_matrix = cosine_similarity(tfidf)
                # Calculate average similarity (excluding self-similarity)
                n = len(sentences)
                coherence = (sim_matrix.sum() - n) / (n * (n - 1)) if n > 1 else 1.0
                internal_coherence.append(coherence)
            except:
                # Fallback if TF-IDF fails (e.g., empty sentences)
                internal_coherence.append(0.5)
        
        # Calculate external coherence (between chunks)
        try:
            tfidf = TfidfVectorizer().fit_transform(texts)
            sim_matrix = cosine_similarity(tfidf)
            # Calculate average external similarity (excluding self-similarity)
            n = len(texts)
            if n > 1:
                # Set diagonal to 0 to exclude self-similarity
                np.fill_diagonal(sim_matrix, 0)
                external_coherence = sim_matrix.sum() / (n * (n - 1))
            else:
                external_coherence = 0.0
        except:
            # Fallback
            external_coherence = 0.0
        
        # Calculate overall coherence score (high internal, low external is good)
        avg_internal = np.mean(internal_coherence)
        coherence_score = avg_internal - (external_coherence * 0.5)  # Penalize external coherence
        
        # Normalize to [0, 1]
        coherence_score = max(0.0, min(1.0, coherence_score))
        
        metrics = {
            "avg_internal_coherence": avg_internal,
            "avg_external_coherence": external_coherence,
            "coherence_score": coherence_score
        }
        
        return metrics
    
    def _calculate_similarity_metrics(self, texts: List[str], plot: bool = False) -> Dict[str, float]:
        """Calculate inter-chunk and intra-chunk similarity metrics using sentence embeddings."""
        if len(texts) < 2:
            return {
                "avg_inter_chunk_similarity": 0.0,
                "avg_intra_chunk_similarity": 1.0
            }
            
        # Get embeddings for each chunk
        chunk_embeddings = self.model.encode(texts)
        
        # Calculate inter-chunk similarity (between chunks)
        inter_similarities = []
        for i in range(len(chunk_embeddings)):
            for j in range(i + 1, len(chunk_embeddings)):
                # Calculate cosine similarity
                sim = np.dot(chunk_embeddings[i], chunk_embeddings[j]) / (
                    np.linalg.norm(chunk_embeddings[i]) * np.linalg.norm(chunk_embeddings[j])
                )
                inter_similarities.append(float(sim))
        
        avg_inter_similarity = np.mean(inter_similarities)
        
        # Calculate intra-chunk similarity (within chunks)
        intra_similarities = []
        for text in texts:
            sentences = sent_tokenize(text)
            if len(sentences) < 2:
                # Perfect similarity for a single sentence
                intra_similarities.append(1.0)
                continue
                
            # Get sentence embeddings
            sentence_embeddings = self.model.encode(sentences)
            
            # Calculate pairwise similarities
            pair_similarities = []
            for i in range(len(sentence_embeddings)):
                for j in range(i + 1, len(sentence_embeddings)):
                    sim = np.dot(sentence_embeddings[i], sentence_embeddings[j]) / (
                        np.linalg.norm(sentence_embeddings[i]) * np.linalg.norm(sentence_embeddings[j])
                    )
                    pair_similarities.append(float(sim))
            
            # Average similarity for this chunk
            if pair_similarities:
                intra_similarities.append(np.mean(pair_similarities))
        
        avg_intra_similarity = np.mean(intra_similarities)
        
        # Create plots if requested
        if plot:
            self._plot_similarity_distributions(inter_similarities, intra_similarities)
        
        metrics = {
            "avg_inter_chunk_similarity": avg_inter_similarity,
            "avg_intra_chunk_similarity": avg_intra_similarity,
            "min_inter_chunk_similarity": min(inter_similarities) if inter_similarities else 0.0,
            "max_inter_chunk_similarity": max(inter_similarities) if inter_similarities else 0.0,
            "min_intra_chunk_similarity": min(intra_similarities) if intra_similarities else 0.0,
            "max_intra_chunk_similarity": max(intra_similarities) if intra_similarities else 0.0
        }
        
        return metrics
    
    def _plot_similarity_distributions(self, inter_similarities: List[float], intra_similarities: List[float]) -> None:
        """Plot the distributions of inter-chunk and intra-chunk similarities."""
        # Create output directory if it doesn't exist
        output_dir = "evaluation_plots"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Plot inter-chunk similarity distribution
        plt.figure(figsize=(10, 6))
        plt.hist(inter_similarities, bins=20, color='skyblue', edgecolor='black')
        plt.axvline(np.median(inter_similarities), color='red', linestyle='dashed', linewidth=1,
                  label=f'Median: {np.median(inter_similarities):.2f}')
        plt.title('Distribution of Inter-Chunk Cosine Similarities')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'inter_chunk_similarities.png'))
        plt.close()
        
        # Plot intra-chunk similarity distribution
        plt.figure(figsize=(10, 6))
        plt.hist(intra_similarities, bins=15, color='skyblue', edgecolor='black')
        plt.axvline(np.median(intra_similarities), color='red', linestyle='dashed', linewidth=1,
                  label=f'Median: {np.median(intra_similarities):.2f}')
        plt.title('Distribution of Avg Pairwise Cosine Similarities Within Chunks')
        plt.xlabel('Average Pairwise Cosine Similarity')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'intra_chunk_similarities.png'))
        plt.close()
        
        self.logger.info(f"Saved similarity distribution plots to {output_dir}")
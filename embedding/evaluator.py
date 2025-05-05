import os
import json
import pickle
import faiss
import numpy as np
import time
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

from utils.logger import setup_logger

from gpu_utils import GPUVerifier

# Initialize GPU verification
gpu_verifier = GPUVerifier(require_gpu=True)

class EmbeddingEvaluator:
    """Evaluate the quality of embeddings."""
    
    def __init__(self, embedding_dir="embeddings_output", eval_dir="evaluation"):
        """Initialize the embedding evaluator.
        
        Args:
            embedding_dir: Directory containing embeddings
            eval_dir: Directory to save evaluation results
        """
        self.embedding_dir = embedding_dir
        self.eval_dir = eval_dir
        self.logger = setup_logger("EmbeddingEvaluator")
        
        # Create evaluation directory if it doesn't exist
        if not os.path.exists(self.eval_dir):
            os.makedirs(self.eval_dir)
            self.logger.info(f"Created evaluation directory: {self.eval_dir}")
    
    def evaluate_embeddings(self, chunking_type: str, embedding_type: str, sample_size: int = 1000):
        """Evaluate embeddings of a specific type.
        
        Args:
            chunking_type: Type of chunking (semantic, naive, recursive)
            embedding_type: Type of embedding (naive, tfidf, prefix)
            sample_size: Maximum number of samples to use for evaluation
        """
        self.logger.info(f"Evaluating {embedding_type} embeddings for {chunking_type} chunks")
        
        # Load embeddings and metadata
        embedding_type_dir = self._get_embedding_type_dir(chunking_type, embedding_type)
        if not os.path.exists(embedding_type_dir):
            self.logger.error(f"Embedding directory does not exist: {embedding_type_dir}")
            return
        
        try:
            # Load FAISS index
            index_path = os.path.join(embedding_type_dir, "index.faiss")
            if not os.path.exists(index_path):
                self.logger.error(f"FAISS index does not exist: {index_path}")
                return
            
            index = faiss.read_index(index_path)
            
            # Load ID mappings
            mapping_path = os.path.join(embedding_type_dir, "id_mapping.pkl")
            with open(mapping_path, "rb") as f:
                mappings = pickle.load(f)
                id_to_index = mappings["id_to_index"]
                index_to_id = mappings["index_to_id"]
            
            # Load metadata
            metadata_path = os.path.join(embedding_type_dir, "metadata.json")
            with open(metadata_path, "r", encoding='utf-8') as f:
                id_to_metadata = json.load(f)
            
            try:
                embeddings = self._extract_embeddings_from_index(index)
            except Exception as e:
                self.logger.error(f"Error extracting embeddings: {str(e)}")
                # Return basic info without embedding-based evaluation
                return {
                    "chunking_type": chunking_type,
                    "embedding_type": embedding_type,
                    "total_vectors": index.ntotal,
                    "embedding_dimension": index.d,
                    "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "note": "Basic evaluation only - couldn't extract vectors"
                }
            
            # Sample if embeddings are too many
            if embeddings.shape[0] > sample_size:
                self.logger.info(f"Sampling {sample_size} embeddings from {embeddings.shape[0]} total")
                indices = np.random.choice(embeddings.shape[0], sample_size, replace=False)
                sampled_embeddings = embeddings[indices]
                sampled_metadata = [id_to_metadata[index_to_id[idx]] for idx in indices]
            else:
                sampled_embeddings = embeddings
                sampled_metadata = [id_to_metadata[index_to_id[idx]] for idx in range(embeddings.shape[0])]
            
            # Evaluate embeddings
            result = {}
            
            # Calculate metadata consistency
            self.logger.info("Calculating metadata consistency...")
            consistency_score, field_scores = self._metadata_consistency(sampled_embeddings, sampled_metadata)
            
            # Calculate nearest neighbor statistics
            self.logger.info("Calculating nearest neighbor statistics...")
            nn_stats = self._nearest_neighbor_stats(sampled_embeddings, sampled_metadata)
            
            # Combine results
            result = {
                "chunking_type": chunking_type,
                "embedding_type": embedding_type,
                "sample_size": len(sampled_embeddings),
                "total_vectors": embeddings.shape[0],
                "embedding_dimension": embeddings.shape[1],
                "metadata_consistency": {
                    "overall": consistency_score,
                    "field_scores": field_scores
                },
                "nearest_neighbor_stats": nn_stats,
                "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Save results
            output_dir = os.path.join(self.eval_dir, chunking_type)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            output_path = os.path.join(output_dir, f"{embedding_type}_evaluation.json")
            with open(output_path, "w", encoding='utf-8') as f:
                json.dump(result, f, indent=2)
                
            self.logger.info(f"Saved evaluation results to {output_path}")
            
            # Generate visualizations
            self._generate_visualization(result, output_dir, embedding_type)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error evaluating embeddings: {str(e)}")
            return None
    
    def _get_embedding_type_dir(self, chunking_type: str, embedding_type: str) -> str:
        """Get the directory for a specific embedding type."""
        embedding_dir_map = {
            "naive": "naive_embedding",
            "tfidf": "tfidf_embedding",
            "prefix": "prefix_fusion_embedding"
        }
        
        embedding_dir = embedding_dir_map.get(embedding_type, embedding_type)
        return os.path.join(self.embedding_dir, chunking_type, embedding_dir)
    
    def _extract_embeddings_from_index(self, index):
        """Extract embeddings from a FAISS index."""
        try:
            # Try different methods based on FAISS version and index type
            if hasattr(index, 'xb'):
                return faiss.vector_to_array(index.xb).reshape(index.ntotal, index.d)
            elif hasattr(index, 'codes'):
                return faiss.vector_to_array(index.codes).reshape(index.ntotal, index.d)
            elif hasattr(index, 'reconstruct'):
                dimension = index.d
                num_vectors = index.ntotal
                
                # Create array for vectors
                embeddings = np.empty((num_vectors, dimension), dtype=np.float32)
                
                # Extract vectors one by one
                for i in range(num_vectors):
                    embeddings[i] = index.reconstruct(i)
                    
                return embeddings
            else:
                raise AttributeError("Index doesn't support vector extraction")
        except Exception as e:
            self.logger.error(f"Unable to extract embeddings: {str(e)}")
            raise ValueError(f"Cannot extract embeddings from index type: {type(index)}")
    
    def _metadata_consistency(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]], k: int = 5) -> Tuple[float, Dict[str, float]]:
        """Evaluate metadata consistency in the embedding space."""
        # Initialize nearest neighbors model
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(embeddings)  # +1 to include self
        
        # Get nearest neighbors for each embedding
        distances, indices = nbrs.kneighbors(embeddings)
        
        # Calculate consistency scores for different metadata fields
        consistency_scores = {}
        
        # Define metadata fields to evaluate
        metadata_fields = []
        
        # Check what fields are available in metadata
        if metadata and "content_type" in metadata[0]:
            metadata_fields.append(
                ("content_type", lambda m: m.get("content_type", "Unknown"))
            )
            
        if metadata and "primary_category" in metadata[0]:
            metadata_fields.append(
                ("primary_category", lambda m: m.get("primary_category", "Unknown"))
            )
            
        if metadata and "intents" in metadata[0]:
            metadata_fields.append(
                ("intent", lambda m: m.get("intents", ["Unknown"])[0] if m.get("intents") else "Unknown")
            )
        
        if not metadata_fields:
            self.logger.warning("No metadata fields found for consistency evaluation")
            return 0.0, {}
        
        for field_name, field_extractor in metadata_fields:
            # Extract field values
            field_values = [field_extractor(m) for m in metadata]
            
            # Calculate consistency (skip first neighbor as it's self)
            consistencies = []
            for i, neighbors in enumerate(indices):
                base_value = field_values[i]
                if base_value == "Unknown":
                    continue
                    
                neighbor_consistency = sum(field_values[n] == base_value 
                                          for n in neighbors[1:]) / k
                consistencies.append(neighbor_consistency)
            
            if consistencies:
                # Average consistency across all points
                consistency_scores[field_name] = float(np.mean(consistencies))
            else:
                consistency_scores[field_name] = 0.0
        
        # Overall consistency score (equal weights)
        if consistency_scores:
            overall_consistency = sum(consistency_scores.values()) / len(consistency_scores)
        else:
            overall_consistency = 0.0
        
        return overall_consistency, consistency_scores
    
    def _nearest_neighbor_stats(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]], k: int = 10) -> Dict[str, Any]:
        """Calculate nearest neighbor statistics."""
        # Initialize nearest neighbors model
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(embeddings)  # +1 to include self
        
        # Get nearest neighbors for each embedding
        distances, indices = nbrs.kneighbors(embeddings)
        
        # Calculate average distance to nearest neighbors (excluding self)
        avg_distances = np.mean(distances[:, 1:], axis=1)
        
        return {
            "avg_nearest_neighbor_distance": float(np.mean(avg_distances)),
            "std_nearest_neighbor_distance": float(np.std(avg_distances)),
            "min_nearest_neighbor_distance": float(np.min(avg_distances)),
            "max_nearest_neighbor_distance": float(np.max(avg_distances))
        }
    
    def _generate_visualization(self, result: Dict[str, Any], output_dir: str, embedding_type: str):
        """Generate visualizations for evaluation results."""
        try:
            # Create visualization directory
            viz_dir = os.path.join(output_dir, "visualizations")
            if not os.path.exists(viz_dir):
                os.makedirs(viz_dir)
            
            # Visualize metadata consistency
            field_scores = result.get("metadata_consistency", {}).get("field_scores", {})
            if field_scores:
                plt.figure(figsize=(10, 6))
                fields = list(field_scores.keys())
                scores = [field_scores[f] for f in fields]
                
                plt.bar(fields, scores, color='skyblue')
                plt.xlabel('Metadata Field')
                plt.ylabel('Consistency Score')
                plt.title(f'Metadata Consistency Scores - {embedding_type.capitalize()} Embedding')
                plt.ylim(0, 1)
                plt.tight_layout()
                
                output_path = os.path.join(viz_dir, f"{embedding_type}_consistency.png")
                plt.savefig(output_path)
                plt.close()
                
                self.logger.info(f"Generated consistency visualization: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {str(e)}")
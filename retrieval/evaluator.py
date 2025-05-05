import os
import json
import numpy as np
from typing import List, Dict, Any
from collections import Counter
from scipy.stats import entropy
from sklearn.metrics import ndcg_score

from utils.logger import setup_logger
from gpu_utils import GPUVerifier
# Initialize GPU verification
gpu_verifier = GPUVerifier(require_gpu=True)


class RetrievalEvaluator:
    """Evaluator for retrieval systems."""
    
    def __init__(self, output_dir: str = "retrieval_output"):
        """Initialize the retrieval evaluator.
        
        Args:
            output_dir: Directory to store evaluation results
        """
        self.output_dir = output_dir
        self.logger = setup_logger("RetrievalEvaluator")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def contextual_precision(self, results: List[Dict[str, Any]], relevant_ids: List[str], k: int = None) -> float:
        """Calculate Contextual Precision@K.
        
        Args:
            results: List of retrieved results
            relevant_ids: List of relevant chunk IDs
            k: Number of results to consider (default: all)
            
        Returns:
            Contextual Precision@K score
        """
        if not results or not relevant_ids:
            return 0.0
        
        # Use all results if k is not specified
        if k is None:
            k = len(results)
        
        # Limit to k results
        results = results[:k]
        
        # Count relevant results
        relevant_count = sum(1 for result in results if result.get("chunk_id") in relevant_ids)
        
        return relevant_count / min(k, len(results)) if results else 0.0
    
    def mean_reciprocal_rank(self, results: List[Dict[str, Any]], relevant_ids: List[str]) -> float:
        """Calculate Mean Reciprocal Rank (MRR).
        
        Args:
            results: List of retrieved results
            relevant_ids: List of relevant chunk IDs
            
        Returns:
            MRR score
        """
        if not results or not relevant_ids:
            return 0.0
        
        # Find first relevant result
        for i, result in enumerate(results):
            if result.get("chunk_id") in relevant_ids:
                return 1.0 / (i + 1)
        
        return 0.0
    
    def ndcg_at_k(self, results: List[Dict[str, Any]], relevant_ids: List[str], relevance_grades: Dict[str, int] = None, k: int = None) -> float:
        """Calculate Normalized Discounted Cumulative Gain@K.
        
        Args:
            results: List of retrieved results
            relevant_ids: List of relevant chunk IDs
            relevance_grades: Dictionary mapping chunk IDs to relevance grades (default: binary relevance)
            k: Number of results to consider (default: all)
            
        Returns:
            NDCG@K score
        """
        if not results or not relevant_ids:
            return 0.0
        
        # Use all results if k is not specified
        if k is None:
            k = len(results)
        
        # Limit to k results
        results = results[:k]
        
        # If no relevance grades provided, use binary relevance
        if relevance_grades is None:
            relevance_grades = {chunk_id: 1 for chunk_id in relevant_ids}
        
        # Convert results to relevance scores
        relevance_scores = []
        for result in results:
            chunk_id = result.get("chunk_id")
            score = relevance_grades.get(chunk_id, 0) if chunk_id in relevant_ids else 0
            relevance_scores.append(score)
        
        # Create ideal ranking (sorted by relevance)
        ideal_ranking = sorted([relevance_grades.get(chunk_id, 0) for chunk_id in relevant_ids], reverse=True)
        ideal_ranking.extend([0] * max(0, k - len(ideal_ranking)))  # Pad with zeros if needed
        
        # Calculate NDCG using sklearn
        if len(relevance_scores) < k:
            relevance_scores.extend([0] * (k - len(relevance_scores)))  # Pad with zeros if needed
            
        # Handle edge case where all scores are zero
        if all(score == 0 for score in relevance_scores) and all(score == 0 for score in ideal_ranking):
            return 1.0  # Perfect match if both are all zeros
            
        try:
            ndcg = ndcg_score([ideal_ranking], [relevance_scores], k=k)
            return float(ndcg)
        except Exception as e:
            self.logger.error(f"Error calculating NDCG: {str(e)}")
            return 0.0
    
    def recall_at_k(self, results: List[Dict[str, Any]], relevant_ids: List[str], k: int = None) -> float:
        """Calculate Recall@K.
        
        Args:
            results: List of retrieved results
            relevant_ids: List of relevant chunk IDs
            k: Number of results to consider (default: all)
            
        Returns:
            Recall@K score
        """
        if not results or not relevant_ids:
            return 0.0
        
        # Use all results if k is not specified
        if k is None:
            k = len(results)
        
        # Limit to k results
        results = results[:k]
        
        # Count relevant results
        retrieved_relevant = set(result.get("chunk_id") for result in results if result.get("chunk_id") in relevant_ids)
        
        return len(retrieved_relevant) / len(relevant_ids) if relevant_ids else 0.0
    
    def chunk_utilization(self, results: List[Dict[str, Any]]) -> float:
        """Calculate Chunk Utilization Rate.
        
        Args:
            results: List of retrieved results
            
        Returns:
            Chunk utilization rate
        """
        if not results:
            return 0.0
        
        # Count unique chunks
        unique_chunks = set(result.get("chunk_id") for result in results if "chunk_id" in result)
        
        return len(unique_chunks) / len(results) if results else 0.0
    
    def api_element_recall(self, query: str, results: List[Dict[str, Any]], expected_apis: List[str] = None) -> float:
        """Calculate API Element Recall.
        
        Args:
            query: The query string
            results: List of retrieved results
            expected_apis: List of expected API elements (default: extracted from query)
            
        Returns:
            API element recall score
        """
        if not results:
            return 0.0
        
        # Extract expected APIs from query if not provided
        if expected_apis is None:
            # Simple extraction based on common AWS API patterns
            api_patterns = [
                r'S3\.[A-Za-z]+',
                r'Glacier\.[A-Za-z]+',
                r'IAM\.[A-Za-z]+',
                r'[A-Za-z]+Bucket',
                r'[A-Za-z]+Object',
                r'[A-Za-z]+Archive',
                r'[A-Za-z]+Vault'
            ]
            
            expected_apis = []
            query_lower = query.lower()
            
            # Look for common AWS API references
            for service in ['s3', 'glacier', 'iam']:
                if service in query_lower and 'api' in query_lower:
                    expected_apis.append(service)
            
            # If no APIs found, return 1.0 (perfect score when no APIs expected)
            if not expected_apis:
                return 1.0
        
        if not expected_apis:
            return 1.0
        
        # Count found APIs
        found_apis = set()
        for result in results:
            text = result.get("text", "")
            
            # Skip if no text
            if not text:
                continue
                
            # Check for each API
            for api in expected_apis:
                if api.lower() in text.lower():
                    found_apis.add(api)
        
        return len(found_apis) / len(expected_apis) if expected_apis else 1.0
    
    def metadata_consistency(self, results: List[Dict[str, Any]]) -> float:
        """Calculate Metadata Consistency Score.
        
        Args:
            results: List of retrieved results
            
        Returns:
            Metadata consistency score
        """
        if not results:
            return 0.0
        
        # Extract primary categories
        categories = []
        for result in results:
            if "primary_category" in result:
                categories.append(result["primary_category"])
            elif "content_type" in result:
                categories.append(result["content_type"])
        
        if not categories:
            return 0.0
        
        # Calculate category distribution
        category_counts = Counter(categories)
        category_probs = [count / len(categories) for count in category_counts.values()]
        
        # Calculate entropy (lower is more consistent)
        try:
            ent = entropy(category_probs)
            # Normalize to 0-1 range (1 is most consistent)
            max_entropy = np.log(len(category_counts))
            if max_entropy > 0:
                consistency = 1 - (ent / max_entropy)
            else:
                consistency = 1.0  # Perfect consistency if only one category
        except Exception as e:
            self.logger.error(f"Error calculating entropy: {str(e)}")
            consistency = 0.0
        
        return consistency
    
    def evaluate_results(self, query: str, results: List[Dict[str, Any]], relevant_ids: List[str], k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, Any]:
        """Evaluate retrieval results for a single query.
        
        Args:
            query: The query string
            results: List of retrieved results
            relevant_ids: List of relevant chunk IDs
            k_values: List of k values to evaluate at
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {
            "query": query,
            "result_count": len(results),
            "relevant_count": len(relevant_ids)
        }
        
        # Calculate metrics at different k values
        for k in k_values:
            if k <= len(results):
                metrics[f"precision@{k}"] = self.contextual_precision(results, relevant_ids, k)
                metrics[f"recall@{k}"] = self.recall_at_k(results, relevant_ids, k)
                metrics[f"ndcg@{k}"] = self.ndcg_at_k(results, relevant_ids, k=k)
        
        # Calculate MRR
        metrics["mrr"] = self.mean_reciprocal_rank(results, relevant_ids)
        
        # Calculate AWS-specific metrics
        metrics["chunk_utilization"] = self.chunk_utilization(results)
        metrics["api_element_recall"] = self.api_element_recall(query, results)
        metrics["metadata_consistency"] = self.metadata_consistency(results)
        
        return metrics
    
    def evaluate_retriever(self, retriever_name: str, query_results: Dict[str, List[Dict[str, Any]]], query_relevance: Dict[str, List[str]], run_id: str = None) -> Dict[str, Any]:
        """Evaluate retrieval results for multiple queries.
        
        Args:
            retriever_name: Name of the retriever
            query_results: Dictionary mapping queries to results
            query_relevance: Dictionary mapping queries to relevant chunk IDs
            run_id: Unique ID for this evaluation run
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Generate run ID if not provided
        if run_id is None:
            run_id = f"{retriever_name}_{int(time.time())}"
        
        # Store all metrics
        all_metrics = {
            "run_id": run_id,
            "retriever_name": retriever_name,
            "query_count": len(query_results),
            "metrics_per_query": {},
            "aggregated_metrics": {}
        }
        
        # Evaluate each query
        k_values = [1, 3, 5, 10]
        metric_sums = {f"precision@{k}": 0.0 for k in k_values}
        metric_sums.update({f"recall@{k}": 0.0 for k in k_values})
        metric_sums.update({f"ndcg@{k}": 0.0 for k in k_values})
        metric_sums["mrr"] = 0.0
        metric_sums["chunk_utilization"] = 0.0
        metric_sums["api_element_recall"] = 0.0
        metric_sums["metadata_consistency"] = 0.0
        
        for query, results in query_results.items():
            relevant_ids = query_relevance.get(query, [])
            query_metrics = self.evaluate_results(query, results, relevant_ids, k_values)
            all_metrics["metrics_per_query"][query] = query_metrics
            
            # Accumulate metrics for averaging
            for metric, value in query_metrics.items():
                if metric in metric_sums:
                    metric_sums[metric] += value
        
        # Calculate averages
        query_count = len(query_results)
        if query_count > 0:
            for metric, total in metric_sums.items():
                all_metrics["aggregated_metrics"][metric] = total / query_count
        
        # Save results
        output_path = os.path.join(self.output_dir, f"{run_id}_evaluation.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=2)
        
        self.logger.info(f"Saved evaluation results to {output_path}")
        
        return all_metrics
    
    def compare_retrievers(self, evaluation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare different retrievers based on evaluation results.
        
        Args:
            evaluation_results: List of evaluation result dictionaries
            
        Returns:
            Dictionary with comparison results
        """
        if not evaluation_results:
            return {}
        
        # Get retriever names
        retriever_names = [results["retriever_name"] for results in evaluation_results]
        
        # Collect key metrics for comparison
        key_metrics = ["precision@5", "recall@5", "ndcg@5", "mrr"]
        
        # Create comparison table
        comparison = {
            "retrievers": retriever_names,
            "metrics": {}
        }
        
        for metric in key_metrics:
            comparison["metrics"][metric] = {
                name: results["aggregated_metrics"].get(metric, 0.0)
                for name, results in zip(retriever_names, evaluation_results)
            }
        
        # Find best retriever for each metric
        comparison["best_retrievers"] = {}
        for metric in key_metrics:
            metric_values = comparison["metrics"][metric]
            best_retriever = max(metric_values.items(), key=lambda x: x[1])
            comparison["best_retrievers"][metric] = {
                "retriever": best_retriever[0],
                "value": best_retriever[1]
            }
        
        # Overall best retriever based on average rank across metrics
        ranks = {name: 0 for name in retriever_names}
        for metric in key_metrics:
            sorted_retrievers = sorted(
                retriever_names,
                key=lambda name: comparison["metrics"][metric].get(name, 0.0),
                reverse=True
            )
            for i, name in enumerate(sorted_retrievers):
                ranks[name] += i
        
        best_overall = min(ranks.items(), key=lambda x: x[1])
        comparison["best_overall"] = {
            "retriever": best_overall[0],
            "average_rank": best_overall[1] / len(key_metrics)
        }
        
        # Save comparison
        output_path = os.path.join(self.output_dir, "retriever_comparison.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2)
        
        self.logger.info(f"Saved retriever comparison to {output_path}")
        
        return comparison
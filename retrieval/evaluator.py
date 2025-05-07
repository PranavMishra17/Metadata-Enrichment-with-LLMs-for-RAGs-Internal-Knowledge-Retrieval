import os
import json
import time
import numpy as np
from typing import List, Dict, Any
from collections import Counter
from scipy.stats import entropy
from sklearn.metrics import ndcg_score

from utils.logger import setup_logger


class RetrievalEvaluator:
    """Evaluator for retrieval systems with unsupervised metrics."""
    
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
        """Calculate Contextual Precision@K (ratio of relevant retrieved chunks / top-k chunks).
        
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
        
        # Limit to k results and ensure we don't go out of bounds
        k = min(k, len(results))
        results_subset = results[:k]
        
        # Count relevant results
        relevant_count = sum(1 for result in results_subset if result.get("chunk_id") in relevant_ids)
        
        return relevant_count / k if k > 0 else 0.0
    
    def precision_at_k(self, results: List[Dict[str, Any]], relevant_ids: List[str], k: int = None) -> float:
        """Calculate Precision@K (ratio of relevant retrieved chunks / top-k chunks).
        
        Args:
            results: List of retrieved results
            relevant_ids: List of relevant chunk IDs
            k: Number of results to consider (default: all)
            
        Returns:
            Precision@K score
        """
        return self.contextual_precision(results, relevant_ids, k)
    
    def mean_reciprocal_rank(self, results: List[Dict[str, Any]], relevant_ids: List[str]) -> float:
        """Calculate Mean Reciprocal Rank (MRR).
        
        Args:
            results: List of retrieved results
            relevant_ids: List of relevant chunk IDs
            
        Returns:
            MRR score (1/rank of first relevant result)
        """
        if not results or not relevant_ids:
            return 0.0
        
        # Find first relevant result
        for i, result in enumerate(results):
            if result.get("chunk_id") in relevant_ids:
                return 1.0 / (i + 1)
        
        return 0.0

    def ndcg_at_k(self, results: List[Dict[str, Any]], relevant_ids: List[str], 
                relevance_grades: Dict[str, int] = None, k: int = None) -> float:
        
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
        
        # Ensure k isn't larger than our results
        k = min(k, len(results))
        
        # If no relevance grades provided, use binary relevance
        if relevance_grades is None:
            relevance_grades = {chunk_id: 1 for chunk_id in relevant_ids}
        
        # Convert results to relevance scores
        relevance_scores = []
        for result in results[:k]:
            chunk_id = result.get("chunk_id")
            score = relevance_grades.get(chunk_id, 0) if chunk_id in relevant_ids else 0
            relevance_scores.append(score)
        
        # Create ideal ranking (sorted by relevance)
        ideal_ranking = sorted([relevance_grades.get(chunk_id, 0) for chunk_id in relevant_ids], reverse=True)
        
        # KEY FIX: Ensure both arrays are exactly k elements
        # Truncate ideal ranking if too long
        ideal_ranking = ideal_ranking[:k]
        # Pad with zeros if too short
        ideal_ranking = ideal_ranking + [0] * (k - len(ideal_ranking))
        
        # Pad relevance_scores with zeros if needed
        relevance_scores = relevance_scores + [0] * (k - len(relevance_scores))
        
        # Handle edge case where all scores are zero
        if all(score == 0 for score in relevance_scores) and all(score == 0 for score in ideal_ranking):
            return 1.0  # Perfect match if both are all zeros
            
        try:
            # Reshape to 2D arrays as required by sklearn
            y_true = np.array([ideal_ranking])
            y_score = np.array([relevance_scores])
            ndcg = ndcg_score(y_true, y_score)
            return float(ndcg)
        except Exception as e:
            self.logger.error(f"Error calculating NDCG: {str(e)} - reverting to manual calculation")
            # Fallback to manual calculation
            dcg = 0.0
            idcg = 0.0
            for i, (rel, ideal_rel) in enumerate(zip(relevance_scores, ideal_ranking)):
                discount = np.log2(i + 2)  # +2 because i is 0-indexed and log2(1) = 0
                dcg += rel / discount
                idcg += ideal_rel / discount
            
            return dcg / idcg if idcg > 0 else 0.0

    def recall_at_k(self, results: List[Dict[str, Any]], relevant_ids: List[str], k: int = None) -> float:
        """Calculate Recall@K (ratio of relevant retrieved chunks / all relevant chunks).
        
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
        
        return len(retrieved_relevant) / len(relevant_ids)
    
    def chunk_utilization(self, results: List[Dict[str, Any]]) -> float:
        """Calculate Chunk Utilization Rate (ratio of unique chunks to total chunks).
        
        Args:
            results: List of retrieved results
            
        Returns:
            Chunk utilization rate (0-1, higher is better)
        """
        if not results:
            return 0.0
        
        # Count unique chunks
        unique_chunks = set(result.get("chunk_id") for result in results if "chunk_id" in result)
        
        return len(unique_chunks) / len(results)
    
    def api_element_recall(self, query: str, results: List[Dict[str, Any]], expected_apis: List[str] = None) -> float:
        """Calculate API Element Recall (ratio of retrieved API elements to expected API elements).
        
        Args:
            query: The query string
            results: List of retrieved results
            expected_apis: List of expected API elements (default: extracted from query)
            
        Returns:
            API element recall score (0-1, higher is better)
        """
        if not results:
            return 0.0
        
        # Extract expected APIs from query if not provided
        if expected_apis is None:
            # Simple extraction based on common AWS API patterns
            import re
            
            # Look for specific API patterns in query
            api_patterns = [
                r'S3\.([A-Za-z]+)',
                r'Glacier\.([A-Za-z]+)',
                r'IAM\.([A-Za-z]+)',
                r'([A-Za-z]+)Bucket',
                r'([A-Za-z]+)Object',
                r'([A-Za-z]+)Item',
                r'([A-Za-z]+)Archive',
                r'([A-Za-z]+)Vault'
            ]
            
            expected_apis = []
            
            # Extract all API matches
            for pattern in api_patterns:
                matches = re.findall(pattern, query)
                expected_apis.extend(matches)
                
            # Also check for common AWS service mentions
            service_mentions = [
                'S3', 'Glacier', 'IAM', 'EC2', 'Lambda', 'DynamoDB', 'SQS', 'SNS'
            ]
            
            for service in service_mentions:
                if service in query and service not in expected_apis:
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
        
        return len(found_apis) / len(expected_apis)
    
    def metadata_consistency(self, results: List[Dict[str, Any]]) -> float:
        """Calculate Metadata Consistency Score based on entropy of categories.
        
        Args:
            results: List of retrieved results
            
        Returns:
            Metadata consistency score (0-1, higher is better - means less entropy)
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
                
        # If no categories found, try intents
        if not categories:
            for result in results:
                if "intents" in result and isinstance(result["intents"], list) and result["intents"]:
                    categories.append(result["intents"][0])  # Use first intent
        
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
    
    def retrieval_time_score(self, results: List[Dict[str, Any]]) -> float:
        """Calculate a score based on retrieval time (if available).
        
        Args:
            results: List of retrieved results
            
        Returns:
            Retrieval time score (normalized 0-1, higher is better - means faster)
        """
        # Check if results have timing information
        times = []
        for result in results:
            if "retrieval_time" in result:
                times.append(result["retrieval_time"])
                
        if not times:
            return 1.0  # Default if no timing info
            
        # Calculate average time
        avg_time = sum(times) / len(times)
        
        # Normalize to 0-1 (higher is better)
        # Using an inverse exponential function to map times to scores
        # Fast retrievals (< 100ms) get close to 1.0
        # Very slow retrievals (> 1000ms) get close to 0.0
        score = np.exp(-avg_time / 500)  # 500ms as midpoint
        
        return min(1.0, max(0.0, score))
    
    def evaluate_results(self, query: str, results: List[Dict[str, Any]], 
                        relevant_ids: List[str], k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, Any]:
        """Evaluate retrieval results for a single query.
        
        Args:
            query: The query string
            results: List of retrieved results
            relevant_ids: List of relevant chunk IDs (or pseudo-relevant)
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
            if k <= len(results) or not results:
                metrics[f"precision@{k}"] = self.precision_at_k(results, relevant_ids, k)
                metrics[f"recall@{k}"] = self.recall_at_k(results, relevant_ids, k)
                metrics[f"ndcg@{k}"] = self.ndcg_at_k(results, relevant_ids, k=k)
        
        # Calculate MRR
        metrics["mrr"] = self.mean_reciprocal_rank(results, relevant_ids)
        
        # Calculate AWS-specific metrics
        metrics["chunk_utilization"] = self.chunk_utilization(results)
        metrics["api_element_recall"] = self.api_element_recall(query, results)
        metrics["metadata_consistency"] = self.metadata_consistency(results)
        
        # Calculate retrieval time if available
        metrics["retrieval_time_score"] = self.retrieval_time_score(results)
        
        return metrics
    
    def evaluate_retriever(self, retriever_name: str, query_results: Dict[str, List[Dict[str, Any]]], 
                          query_relevance: Dict[str, List[str]], k_values: List[int] = [1, 3, 5, 10], 
                          run_id: str = None) -> Dict[str, Any]:
        """Evaluate retrieval results for multiple queries.
        
        Args:
            retriever_name: Name of the retriever
            query_results: Dictionary mapping queries to results
            query_relevance: Dictionary mapping queries to relevant chunk IDs
            k_values: List of k values to evaluate at
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
        metric_sums = {f"precision@{k}": 0.0 for k in k_values}
        metric_sums.update({f"recall@{k}": 0.0 for k in k_values})
        metric_sums.update({f"ndcg@{k}": 0.0 for k in k_values})
        metric_sums["mrr"] = 0.0
        metric_sums["chunk_utilization"] = 0.0
        metric_sums["api_element_recall"] = 0.0
        metric_sums["metadata_consistency"] = 0.0
        metric_sums["retrieval_time_score"] = 0.0
        
        for query, results in query_results.items():
            relevant_ids = query_relevance.get(query, [])
            query_metrics = self.evaluate_results(query, results, relevant_ids, k_values)
            all_metrics["metrics_per_query"][query] = query_metrics
            
            # Accumulate metrics for averaging
            for metric, value in query_metrics.items():
                if metric in metric_sums and isinstance(value, (int, float)):
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
        primary_metrics = ["precision@5", "recall@5", "ndcg@5", "mrr"]
        secondary_metrics = ["chunk_utilization", "metadata_consistency", "retrieval_time_score"]
        all_metrics = primary_metrics + secondary_metrics
        
        # Create comparison table
        comparison = {
            "retrievers": retriever_names,
            "metrics": {},
            "metric_rankings": {}
        }
        
        # Collect metric values
        for metric in all_metrics:
            comparison["metrics"][metric] = {
                name: results["aggregated_metrics"].get(metric, 0.0)
                for name, results in zip(retriever_names, evaluation_results)
            }
        
        # Generate rankings for each metric
        for metric in all_metrics:
            metric_values = comparison["metrics"][metric]
            sorted_retrievers = sorted(
                retriever_names,
                key=lambda name: metric_values.get(name, 0.0),
                reverse=True  # Higher values are better
            )
            comparison["metric_rankings"][metric] = {
                name: i+1 for i, name in enumerate(sorted_retrievers)
            }
        
        # Find best retriever for each metric
        comparison["best_retrievers"] = {}
        for metric in all_metrics:
            metric_values = comparison["metrics"][metric]
            best_retriever = max(metric_values.items(), key=lambda x: x[1])
            comparison["best_retrievers"][metric] = {
                "retriever": best_retriever[0],
                "value": best_retriever[1]
            }
        
        # Calculate overall rankings by average rank across metrics
        overall_ranks = {name: 0 for name in retriever_names}
        
        # Weight primary metrics more (2x)
        for metric in primary_metrics:
            for name in retriever_names:
                overall_ranks[name] += 2 * comparison["metric_rankings"][metric].get(name, 0)
                
        # Add secondary metrics
        for metric in secondary_metrics:
            for name in retriever_names:
                overall_ranks[name] += comparison["metric_rankings"][metric].get(name, 0)
                
        # Calculate average rank
        denominator = 2 * len(primary_metrics) + len(secondary_metrics)
        avg_ranks = {name: rank / denominator for name, rank in overall_ranks.items()}
        
        # Sort retrievers by average rank
        sorted_retrievers = sorted(
            retriever_names,
            key=lambda name: avg_ranks.get(name, float('inf'))
        )
        
        comparison["overall_ranking"] = {
            name: {"rank": i+1, "average_rank": avg_ranks.get(name, 0)}
            for i, name in enumerate(sorted_retrievers)
        }
        
        # Identify best overall retriever
        best_overall = sorted_retrievers[0] if sorted_retrievers else None
        if best_overall:
            comparison["best_overall"] = {
                "retriever": best_overall,
                "average_rank": avg_ranks.get(best_overall, 0)
            }
        
        # Calculate pairwise improvements
        comparison["pairwise_improvements"] = {}
        if len(retriever_names) > 1:
            for i, name1 in enumerate(retriever_names):
                for j, name2 in enumerate(retriever_names):
                    if i >= j:  # Skip self-comparisons and duplicates
                        continue
                        
                    key = f"{name1}_vs_{name2}"
                    improvements = {}
                    
                    for metric in all_metrics:
                        val1 = comparison["metrics"][metric].get(name1, 0)
                        val2 = comparison["metrics"][metric].get(name2, 0)
                        
                        if val1 > val2:
                            rel_improvement = (val1 - val2) / val2 if val2 > 0 else float('inf')
                            improvements[metric] = {
                                "better": name1,
                                "absolute_diff": val1 - val2,
                                "relative_improvement": min(rel_improvement, 10.0)  # Cap at 1000% for readability
                            }
                        elif val2 > val1:
                            rel_improvement = (val2 - val1) / val1 if val1 > 0 else float('inf')
                            improvements[metric] = {
                                "better": name2,
                                "absolute_diff": val2 - val1,
                                "relative_improvement": min(rel_improvement, 10.0)  # Cap at 1000% for readability
                            }
                    
                    comparison["pairwise_improvements"][key] = improvements
        
        # Save comparison
        output_path = os.path.join(self.output_dir, "retriever_comparison.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2)
        
        self.logger.info(f"Saved retriever comparison to {output_path}")
        
        return comparison
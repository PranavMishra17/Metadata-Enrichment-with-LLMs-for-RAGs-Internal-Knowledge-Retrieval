#!/usr/bin/env python3
import os
import json
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from gpu_utils import GPUVerifier

# Initialize GPU verification
gpu_verifier = GPUVerifier(require_gpu=True)



# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RetrieverEval")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Retriever Evaluation Tool")
    
    parser.add_argument("--retrieval_dir", type=str, required=True,
                        help="Directory containing retrieval results")
    parser.add_argument("--ground_truth_dir", type=str, required=True,
                        help="Directory containing ground truth data")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to store evaluation results")
    parser.add_argument("--k_values", type=int, nargs="+", default=[1, 3, 5, 10, 20],
                        help="K values for precision, recall, etc.")
    parser.add_argument("--threads", type=int, default=4,
                        help="Number of parallel threads to use")
    
    return parser.parse_args()

def normalize_retriever_name(name):
    """Normalize retriever names to handle format differences."""
    # Convert spaces in parentheses to underscores
    if " (" in name:
        name = name.replace(" (", "_(")
    # Handle other potential differences
    return name.strip()

def load_ground_truth(ground_truth_dir):
    # Same as before but store with normalized names
    ground_truth = {}
    for filename in os.listdir(ground_truth_dir):
        if filename.endswith("_ground_truth.json"):
            try:
                with open(os.path.join(ground_truth_dir, filename), 'r') as f:
                    data = json.load(f)
                    retriever_name = data.get("retriever_name")
                    if retriever_name:
                        # Normalize name for consistent lookup
                        norm_name = normalize_retriever_name(retriever_name)
                        ground_truth[norm_name] = data["ground_truth"]
                        logger.info(f"Loaded ground truth for {retriever_name}")
            except Exception as e:
                logger.error(f"Error loading {filename}: {str(e)}")
    return ground_truth

def load_retrieval_results(retrieval_dir):
    # Similar update for retrieval results
    results = {}
    for filename in os.listdir(retrieval_dir):
        if filename.endswith("_retrieval.json"):
            try:
                with open(os.path.join(retrieval_dir, filename), 'r') as f:
                    data = json.load(f)
                    retriever_name = data.get("run_info", {}).get("retriever_name")
                    if retriever_name:
                        # Normalize name
                        norm_name = normalize_retriever_name(retriever_name)
                        # Transform format
                        query_results = {}
                        for query_id, query_data in data["queries"].items():
                            if "retrieved_chunks" in query_data:
                                query_results[query_id] = query_data["retrieved_chunks"]
                        results[norm_name] = query_results
                        logger.info(f"Loaded retrieval results for {retriever_name}")
            except Exception as e:
                logger.error(f"Error loading {filename}: {str(e)}")
    return results




def precision_at_k(retrieved: List[Dict[str, Any]], ground_truth: List[Dict[str, Any]], k: int) -> float:
    """Calculate precision@k."""
    if not retrieved or not ground_truth or k <= 0:
        return 0.0
    
    k = min(k, len(retrieved))
    
    # Extract top-k chunk IDs from retrieval results
    retrieved_ids = [chunk.get("chunk_id") for chunk in retrieved[:k]]
    
    # Extract top-k chunk IDs from ground truth (these are the most relevant ones)
    relevant_ids = [chunk.get("chunk_id") for chunk in ground_truth[:k]]
    
    # Count matches
    matches = sum(1 for chunk_id in retrieved_ids if chunk_id in relevant_ids)
    
    return matches / k if k > 0 else 0.0

def recall_at_k(retrieved: List[Dict[str, Any]], ground_truth: List[Dict[str, Any]], k: int) -> float:
    """Calculate recall@k."""
    if not retrieved or not ground_truth:
        return 0.0
    
    k = min(k, len(retrieved))
    
    # Extract top-k chunk IDs from retrieval results
    retrieved_ids = [chunk.get("chunk_id") for chunk in retrieved[:k]]
    
    # Extract all chunk IDs from ground truth (all are considered relevant)
    relevant_ids = [chunk.get("chunk_id") for chunk in ground_truth]
    
    # Count matches
    retrieved_relevant = sum(1 for chunk_id in retrieved_ids if chunk_id in relevant_ids)
    
    return retrieved_relevant / len(relevant_ids) if relevant_ids else 0.0

def mean_reciprocal_rank(retrieved: List[Dict[str, Any]], ground_truth: List[Dict[str, Any]]) -> float:
    """Calculate Mean Reciprocal Rank (MRR)."""
    if not retrieved or not ground_truth:
        return 0.0
    
    # Get top ground truth chunks (most relevant ones)
    top_relevant_ids = [chunk.get("chunk_id") for chunk in ground_truth[:1]]
    
    # Find first position where a retrieved chunk is in top ground truth
    for i, chunk in enumerate(retrieved):
        if chunk.get("chunk_id") in top_relevant_ids:
            return 1.0 / (i + 1)  # MRR = 1/rank
    
    return 0.0

def ndcg_at_k(retrieved: List[Dict[str, Any]], ground_truth: List[Dict[str, Any]], k: int) -> float:
    """Calculate normalized discounted cumulative gain at k."""
    if not retrieved or not ground_truth or k <= 0:
        return 0.0
    
    k = min(k, len(retrieved))
    
    # Create a mapping of chunk_id to relevance score from ground truth
    relevance_map = {chunk.get("chunk_id"): chunk.get("score", 0.0) for chunk in ground_truth}
    
    # Calculate DCG
    dcg = 0.0
    for i, chunk in enumerate(retrieved[:k]):
        chunk_id = chunk.get("chunk_id")
        rel = relevance_map.get(chunk_id, 0.0)
        # Use log base 2 for standard NDCG calculation
        dcg += rel / np.log2(i + 2)  # +2 because i is 0-indexed and log2(1) = 0
    
    # Calculate ideal DCG
    ideal_dcg = 0.0
    sorted_relevance = sorted([chunk.get("score", 0.0) for chunk in ground_truth], reverse=True)
    for i, rel in enumerate(sorted_relevance[:k]):
        ideal_dcg += rel / np.log2(i + 2)
    
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

def metadata_consistency(chunks: List[Dict[str, Any]]) -> float:
    """Calculate metadata consistency based on category entropy."""
    if not chunks:
        return 0.0
    
    # Extract categories
    categories = []
    for chunk in chunks:
        if "primary_category" in chunk:
            categories.append(chunk["primary_category"])
        elif "content_type" in chunk:
            categories.append(chunk["content_type"])
        elif "intents" in chunk and isinstance(chunk["intents"], list) and chunk["intents"]:
            categories.append(chunk["intents"][0])
    
    if not categories:
        return 0.0
    
    # Calculate entropy
    from collections import Counter
    import numpy as np
    from scipy.stats import entropy
    
    category_counts = Counter(categories)
    category_probs = [count / len(categories) for count in category_counts.values()]
    
    try:
        ent = entropy(category_probs)
        max_entropy = np.log(len(category_counts))
        if max_entropy > 0:
            consistency = 1 - (ent / max_entropy)
        else:
            consistency = 1.0
    except Exception as e:
        logger.error(f"Error calculating entropy: {str(e)}")
        consistency = 0.0
    
    return consistency

def hit_rate_at_k(retrieved: List[Dict[str, Any]], ground_truth: List[Dict[str, Any]], k: int, threshold: float = 0.95) -> float:
    """Calculate Hit Rate@k - proportion of highly relevant docs found in top-k.
    
    Args:
        retrieved: List of retrieved chunks
        ground_truth: List of ground truth chunks with scores
        k: Number of top chunks to consider
        threshold: Score threshold to consider a document relevant (percentile)
        
    Returns:
        Hit Rate@k score
    """
    if not retrieved or not ground_truth or k <= 0:
        return 0.0
    
    k = min(k, len(retrieved))
    
    # Determine relevance threshold (95th percentile by default)
    scores = [chunk.get("score", 0.0) for chunk in ground_truth]
    if not scores:
        return 0.0
    
    threshold_score = np.percentile(scores, threshold * 100)
    
    # Get highly relevant docs from ground truth
    relevant_ids = [chunk.get("chunk_id") for chunk in ground_truth 
                   if chunk.get("score", 0.0) >= threshold_score]
    
    if not relevant_ids:
        return 0.0
    
    # Count relevant docs in top-k
    retrieved_relevant = sum(1 for chunk in retrieved[:k] 
                            if chunk.get("chunk_id") in relevant_ids)
    
    return retrieved_relevant / len(relevant_ids)


def evaluate_retriever(retriever_name: str, retrieval_results: Dict[str, List[Dict[str, Any]]], 
                      ground_truth: Dict[str, List[Dict[str, Any]]], k_values: List[int]) -> Dict[str, Any]:
    """Evaluate a single retriever across all queries and metrics."""
    metrics = {
        "retriever_name": retriever_name,
        "query_metrics": {},
        "aggregated_metrics": defaultdict(float)
    }
    
    # Track metrics for averaging
    metric_counts = defaultdict(int)
    
    # Evaluate each query
    for query_id, retrieved_chunks in retrieval_results.items():
        # Skip if no ground truth exists
        if query_id not in ground_truth:
            logger.warning(f"No ground truth for query {query_id}, retriever {retriever_name}")
            continue
        
        gt_chunks = ground_truth[query_id]
        
        query_metrics = {}
        
        # Calculate MRR
        query_metrics["mrr"] = mean_reciprocal_rank(retrieved_chunks, gt_chunks)
        metrics["aggregated_metrics"]["mrr"] += query_metrics["mrr"]
        metric_counts["mrr"] += 1
        
        # Calculate precision, recall, and NDCG at different k values
        for k in k_values:
            if k <= len(retrieved_chunks):
                p_k = precision_at_k(retrieved_chunks, gt_chunks, k)
                r_k = recall_at_k(retrieved_chunks, gt_chunks, k)
                ndcg_k = ndcg_at_k(retrieved_chunks, gt_chunks, k)
                
                query_metrics[f"precision@{k}"] = p_k
                query_metrics[f"recall@{k}"] = r_k
                query_metrics[f"ndcg@{k}"] = ndcg_k
                
                metrics["aggregated_metrics"][f"precision@{k}"] += p_k
                metrics["aggregated_metrics"][f"recall@{k}"] += r_k
                metrics["aggregated_metrics"][f"ndcg@{k}"] += ndcg_k
                
                metric_counts[f"precision@{k}"] += 1
                metric_counts[f"recall@{k}"] += 1
                metric_counts[f"ndcg@{k}"] += 1

                # Add Hit Rate
                hit_rate = hit_rate_at_k(retrieved_chunks, gt_chunks, k)
                query_metrics[f"hit_rate@{k}"] = hit_rate
                metrics["aggregated_metrics"][f"hit_rate@{k}"] += hit_rate
                metric_counts[f"hit_rate@{k}"] += 1

        
        # Calculate metadata consistency
        consistency = metadata_consistency(retrieved_chunks)
        query_metrics["metadata_consistency"] = consistency
        metrics["aggregated_metrics"]["metadata_consistency"] += consistency
        metric_counts["metadata_consistency"] += 1
        
        metrics["query_metrics"][query_id] = query_metrics
    
    # Calculate averages
    for metric, total in metrics["aggregated_metrics"].items():
        count = metric_counts[metric]
        if count > 0:
            metrics["aggregated_metrics"][metric] = total / count
    
    return metrics




def generate_comparison_tables(all_metrics: Dict[str, Dict[str, Any]], k_values: List[int]) -> Dict[str, pd.DataFrame]:
    """Generate comparison tables for different metrics."""
    tables = {}
    
    # Extract retriever names and organize by type and chunking method
    retrievers = {}
    for retriever_name in all_metrics.keys():
        if "(" in retriever_name and ")" in retriever_name:
            parts = retriever_name.split("(")
            retriever_type = parts[0].strip()
            chunking_type = parts[1].replace(")", "").strip()
            
            if retriever_type not in retrievers:
                retrievers[retriever_type] = {}
            
            retrievers[retriever_type][chunking_type] = retriever_name
    
    # Create metrics tables
    for k in k_values:
        # Precision table
        precision_data = []
        for retriever_type, chunking_methods in retrievers.items():
            row = {"Retriever": retriever_type}
            for chunking, full_name in chunking_methods.items():
                if full_name in all_metrics:
                    row[chunking] = all_metrics[full_name]["aggregated_metrics"].get(f"precision@{k}", 0.0)
            precision_data.append(row)
        
        precision_df = pd.DataFrame(precision_data)
        tables[f"precision@{k}"] = precision_df
        
        # Recall table
        recall_data = []
        for retriever_type, chunking_methods in retrievers.items():
            row = {"Retriever": retriever_type}
            for chunking, full_name in chunking_methods.items():
                if full_name in all_metrics:
                    row[chunking] = all_metrics[full_name]["aggregated_metrics"].get(f"recall@{k}", 0.0)
            recall_data.append(row)
        
        recall_df = pd.DataFrame(recall_data)
        tables[f"recall@{k}"] = recall_df
        
        # NDCG table
        ndcg_data = []
        for retriever_type, chunking_methods in retrievers.items():
            row = {"Retriever": retriever_type}
            for chunking, full_name in chunking_methods.items():
                if full_name in all_metrics:
                    row[chunking] = all_metrics[full_name]["aggregated_metrics"].get(f"ndcg@{k}", 0.0)
            ndcg_data.append(row)
        
        ndcg_df = pd.DataFrame(ndcg_data)
        tables[f"ndcg@{k}"] = ndcg_df
    
    # MRR and metadata consistency tables
    for metric in ["mrr", "metadata_consistency"]:
        metric_data = []
        for retriever_type, chunking_methods in retrievers.items():
            row = {"Retriever": retriever_type}
            for chunking, full_name in chunking_methods.items():
                if full_name in all_metrics:
                    row[chunking] = all_metrics[full_name]["aggregated_metrics"].get(metric, 0.0)
            metric_data.append(row)
        
        metric_df = pd.DataFrame(metric_data)
        tables[metric] = metric_df
    
    # Add Hit Rate tables
    for k in k_values:
        hit_rate_data = []
        for retriever_type, chunking_methods in retrievers.items():
            row = {"Retriever": retriever_type}
            for chunking, full_name in chunking_methods.items():
                if full_name in all_metrics:
                    row[chunking] = all_metrics[full_name]["aggregated_metrics"].get(f"hit_rate@{k}", 0.0)
            hit_rate_data.append(row)
        
        hit_rate_df = pd.DataFrame(hit_rate_data)
        tables[f"hit_rate@{k}"] = hit_rate_df

    return tables


def generate_visualizations(tables: Dict[str, pd.DataFrame], output_dir: str):
    """Generate visualizations from comparison tables."""
    # Create a directory for visualizations
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    sns.set(style="whitegrid")
    
    for metric, df in tables.items():
        # Skip empty dataframes
        if df.empty or "Retriever" not in df.columns:
            logger.warning(f"Skipping visualization for {metric} - empty data")
            continue
        # Set up the figure
        plt.figure(figsize=(12, 6))
        
        # Reshape data for visualization
        df_melted = df.melt(id_vars=["Retriever"], var_name="Chunking", value_name="Score")
        
        # Create barplot
        ax = sns.barplot(x="Chunking", y="Score", hue="Retriever", data=df_melted)
        
        # Add labels and title
        plt.title(f"{metric.capitalize()} by Retriever and Chunking Method")
        plt.xlabel("Chunking Method")
        plt.ylabel(f"{metric.capitalize()} Score")
        plt.legend(title="Retriever Type")
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(vis_dir, f"{metric}_comparison.png")
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        logger.info(f"Generated visualization for {metric}")
        
        # Generate heatmap
        if "Retriever" in df.columns:
            # Pivot the dataframe
            pivot_df = df.set_index("Retriever")
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt=".3f", linewidths=.5)
            
            plt.title(f"{metric.capitalize()} Heatmap by Retriever and Chunking Method")
            plt.tight_layout()
            
            # Save heatmap
            heatmap_path = os.path.join(vis_dir, f"{metric}_heatmap.png")
            plt.savefig(heatmap_path, dpi=300)
            plt.close()
            
            logger.info(f"Generated heatmap for {metric}")

def main():
    args = parse_arguments()
    
    # Set output directory if not specified
    if args.output_dir is None:
        args.output_dir = os.path.join(args.retrieval_dir, "evaluation")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load ground truth and retrieval results
    ground_truth = load_ground_truth(args.ground_truth_dir)
    retrieval_results = load_retrieval_results(args.retrieval_dir)
    
    if not ground_truth or not retrieval_results:
        logger.error("No ground truth or retrieval results found, exiting")
        return
    
    # Evaluate all retrievers
    all_metrics = {}
    
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        future_to_retriever = {}
        
        for retriever_name, results in retrieval_results.items():
            # Skip if no ground truth exists
            if retriever_name not in ground_truth:
                logger.warning(f"No ground truth for retriever {retriever_name}")
                continue
            
            future = executor.submit(
                evaluate_retriever,
                retriever_name,
                results,
                ground_truth[retriever_name],
                args.k_values
            )
            future_to_retriever[future] = retriever_name
        
        for future in as_completed(future_to_retriever):
            retriever_name = future_to_retriever[future]
            try:
                metrics = future.result()
                all_metrics[retriever_name] = metrics
                logger.info(f"Completed evaluation for {retriever_name}")
                
                # Save individual metrics
                output_path = os.path.join(args.output_dir, f"{retriever_name}_metrics.json")
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2)
            except Exception as e:
                logger.error(f"Error evaluating {retriever_name}: {str(e)}")
    
    # Generate comparison tables
    comparison_tables = generate_comparison_tables(all_metrics, args.k_values)
    
    # Save tables
    tables_dir = os.path.join(args.output_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    
    for metric, df in comparison_tables.items():
        # Save as CSV
        csv_path = os.path.join(tables_dir, f"{metric}_comparison.csv")
        df.to_csv(csv_path, index=False)
        
        # Save as HTML
        html_path = os.path.join(tables_dir, f"{metric}_comparison.html")
        df.to_html(html_path, index=False)
        
        logger.info(f"Saved comparison table for {metric}")
    
    # Generate visualizations
    generate_visualizations(comparison_tables, args.output_dir)
    
    # Save summary of all metrics
    summary = {
        "retrievers": list(all_metrics.keys()),
        "metrics": args.k_values,
        "output_path": args.output_dir
    }
    
    summary_path = os.path.join(args.output_dir, "evaluation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Evaluation completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
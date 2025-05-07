#!/usr/bin/env python3
import os
import json
import argparse
import time
import numpy as np
from collections import defaultdict
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from retrieval.evaluator import RetrievalEvaluator
from utils.logger import setup_logger

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Retrieval evaluation tool")
    
    # Input/output arguments
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing retrieval results (run folder)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to store evaluation results (default: input_dir/evaluation)")
    parser.add_argument("--relevance_file", type=str, required=False,
                        help="JSON file with relevance judgments (optional)")
    
    # Evaluation options
    parser.add_argument("--retrievers", type=str, nargs="+", default=None,
                        help="Specific retrievers to evaluate (default: all)")
    parser.add_argument("--k_values", type=int, nargs="+", default=[1, 3, 5, 10, 20],
                        help="K values for evaluation metrics (default: 1, 3, 5, 10, 20)")
    
    # Pseudo-relevance options
    parser.add_argument("--relevance_threshold", type=float, default=0.8,
                        help="Score threshold for pseudo-relevance labeling (default: 0.8)")
    parser.add_argument("--relevance_top_k", type=float, default=0.2,
                        help="Percentage of top results to consider relevant (default: 0.2)")
    parser.add_argument("--relevance_method", type=str, default="threshold",
                        choices=["threshold", "percentile", "reranker", "combined"],
                        help="Method for determining pseudo-relevance (default: threshold)")
    
    # Performance options
    parser.add_argument("--threads", type=int, default=4,
                        help="Number of parallel threads to use")
    
    return parser.parse_args()

def group_retrieval_files(input_dir: str) -> Dict[str, Dict[str, str]]:
    """Group retrieval files by retriever type.
    
    Args:
        input_dir: Directory containing retrieval results
        
    Returns:
        Dictionary mapping retriever types to file paths for results and retrieval files
    """
    logger = setup_logger("FileGrouper")
    
    # Check if directory exists
    if not os.path.exists(input_dir):
        logger.error(f"Input directory not found: {input_dir}")
        return {}
    
    # Find all result files
    retriever_groups = defaultdict(dict)
    
    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        
        # Skip directories and non-JSON files
        if os.path.isdir(file_path) or not file.endswith('.json'):
            continue
            
        # Group by retriever name pattern
        if file.endswith('_results.json'):
            # Extract retriever name from filename (e.g., "Content_(naive)")
            retriever_name = file.replace('_results.json', '')
            retriever_groups[retriever_name]['results'] = file_path
            
        elif file.endswith('_retrieval.json'):
            # Extract retriever name from filename
            retriever_name = file.replace('_retrieval.json', '')
            retriever_groups[retriever_name]['retrieval'] = file_path
    
    # Check that each group has both files
    complete_groups = {}
    for retriever_name, files in retriever_groups.items():
        if 'results' in files and 'retrieval' in files:
            complete_groups[retriever_name] = files
            logger.info(f"Found complete file group for {retriever_name}")
        else:
            logger.warning(f"Incomplete file group for {retriever_name}")
    
    logger.info(f"Grouped {len(complete_groups)} complete retriever file sets")
    return complete_groups

def load_retrieval_data(file_groups: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, Any]]:
    """Load retrieval data from grouped files.
    
    Args:
        file_groups: Dictionary mapping retriever names to file paths
        
    Returns:
        Dictionary mapping retriever names to retrieval data
    """
    logger = setup_logger("DataLoader")
    
    retrieval_data = {}
    
    for retriever_name, files in file_groups.items():
        try:
            # Load retrieval data (contains run_info and structured results)
            with open(files['retrieval'], 'r', encoding='utf-8') as f:
                retrieval_info = json.load(f)
            
            # Load results data (contains raw results)
            with open(files['results'], 'r', encoding='utf-8') as f:
                results_data = json.load(f)
            
            # Combine both data sets
            retrieval_data[retriever_name] = {
                'info': retrieval_info.get('run_info', {}),
                'queries': retrieval_info.get('queries', {}),
                'results': results_data
            }
            
            logger.info(f"Loaded data for {retriever_name}")
        except Exception as e:
            logger.error(f"Error loading data for {retriever_name}: {str(e)}")
    
    return retrieval_data

def load_relevance_judgments(relevance_file: str, queries: List[str]) -> Dict[str, List[str]]:
    """Load relevance judgments from a JSON file.
    
    Args:
        relevance_file: Path to relevance judgments file
        queries: List of query IDs to ensure coverage
        
    Returns:
        Dictionary mapping query IDs to relevant chunk IDs
    """
    logger = setup_logger("RelevanceLoader")
    
    relevance = {}
    
    # Initialize with empty relevance for all queries
    for query_id in queries:
        relevance[query_id] = []
    
    if not relevance_file or not os.path.exists(relevance_file):
        logger.warning("No relevance judgments file provided or file not found")
        logger.info("Will use pseudo-relevance labeling for evaluation")
        return relevance
    
    try:
        with open(relevance_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Support different formats
        if isinstance(data, dict):
            for query_id, judgments in data.items():
                if isinstance(judgments, list):
                    relevance[query_id] = judgments
                elif isinstance(judgments, dict) and "relevant_ids" in judgments:
                    relevance[query_id] = judgments["relevant_ids"]
                    
        # Fill in any missing queries with empty relevance
        for query_id in queries:
            if query_id not in relevance:
                relevance[query_id] = []
        
        logger.info(f"Loaded relevance judgments for {len(relevance)} queries")
    except Exception as e:
        logger.error(f"Error loading relevance judgments: {str(e)}")
        logger.info("Will use pseudo-relevance labeling for evaluation")
    
    return relevance

def generate_pseudo_relevance_labels(results: List[Dict[str, Any]], 
                                    method: str = "threshold",
                                    threshold: float = 0.8,
                                    top_k_percent: float = 0.2) -> List[str]:
    """Generate pseudo-relevance labels for a set of retrieval results.
    
    Args:
        results: List of retrieval results with scores
        method: Method for determining relevance ('threshold', 'percentile', 'reranker', 'combined')
        threshold: Score threshold for relevance (used with 'threshold' method)
        top_k_percent: Percentage of top results to consider relevant (used with 'percentile' method)
        
    Returns:
        List of chunk IDs deemed relevant
    """
    if not results:
        return []
    
    # Extract scores and chunk IDs
    scores = []
    chunk_ids = []
    
    for result in results:
        chunk_id = result.get("chunk_id")
        if not chunk_id:
            continue
            
        if method == "reranker" and "rerank_score" in result:
            score = result.get("rerank_score", 0.0)
        elif method == "combined" and "rerank_score" in result:
            # Combine base score and reranker score (with more weight on reranker)
            base_score = result.get("score", 0.0)
            rerank_score = result.get("rerank_score", 0.0)
            score = 0.3 * base_score + 0.7 * rerank_score
        else:
            # Default to retriever score
            score = result.get("score", 0.0)
        
        scores.append(score)
        chunk_ids.append(chunk_id)
    
    # Create relevance labels based on method
    relevant_ids = []
    
    if method == "threshold":
        # Label chunks with scores above threshold as relevant
        relevant_ids = [chunk_id for i, chunk_id in enumerate(chunk_ids) 
                       if i < len(scores) and scores[i] >= threshold]
    
    elif method == "percentile":
        # Label top X% as relevant
        if not chunk_ids:
            return []
            
        # Calculate how many to include
        top_k = max(1, int(len(chunk_ids) * top_k_percent))
        
        # Get indices of top scoring chunks
        if scores:
            top_indices = np.argsort(scores)[-top_k:]
            relevant_ids = [chunk_ids[i] for i in top_indices if i < len(chunk_ids)]
    
    else:  # "reranker" or "combined" or fallback
        # Mix of threshold and percentile
        # Start with threshold
        threshold_relevance = [chunk_id for i, chunk_id in enumerate(chunk_ids) 
                              if i < len(scores) and scores[i] >= threshold]
        
        # If too few results, use percentile
        if len(threshold_relevance) < 2:
            top_k = max(2, int(len(chunk_ids) * top_k_percent))
            if scores:
                top_indices = np.argsort(scores)[-top_k:]
                relevant_ids = [chunk_ids[i] for i in top_indices if i < len(chunk_ids)]
        else:
            relevant_ids = threshold_relevance
    
    return relevant_ids

def process_retriever(retriever_name: str, retrieval_data: Dict[str, Any], 
                     evaluator: RetrievalEvaluator, args: argparse.Namespace) -> Dict[str, Any]:
    """Process and evaluate results for a single retriever.
    
    Args:
        retriever_name: Name of the retriever
        retrieval_data: Retrieval data for this retriever
        evaluator: Evaluator instance
        args: Command line arguments
        
    Returns:
        Evaluation results
    """
    logger = setup_logger(f"Evaluator_{retriever_name}")
    logger.info(f"Evaluating {retriever_name}")
    
    # Extract data
    queries = retrieval_data.get('queries', {})
    results = retrieval_data.get('results', {})
    
    # Generate query relevance based on pseudo-relevance if no ground truth
    query_relevance = {}
    for query_id, query_data in queries.items():
        # Use results data
        query_results = results.get(query_id, [])
        
        # Generate pseudo-relevance labels
        relevant_ids = generate_pseudo_relevance_labels(
            query_results,
            method=args.relevance_method,
            threshold=args.relevance_threshold,
            top_k_percent=args.relevance_top_k
        )
        
        query_relevance[query_id] = relevant_ids
        logger.debug(f"Generated {len(relevant_ids)} pseudo-relevance labels for query {query_id}")
    
    # Run evaluation
    try:
        eval_results = evaluator.evaluate_retriever(
            retriever_name=retriever_name,
            query_results=results,
            query_relevance=query_relevance,
            k_values=args.k_values,
            run_id=f"eval_{int(time.time())}_{retriever_name}"
        )
        logger.info(f"Completed evaluation for {retriever_name}")
        return eval_results
    except Exception as e:
        logger.error(f"Error evaluating {retriever_name}: {str(e)}")
        return None

def main():
    """Main entry point for evaluation."""
    start_time = time.time()
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logger("RetrievalEvaluator")
    logger.info("Starting retrieval evaluation")
    
    # Group retrieval files by retriever type
    file_groups = group_retrieval_files(args.input_dir)
    if not file_groups:
        logger.error("No complete retrieval file groups found, exiting")
        return
    
    # Filter retrievers if specified
    if args.retrievers:
        file_groups = {name: files for name, files in file_groups.items() 
                      if any(r in name for r in args.retrievers)}
        logger.info(f"Filtered to {len(file_groups)} retrievers")
    
    # Load retrieval data
    retrieval_data = load_retrieval_data(file_groups)
    if not retrieval_data:
        logger.error("No retrieval data loaded, exiting")
        return
    
    # Extract all query IDs
    all_query_ids = set()
    for retriever_name, data in retrieval_data.items():
        all_query_ids.update(data['results'].keys())
    
    # Set output directory and create evaluation subfolder
    output_dir = args.output_dir or os.path.join(args.input_dir, "evaluation")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize evaluator with the evaluation subfolder
    evaluator = RetrievalEvaluator(output_dir=output_dir)
    
    # Process each retriever's results in parallel
    all_evaluations = {}
    
    if args.threads > 1:
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            # Submit tasks
            future_to_retriever = {}
            for retriever_name, data in retrieval_data.items():
                future = executor.submit(
                    process_retriever,
                    retriever_name,
                    data,
                    evaluator,
                    args
                )
                future_to_retriever[future] = retriever_name
            
            # Collect results
            for future in as_completed(future_to_retriever):
                retriever_name = future_to_retriever[future]
                try:
                    eval_results = future.result()
                    if eval_results:
                        all_evaluations[retriever_name] = eval_results
                        logger.info(f"Processed evaluation for {retriever_name}")
                except Exception as e:
                    logger.error(f"Error processing {retriever_name}: {str(e)}")
    else:
        # Process sequentially
        for retriever_name, data in retrieval_data.items():
            eval_results = process_retriever(
                retriever_name,
                data,
                evaluator,
                args
            )
            if eval_results:
                all_evaluations[retriever_name] = eval_results
    
    # Compare retrievers if multiple were evaluated
    if len(all_evaluations) > 1:
        try:
            comparison = evaluator.compare_retrievers(list(all_evaluations.values()))
            comparison_path = os.path.join(output_dir, "retriever_comparison.json")
            with open(comparison_path, "w", encoding="utf-8") as f:
                json.dump(comparison, f, indent=2)
            logger.info(f"Completed retriever comparison and saved to {comparison_path}")
        except Exception as e:
            logger.error(f"Error comparing retrievers: {str(e)}")
    
    # Calculate and log performance metrics
    end_time = time.time()
    elapsed = end_time - start_time
    logger.info(f"Evaluation completed in {elapsed:.2f} seconds")
    logger.info(f"Results saved to {output_dir}")
    
    # Save evaluation metadata
    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_seconds": elapsed,
        "num_retrievers": len(all_evaluations),
        "evaluation_method": args.relevance_method,
        "threshold": args.relevance_threshold if args.relevance_method == "threshold" else None,
        "top_k_percent": args.relevance_top_k if args.relevance_method == "percentile" else None,
        "k_values": args.k_values,
        "retrievers": list(all_evaluations.keys())
    }
    
    metadata_path = os.path.join(output_dir, "evaluation_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved evaluation metadata to {metadata_path}")

if __name__ == "__main__":
    main()
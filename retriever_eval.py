#!/usr/bin/env python3
import os
import json
import argparse
import time
from typing import List, Dict, Any

from retrieval.evaluator import RetrievalEvaluator
from utils.logger import setup_logger

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Retrieval evaluation tool")
    
    # Input/output arguments
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing retrieval results")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to store evaluation results (default: input_dir)")
    parser.add_argument("--relevance_file", type=str, required=False,
                        help="JSON file with relevance judgments")
    
    # Evaluation options
    parser.add_argument("--retrievers", type=str, nargs="+", default=None,
                        help="Specific retrievers to evaluate (default: all)")
    
    return parser.parse_args()

def load_retrieval_results(input_dir: str) -> Dict[str, Dict[str, Any]]:
    """Load retrieval results from the input directory.
    
    Args:
        input_dir: Directory containing retrieval results
        
    Returns:
        Dictionary mapping retriever names to results
    """
    logger = setup_logger("ResultLoader")
    
    # Check if directory exists
    if not os.path.exists(input_dir):
        logger.error(f"Input directory not found: {input_dir}")
        return {}
    
    # Find all retrieval result files
    result_files = []
    for file in os.listdir(input_dir):
        if file.endswith("_retrieval.json"):
            result_files.append(os.path.join(input_dir, file))
    
    if not result_files:
        logger.error(f"No retrieval result files found in {input_dir}")
        return {}
    
    # Load results
    retrieval_results = {}
    for file_path in result_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Extract retriever info
            if "run_info" in data and "retriever_name" in data["run_info"]:
                retriever_name = data["run_info"]["retriever_name"]
                retrieval_results[retriever_name] = data
                logger.info(f"Loaded results for {retriever_name}")
            else:
                logger.warning(f"Invalid retrieval result format in {file_path}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
    
    logger.info(f"Loaded {len(retrieval_results)} retrieval result sets")
    return retrieval_results

def load_relevance_judgments(relevance_file: str) -> Dict[str, Dict[str, List[str]]]:
    """Load relevance judgments from a JSON file.
    
    Args:
        relevance_file: Path to relevance judgments file
        
    Returns:
        Dictionary mapping query IDs to relevant chunk IDs
    """
    logger = setup_logger("RelevanceLoader")
    
    if not relevance_file or not os.path.exists(relevance_file):
        logger.warning("No relevance judgments file provided or file not found")
        return {}
    
    try:
        with open(relevance_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        relevance = {}
        
        # Support different formats
        if isinstance(data, dict):
            for query_id, judgments in data.items():
                if isinstance(judgments, list):
                    relevance[query_id] = judgments
                elif isinstance(judgments, dict) and "relevant_ids" in judgments:
                    relevance[query_id] = judgments["relevant_ids"]
        
        logger.info(f"Loaded relevance judgments for {len(relevance)} queries")
        return relevance
    except Exception as e:
        logger.error(f"Error loading relevance judgments: {str(e)}")
        return {}

def main():
    """Main entry point for evaluation."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logger("RetrievalEvaluator")
    logger.info("Starting retrieval evaluation")
    
    # Set output directory
    output_dir = args.output_dir or args.input_dir
    
    # Load retrieval results
    retrieval_results = load_retrieval_results(args.input_dir)
    if not retrieval_results:
        logger.error("No retrieval results to evaluate, exiting")
        return
    
    # Filter retrievers if specified
    if args.retrievers:
        retrieval_results = {name: results for name, results in retrieval_results.items() 
                           if name in args.retrievers}
        logger.info(f"Filtered to {len(retrieval_results)} retrievers")
    
    # Load relevance judgments if provided
    relevance = load_relevance_judgments(args.relevance_file)
    
    # Set output directory and create evaluation subfolder
    output_dir = args.output_dir or args.input_dir
    eval_dir = os.path.join(output_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    
    # Initialize evaluator with the evaluation subfolder
    evaluator = RetrievalEvaluator(output_dir=eval_dir)
    
    # Process each retriever's results
    all_evaluations = {}
    for retriever_name, results in retrieval_results.items():
        try:
            logger.info(f"Evaluating {retriever_name}")
            
            # Extract query results in the right format
            query_results = {}
            for query_id, query_data in results.get("queries", {}).items():
                query_results[query_id] = query_data.get("retrieved_chunks", [])
            
            # Prepare relevance judgments for this retriever
            query_relevance = {query_id: relevance.get(query_id, []) 
                             for query_id in query_results.keys()}
            
            # Run evaluation
            eval_results = evaluator.evaluate_retriever(
                retriever_name=retriever_name,
                query_results=query_results,
                query_relevance=query_relevance,
                run_id=f"eval_{int(time.time())}_{retriever_name}"
            )
            
            all_evaluations[retriever_name] = eval_results
            logger.info(f"Completed evaluation for {retriever_name}")
        except Exception as e:
            logger.error(f"Error evaluating {retriever_name}: {str(e)}")
    
    # Compare retrievers if multiple were evaluated
    if len(all_evaluations) > 1:
        try:
            comparison = evaluator.compare_retrievers(list(all_evaluations.values()))
            logger.info("Completed retriever comparison")
        except Exception as e:
            logger.error(f"Error comparing retrievers: {str(e)}")
    
    logger.info("Evaluation completed")

if __name__ == "__main__":
    main()
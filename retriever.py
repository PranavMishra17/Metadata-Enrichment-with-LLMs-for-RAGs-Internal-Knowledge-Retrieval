#!/usr/bin/env python3
import os
import json
import time
import argparse
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from retrieval.content_retriever import ContentRetriever
from retrieval.prefix_retriever import PrefixFusionRetriever
from retrieval.tfidf_retriever import TFIDFRetriever
from retrieval.reranker_retriever import RerankerRetriever
from retrieval.evaluator import RetrievalEvaluator
from utils.logger import setup_logger

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Retrieval tool")
    
    # Input/output arguments
    parser.add_argument("--embedding_dir", type=str, default="embeddings_output",
                        help="Directory containing embeddings")
    parser.add_argument("--output_dir", type=str, default="retrieval_output",
                        help="Directory to store retrieval results")
    parser.add_argument("--queries_file", type=str, required=False,
                        help="JSON file containing queries to evaluate")
    
    # Retriever options
    parser.add_argument("--retrievers", type=str, nargs="+", 
                      default=["content", "tfidf", "prefix", "reranker"],
                      help="Retrievers to use (content, tfidf, prefix, reranker)")
    parser.add_argument("--chunking_types", type=str, nargs="+",
                      default=["semantic", "naive", "recursive"],
                      help="Chunking types to use")
    
    # Retrieval parameters
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of results to retrieve")
    parser.add_argument("--reranker_k", type=int, default=20,
                        help="Number of initial results for reranker")
    parser.add_argument("--model", type=str, default="Snowflake/arctic-embed-s",
                        help="Embedding model to use")
    parser.add_argument("--reranker_model", type=str, 
                        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
                        help="Reranker model to use")
    
    # Run identification
    parser.add_argument("--run_id", type=str, default=None,
                        help="Unique ID for this retrieval run")
    
    # Performance options
    parser.add_argument("--threads", type=int, default=4,
                        help="Number of parallel threads to use")
    
    return parser.parse_args()


def load_queries(queries_file: str) -> Dict[str, str]:
    """Load queries from a JSON file.
    
    Args:
        queries_file: Path to JSON file with queries
        
    Returns:
        Dictionary mapping query IDs to query strings
    """
    logger = setup_logger("QueryLoader")
    
    if not os.path.exists(queries_file):
        logger.error(f"Queries file not found: {queries_file}")
        return {}
    
    try:
        with open(queries_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        queries = {}
        
        # Support different formats
        if isinstance(data, list):
            # List of query objects or strings
            for i, item in enumerate(data):
                if isinstance(item, str):
                    # Simple list of strings
                    queries[f"q{i+1}"] = item
                elif isinstance(item, dict):
                    # List of objects
                    if "query" in item:
                        query_id = item.get("id", f"q{i+1}")
                        queries[query_id] = item["query"]
                    elif "text" in item:
                        query_id = item.get("id", f"q{i+1}")
                        queries[query_id] = item["text"]
        elif isinstance(data, dict):
            # Dictionary mapping IDs to queries
            for query_id, query_info in data.items():
                if isinstance(query_info, str):
                    # Simple mapping
                    queries[query_id] = query_info
                elif isinstance(query_info, dict) and ("text" in query_info or "query" in query_info):
                    # Object with text field
                    queries[query_id] = query_info.get("text", query_info.get("query", ""))
        
        logger.info(f"Loaded {len(queries)} queries from {queries_file}")
        return queries
        
    except Exception as e:
        logger.error(f"Error loading queries: {str(e)}")
        return {}


def create_retrievers(args) -> Dict[str, Dict[str, Any]]:
    """Create retriever instances based on command line arguments."""
    """Create retriever instances based on command line arguments."""
    logger = setup_logger("RetrieverFactory")
    logger.info(f"Creating retrievers with top_k={args.top_k}")
    retrievers = {}
    
    # Process each chunking type
    for chunking_type in args.chunking_types:
        # Check if embedding directory exists
        chunking_dir = os.path.join(args.embedding_dir, chunking_type)
        if not os.path.exists(chunking_dir):
            logger.warning(f"Embedding directory not found: {chunking_dir}")
            continue
        
        # Create base retrievers first
        base_retrievers = {}
        
        # Create retrievers for this chunking type
        for retriever_type in args.retrievers:
            try:
                if retriever_type.lower() == "content":
                    # Content-only retriever
                    retriever = ContentRetriever(
                        embedding_dir=args.embedding_dir,
                        chunking_type=chunking_type,
                        model_name=args.model,
                        top_k=args.top_k
                    )
                    retriever_key = f"content_{chunking_type}"
                    retrievers[retriever_key] = {
                        "retriever": retriever,
                        "name": f"Content ({chunking_type})",
                        "type": "content",
                        "chunking": chunking_type
                    }
                    base_retrievers["content"] = retriever
                
                elif retriever_type.lower() == "tfidf":
                    # TF-IDF weighted retriever
                    retriever = TFIDFRetriever(
                        embedding_dir=args.embedding_dir,
                        chunking_type=chunking_type,
                        model_name=args.model,
                        top_k=args.top_k,
                        content_weight=0.7,
                        tfidf_weight=0.3
                    )
                    retriever_key = f"tfidf_{chunking_type}"
                    retrievers[retriever_key] = {
                        "retriever": retriever,
                        "name": f"TF-IDF ({chunking_type})",
                        "type": "tfidf",
                        "chunking": chunking_type
                    }
                    base_retrievers["tfidf"] = retriever
                
                elif retriever_type.lower() == "prefix":
                    # Prefix-fusion retriever
                    retriever = PrefixFusionRetriever(
                        embedding_dir=args.embedding_dir,
                        chunking_type=chunking_type,
                        model_name=args.model,
                        top_k=args.top_k
                    )
                    retriever_key = f"prefix_{chunking_type}"
                    retrievers[retriever_key] = {
                        "retriever": retriever,
                        "name": f"Prefix-Fusion ({chunking_type})",
                        "type": "prefix",
                        "chunking": chunking_type
                    }
                    base_retrievers["prefix"] = retriever
                
            except Exception as e:
                logger.error(f"Error creating {retriever_type} retriever for {chunking_type}: {str(e)}")
        
        # Now create reranker versions for content+metadata retrievers
        if "reranker" in args.retrievers:
            try:
                # Create reranker for TF-IDF if available
                if "tfidf" in base_retrievers:
                    tfidf_retriever = base_retrievers["tfidf"]
                    reranker = RerankerRetriever(
                        base_retriever=tfidf_retriever,
                        reranker_model=args.reranker_model,
                        initial_k=args.reranker_k
                    )
                    retriever_key = f"reranker_tfidf_{chunking_type}"
                    retrievers[retriever_key] = {
                        "retriever": reranker,
                        "name": f"Reranker-TFIDF ({chunking_type})",
                        "type": "reranker_tfidf",
                        "chunking": chunking_type
                    }
                
                # Create reranker for Prefix-Fusion if available
                if "prefix" in base_retrievers:
                    prefix_retriever = base_retrievers["prefix"]
                    reranker = RerankerRetriever(
                        base_retriever=prefix_retriever,
                        reranker_model=args.reranker_model,
                        initial_k=args.reranker_k
                    )
                    retriever_key = f"reranker_prefix_{chunking_type}"
                    retrievers[retriever_key] = {
                        "retriever": reranker,
                        "name": f"Reranker-Prefix ({chunking_type})",
                        "type": "reranker_prefix",
                        "chunking": chunking_type
                    }
            
            except Exception as e:
                logger.error(f"Error creating reranker retrievers for {chunking_type}: {str(e)}")
    
    logger.info(f"Created {len(retrievers)} retrievers")
    return retrievers


def run_retrieval(retriever_info: Dict[str, Any], queries: Dict[str, str], output_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    """Run retrieval for a set of queries using a specific retriever.
    
    Args:
        retriever_info: Dictionary with retriever instance and metadata
        queries: Dictionary mapping query IDs to query strings
        output_dir: Directory to store results
        
    Returns:
        Dictionary mapping query IDs to retrieval results with full chunk content
    """
    retriever = retriever_info["retriever"]
    retriever_name = retriever_info["name"]
    logger = setup_logger(f"Retrieval_{retriever_name}")
    
    results = {}
    checkpoint_file = os.path.join(output_dir, f"{retriever_name.replace(' ', '_')}_results.json")
    
    # Create results directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check for existing checkpoint
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                results = json.load(f)
            logger.info(f"Loaded checkpoint with {len(results)} queries")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
    
    # Process queries
    completed_count = 0
    total_count = len(queries)
    start_time = time.time()
    
    try:
        for query_id, query_text in queries.items():
            # Skip if already processed
            if query_id in results:
                continue
            
            # Retrieve results
            retrieval_results = retriever.retrieve(query_text)
            
            # Enhance results with full chunk content if missing
            for result in retrieval_results:
                if "text" not in result or not result["text"]:
                    chunk_id = result.get("chunk_id")
                    if chunk_id and chunk_id in retriever.id_to_metadata:
                        # Get full text from metadata
                        result["text"] = retriever.id_to_metadata[chunk_id].get("text", "")
            
            # Store results
            results[query_id] = retrieval_results
            
            # Update progress
            completed_count += 1
            if completed_count % 5 == 0 or completed_count == total_count:
                elapsed = time.time() - start_time
                qps = completed_count / elapsed if elapsed > 0 else 0
                remaining = (total_count - completed_count) / qps if qps > 0 else 0
                logger.info(f"Processed {completed_count}/{total_count} queries ({qps:.2f} qps, {remaining:.2f}s remaining)")
                
                # Save checkpoint
                with open(checkpoint_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)
    
    except Exception as e:
        logger.error(f"Error in retrieval: {str(e)}")
        # Save current results
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    
    # Save final results in a standardized format
    final_output = {
        "run_info": {
            "retriever_name": retriever_info["name"],
            "retriever_type": retriever_info["type"],
            "chunking_type": retriever_info["chunking"],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "top_k": retriever.top_k
        },
        "queries": {
            query_id: {
                "query_text": query_text,
                "retrieved_chunks": results[query_id]
            } for query_id, query_text in queries.items() if query_id in results
        }
    }
    
    # Save in a different format for answer generation
    final_output_path = os.path.join(output_dir, f"{retriever_info['name'].replace(' ', '_')}_retrieval.json")
    with open(final_output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    
    return results


def evaluate_retrievers(retrievers: Dict[str, Dict[str, Any]], queries: Dict[str, str], relevance: Dict[str, List[str]], output_dir: str, run_id: str = None, threads: int = 4, eval_only: bool = False) -> Dict[str, Dict[str, Any]]:
    """Run retrieval and evaluation for all retrievers.
    
    Args:
        retrievers: Dictionary of retriever information
        queries: Dictionary mapping query IDs to query strings
        relevance: Dictionary mapping query IDs to relevant chunk IDs
        output_dir: Directory to store results
        run_id: Unique ID for this evaluation run
        threads: Number of parallel threads to use
        eval_only: Only run evaluation on existing results
        
    Returns:
        Dictionary of evaluation results by retriever
    """
    logger = setup_logger("Evaluation")
    
    # Generate run ID if not provided
    if run_id is None:
        run_id = f"run_{int(time.time())}"
    
    # Create output directory
    results_dir = os.path.join(output_dir, run_id)
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize evaluator
    evaluator = RetrievalEvaluator(output_dir=results_dir)
    
    # Get all results
    all_results = {}
    all_evaluations = {}
    
    # Run retrieval or load existing results
    if not eval_only:
        # Prepare thread pool for parallel retrieval
        with ThreadPoolExecutor(max_workers=threads) as executor:
            # Submit retrieval tasks
            future_to_retriever = {}
            for retriever_key, retriever_info in retrievers.items():
                future = executor.submit(
                    run_retrieval, 
                    retriever_info, 
                    queries, 
                    results_dir
                )
                future_to_retriever[future] = retriever_key
            
            # Collect results
            for future in as_completed(future_to_retriever):
                retriever_key = future_to_retriever[future]
                try:
                    retriever_results = future.result()
                    all_results[retriever_key] = retriever_results
                    logger.info(f"Completed retrieval for {retriever_key}")
                except Exception as e:
                    logger.error(f"Error in retrieval for {retriever_key}: {str(e)}")
    else:
        # Load existing results
        logger.info("Loading existing retrieval results")
        for retriever_key, retriever_info in retrievers.items():
            retriever_name = retriever_info["name"].replace(" ", "_")
            results_file = os.path.join(results_dir, f"{retriever_name}_results.json")
            
            if os.path.exists(results_file):
                try:
                    with open(results_file, "r", encoding="utf-8") as f:
                        retriever_results = json.load(f)
                    all_results[retriever_key] = retriever_results
                    logger.info(f"Loaded results for {retriever_key}")
                except Exception as e:
                    logger.error(f"Error loading results for {retriever_key}: {str(e)}")
    
    # Run evaluation for each retriever
    for retriever_key, retriever_info in retrievers.items():
        retriever_name = retriever_info["name"]
        
        # Skip if no results
        if retriever_key not in all_results:
            logger.warning(f"No results found for {retriever_name}")
            continue
        
        # Get results
        retriever_results = all_results[retriever_key]
        
        # Evaluate
        try:
            logger.info(f"Evaluating {retriever_name}")
            eval_results = evaluator.evaluate_retriever(
                retriever_name=retriever_name,
                query_results=retriever_results,
                query_relevance=relevance,
                run_id=f"{run_id}_{retriever_key}"
            )
            all_evaluations[retriever_key] = eval_results
            logger.info(f"Completed evaluation for {retriever_name}")
        except Exception as e:
            logger.error(f"Error evaluating {retriever_name}: {str(e)}")
    
    # Compare retrievers
    if len(all_evaluations) > 1:
        try:
            comparison = evaluator.compare_retrievers(list(all_evaluations.values()))
            logger.info("Completed retriever comparison")
        except Exception as e:
            logger.error(f"Error comparing retrievers: {str(e)}")
    
    return all_evaluations


def generate_default_queries() -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """Generate a set of default queries when no query file is provided.
    
    Returns:
        Tuple of (queries, relevance_judgments)
    """
    # Create some sample AWS S3 and Glacier queries
    queries = {
        "q1": "How do I create a bucket in S3?",
        "q2": "What is Amazon S3 Glacier?",
        "q3": "How to upload files to S3?",
        "q4": "S3 bucket access policy examples",
        "q5": "Difference between S3 and Glacier storage",
        "q6": "How to restore files from Glacier",
        "q7": "S3 lifecycle configuration",
        "q8": "Setting up cross-region replication in S3",
        "q9": "How to enable versioning in S3",
        "q10": "Creating a vault in Amazon Glacier"
    }
    
    # Empty relevance judgments (no ground truth)
    relevance = {query_id: [] for query_id in queries}
    
    return queries, relevance

def main():
    """Main entry point for retrieval and evaluation."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logger("RetrieverMain")
    logger.info("Starting retrieval process")
    
    # Load queries
    if args.queries_file:
        queries = load_queries(args.queries_file)
        if not queries:
            logger.error("No queries loaded, using default queries")
            queries, _ = generate_default_queries()
    else:
        logger.info("No queries file provided, using default queries")
        queries, _ = generate_default_queries()
    
    # Create retrievers
    retrievers = create_retrievers(args)
    
    if not retrievers:
        logger.error("No retrievers created, exiting")
        return
    
    # Run retrieval for all retrievers
    all_results = {}
    
    # Prepare directory for this run
    run_id = args.run_id or f"run_{int(time.time())}"
    results_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(results_dir, exist_ok=True)
    
    
    # Run in parallel if multiple threads
    if args.threads > 1:
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            future_to_retriever = {}
            for retriever_key, retriever_info in retrievers.items():
                future = executor.submit(run_retrieval, retriever_info, queries, results_dir)
                future_to_retriever[future] = retriever_key
            
            for future in as_completed(future_to_retriever):
                retriever_key = future_to_retriever[future]
                try:
                    results = future.result()
                    all_results[retriever_key] = results
                except Exception as e:
                    logger.error(f"Error in retrieval for {retriever_key}: {str(e)}")
    else:
        # Run sequentially
        for retriever_key, retriever_info in retrievers.items():
            try:
                results = run_retrieval(retriever_info, queries, results_dir)
                all_results[retriever_key] = results
            except Exception as e:
                logger.error(f"Error in retrieval for {retriever_key}: {str(e)}")
    
    # Save a summary of all results
    summary = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "query_count": len(queries),
        "retrievers": list(retrievers.keys()),
        "output_path": results_dir
    }
    
    summary_path = os.path.join(results_dir, "retrieval_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Retrieval completed. Results saved to {results_dir}")
    logger.info(f"To evaluate these results, run: python retriever_eval.py --input_dir {results_dir}")

if __name__ == "__main__":
    main()
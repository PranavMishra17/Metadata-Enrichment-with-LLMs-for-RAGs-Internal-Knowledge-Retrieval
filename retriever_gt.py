#!/usr/bin/env python3
import os
import json
import time
import argparse
import logging
import re
from typing import List, Dict, Any, Tuple
import concurrent.futures
from gpu_utils import GPUVerifier

# Initialize GPU verification
gpu_verifier = GPUVerifier(require_gpu=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def setup_logger(name):
    """Set up a logger with the given name."""
    return logging.getLogger(name)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Ground Truth Generator for Retrieval System")
    
    # Input/output arguments
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing retrieval output")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to store ground truth files (defaults to input_dir/ground_truth)")
    parser.add_argument("--queries_file", type=str, default="sample_q.json",
                        help="JSON file containing queries")
    
    # LLM evaluation options
    parser.add_argument("--top_k", type=int, default=25,
                        help="Number of chunks to evaluate")
    parser.add_argument("--llm_batch_size", type=int, default=25,
                        help="Batch size for LLM processing")
    
    # Performance options
    parser.add_argument("--threads", type=int, default=4,
                        help="Number of parallel threads to use")
    parser.add_argument("--rate_limit", type=float, default=2.0,
                        help="Rate limit for LLM API calls (calls per second)")
    
    return parser.parse_args()

def load_queries(queries_file: str) -> Dict[str, str]:
    """Load queries from a JSON file."""
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
                    queries[f"q{i+1}"] = item
                elif isinstance(item, dict):
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
                    queries[query_id] = query_info
                elif isinstance(query_info, dict) and ("text" in query_info or "query" in query_info):
                    queries[query_id] = query_info.get("text", query_info.get("query", ""))
        
        logger.info(f"Loaded {len(queries)} queries from {queries_file}")
        return queries
        
    except Exception as e:
        logger.error(f"Error loading queries: {str(e)}")
        return {}

def load_retrieval_results(input_dir: str) -> Dict[str, Dict[str, Any]]:
    """Load retrieval results from the given directory."""
    logger = setup_logger("ResultLoader")
    results = {}
    
    # Look for all JSON files except summary.json
    for filename in os.listdir(input_dir):
        if filename.endswith("_retrieval.json") and not filename.startswith("retrieval_summary"):
            try:
                file_path = os.path.join(input_dir, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Extract retriever name from filename
                retriever_name = filename.replace("_retrieval.json", "")
                results[retriever_name] = data
                logger.info(f"Loaded results for {retriever_name}")
            except Exception as e:
                logger.error(f"Error loading results from {filename}: {str(e)}")
    
    return results

def evaluate_chunks_with_reranker(query: str, chunks: List[Dict[str, Any]], top_k: int = 25) -> Tuple[List[Dict[str, Any]], int, Dict[int, float]]:
    """Evaluate chunks using a reranker model and rank them."""
    logger = setup_logger("RerankerEvaluator")
    
    # Limit chunks to top_k
    chunks = chunks[:top_k]
    
    try:
        # Use CrossEncoder instead of SentenceTransformer for reranking
        from sentence_transformers import CrossEncoder
        model = CrossEncoder("BAAI/bge-reranker-base")
        
        # Prepare query-chunk pairs
        chunk_texts = [chunk.get("text", "") for chunk in chunks]
        sentence_pairs = [[query, text] for text in chunk_texts]
        
        # Calculate scores
        scores = model.predict(sentence_pairs)
        
        # Store original ranks and normalize scores to 0-1
        min_score, max_score = min(scores), max(scores)
        range_score = max_score - min_score if max_score > min_score else 1
        
        scored_chunks = []
        for i, score in enumerate(scores):
            normalized_score = (score - min_score) / range_score
            scored_chunks.append({
                "chunk_id": chunks[i].get("chunk_id"),
                "score": float(normalized_score),
                "original_rank": i + 1  # Original rank (1-based)
            })
        
        # Sort by score
        ranked_chunks = sorted(scored_chunks, key=lambda x: x["score"], reverse=True)
        
        # Calculate percentiles
        all_scores = [chunk["score"] for chunk in scored_chunks]
        import numpy as np
        percentiles = {
            99: float(np.percentile(all_scores, 99)),
            95: float(np.percentile(all_scores, 95)),
            90: float(np.percentile(all_scores, 90)),
            85: float(np.percentile(all_scores, 85))
        }
        
        # Track rank changes - THIS IS THE IMPORTANT FIX
        rank_changes = 0
        final_chunks = []
        for i, chunk in enumerate(ranked_chunks):
            new_rank = i + 1
            original_rank = chunk["original_rank"]  # Keep this for comparison
            if new_rank != original_rank:
                rank_changes += 1
            
            # Remove original_rank from the output
            chunk_copy = chunk.copy()
            chunk_copy.pop("original_rank")
            final_chunks.append(chunk_copy)
        
        return final_chunks, rank_changes, percentiles
        
    except Exception as e:
        logger.error(f"Error evaluating with reranker: {str(e)}")
        return [{"chunk_id": c.get("chunk_id"), "score": 1.0 - (i/len(chunks))} 
                for i, c in enumerate(chunks)], 0, {99: 1.0, 95: 0.95, 90: 0.9, 85: 0.85}


def generate_ground_truth(queries, retrieval_results, output_dir, top_k=25, threads=4):
    logger = setup_logger("GTGenerator")
    os.makedirs(output_dir, exist_ok=True)
    
    all_ground_truth = {}
    all_stats = {"rank_changes": {}, "percentiles": {}}
    
    for retriever_name, retriever_data in retrieval_results.items():
        logger.info(f"Generating ground truth for {retriever_name}")
        
        retriever_gt = {
            "retriever_name": retriever_name,
            "ground_truth": {},
            "statistics": {
                "rank_changes": {},
                "percentiles": {}
            }
        }
        
        retriever_queries = retriever_data.get("queries", {})
        total_rank_changes = 0
        total_chunks = 0
        all_retriever_scores = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            future_to_query = {}
            
            for query_id in queries.keys():
                if query_id not in retriever_queries:
                    continue
                
                query_text = queries[query_id]
                chunks = retriever_queries[query_id].get("retrieved_chunks", [])
                
                future = executor.submit(evaluate_chunks_with_reranker, query_text, chunks, top_k)
                future_to_query[future] = query_id
            
            for future in concurrent.futures.as_completed(future_to_query):
                query_id = future_to_query[future]
                try:
                    ranked_chunks, rank_changes, percentiles = future.result()
                    retriever_gt["ground_truth"][query_id] = ranked_chunks
                    retriever_gt["statistics"]["rank_changes"][query_id] = rank_changes
                    retriever_gt["statistics"]["percentiles"][query_id] = percentiles
                    
                    total_rank_changes += rank_changes
                    total_chunks += len(ranked_chunks)
                    all_retriever_scores.extend([c["score"] for c in ranked_chunks])
                    
                    logger.info(f"Completed GT for {retriever_name}, query {query_id}")
                except Exception as e:
                    logger.error(f"Error generating GT for {retriever_name}, query {query_id}: {str(e)}")
        
        # Calculate overall stats
        if total_chunks > 0:
            retriever_gt["statistics"]["overall_rank_change_percent"] = (total_rank_changes / total_chunks) * 100
            
            # Overall percentiles for this retriever
            import numpy as np
            retriever_gt["statistics"]["overall_percentiles"] = {
                99: float(np.percentile(all_retriever_scores, 99)),
                95: float(np.percentile(all_retriever_scores, 95)),
                90: float(np.percentile(all_retriever_scores, 90)),
                85: float(np.percentile(all_retriever_scores, 85))
            }
        
        # Save ground truth
        output_path = os.path.join(output_dir, f"{retriever_name}_ground_truth.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(retriever_gt, f, indent=2)
        
        all_ground_truth[retriever_name] = retriever_gt
    
    return all_ground_truth


def main():
    """Main entry point for ground truth generation."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logger("GTMain")
    logger.info("Starting ground truth generation")
    
    # Default output directory if not specified
    if args.output_dir is None:
        args.output_dir = os.path.join(args.input_dir, "ground_truth")
    
    # Load queries
    queries = load_queries(args.queries_file)
    if not queries:
        logger.error("No queries loaded, exiting")
        return
    
    # Load retrieval results
    retrieval_results = load_retrieval_results(args.input_dir)
    if not retrieval_results:
        logger.error("No retrieval results loaded, exiting")
        return
    
    # Generate ground truth
    ground_truth = generate_ground_truth(
        queries,
        retrieval_results,
        args.output_dir,
        args.top_k,
        args.threads
    )
    
    logger.info(f"Ground truth generation completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
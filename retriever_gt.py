#!/usr/bin/env python3
import os
import json
import time
import argparse
import logging
import re
from typing import List, Dict, Any, Tuple
import concurrent.futures

import numpy as np
import pandas as pd
from gpu_utils import GPUVerifier
from response_quality_eval import ResponseQualityEvaluator, create_comparison_tables
from utils import logger

# Initialize GPU verification
gpu_verifier = GPUVerifier(require_gpu=True)


# Add these imports at the top
from sentence_transformers import CrossEncoder
from collections import defaultdict

class RerankerEvaluator:
    """Evaluates answers using a cross-encoder reranker."""
    
    def __init__(self, model_name="BAAI/bge-reranker-base"):
        self.logger = logging.getLogger("RerankerEvaluator")
        self.reranker = None
        self.model_name = model_name
        self._load_reranker()
        
    def _load_reranker(self):
        """Load the reranker model."""
        try:
            self.reranker = CrossEncoder(self.model_name)
            self.logger.info(f"Loaded reranker model: {self.model_name}")
        except Exception as e:
            self.logger.error(f"Error loading reranker model: {str(e)}")
            raise
    
    def rerank_answers(self, answers_by_retriever, ground_truth):
        """Rerank answers from all retrievers against ground truth.
        
        Args:
            answers_by_retriever: Dict mapping retriever names to answer dictionaries
            ground_truth: Dict mapping query IDs to ground truth answers
            
        Returns:
            Dict with reranking results
        """
        results = {
            "by_query": {},
            "aggregated": {
                "mean_score": {},
                "median_score": {},
                "wins": {},
                "top3": {},
                "mean_rank": {}
            }
        }
        
        # Count retrievers and initialize counters
        retrievers = list(answers_by_retriever.keys())
        retriever_scores = {r: [] for r in retrievers}
        retriever_ranks = {r: [] for r in retrievers}
        wins = {r: 0 for r in retrievers}
        top3_counts = {r: 0 for r in retrievers}
        
        # Find common queries across all retrievers
        common_queries = set()
        for retriever, answers in answers_by_retriever.items():
            if not common_queries:
                common_queries = set(answers.keys())
            else:
                common_queries &= set(answers.keys())
        
        self.logger.info(f"Found {len(common_queries)} common queries across {len(retrievers)} retrievers")
        
        # Process each query
        for query_id in common_queries:
            if query_id not in ground_truth:
                self.logger.warning(f"Query {query_id} not found in ground truth, skipping")
                continue
                
            gt = ground_truth[query_id]
            query_results = {"scores": {}, "ranking": []}
            
            # Prepare pairs for reranking
            pairs = []
            pair_map = []  # To map back to retrievers
            
            for retriever in retrievers:
                answer = answers_by_retriever[retriever].get(query_id, {})
                if not answer:
                    continue
                
                if isinstance(answer, dict) and "answer" in answer:
                    answer_text = answer["answer"]
                else:
                    answer_text = answer
                
                # Skip empty answers
                if not answer_text or not answer_text.strip():
                    continue
                    
                pairs.append([gt, answer_text])
                pair_map.append(retriever)
            
            # Rerank with cross-encoder
            if not pairs:
                self.logger.warning(f"No valid answers for query {query_id}")
                continue
                
            scores = self.reranker.predict(pairs)
            
            # Store scores and update counters
            for i, score in enumerate(scores):
                retriever = pair_map[i]
                query_results["scores"][retriever] = float(score)
                retriever_scores[retriever].append(float(score))
            
            # Rank retrievers by score
            ranked_retrievers = sorted(query_results["scores"].items(), 
                                      key=lambda x: x[1], reverse=True)
            
            query_results["ranking"] = [r for r, s in ranked_retrievers]
            
            # Update win count for the best retriever
            if ranked_retrievers:
                wins[ranked_retrievers[0][0]] += 1
            
            # Update top-3 counts
            for i, (retriever, _) in enumerate(ranked_retrievers[:3]):
                top3_counts[retriever] += 1
            
            # Update rank counters
            for i, (retriever, _) in enumerate(ranked_retrievers):
                rank = i + 1
                retriever_ranks[retriever].append(rank)
            
            # Store query results
            results["by_query"][query_id] = query_results
        
        # Calculate aggregated metrics
        for retriever in retrievers:
            scores = retriever_scores[retriever]
            ranks = retriever_ranks[retriever]
            
            if not scores:
                continue
                
            results["aggregated"]["mean_score"][retriever] = np.mean(scores)
            results["aggregated"]["median_score"][retriever] = np.median(scores)
            results["aggregated"]["wins"][retriever] = wins[retriever]
            results["aggregated"]["top3"][retriever] = top3_counts[retriever]
            results["aggregated"]["mean_rank"][retriever] = np.mean(ranks)
        
        return results



# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)



def create_reranker_comparison_tables(reranker_results, output_dir):
    """Create tables for reranker comparison results."""
    if not reranker_results:
        return
    
    aggregated = reranker_results["aggregated"]
    
    # Prepare data
    data = []
    for retriever in aggregated["mean_score"].keys():
        # Extract chunking and embedding type
        if "(" in retriever and ")" in retriever:
            chunking = retriever.split("(")[1].split(")")[0]
        else:
            chunking = "unknown"
            
        if "Content" in retriever:
            embedding = "content"
        elif "TF-IDF" in retriever:
            embedding = "tfidf"
        elif "Prefix" in retriever:
            embedding = "prefix"
        elif "Reranker" in retriever:
            if "TFIDF" in retriever:
                embedding = "reranker_tfidf"
            elif "Prefix" in retriever:
                embedding = "reranker_prefix"
            else:
                embedding = "reranker"
        else:
            embedding = "unknown"
        
        # Add to data
        data.append({
            "retriever": retriever,
            "chunking": chunking,
            "embedding": embedding,
            "mean_score": aggregated["mean_score"].get(retriever, 0),
            "median_score": aggregated["median_score"].get(retriever, 0),
            "wins": aggregated["wins"].get(retriever, 0),
            "top3": aggregated["top3"].get(retriever, 0),
            "mean_rank": aggregated["mean_rank"].get(retriever, 0)
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save tables
    df.to_csv(os.path.join(output_dir, "reranker_comparison.csv"), index=False)
    
    # Create pivot tables if possible
    if len(df["chunking"].unique()) > 1 and len(df["embedding"].unique()) > 1:
        # Mean score table
        score_pivot = df.pivot_table(index="chunking", columns="embedding", values="mean_score")
        score_pivot.to_csv(os.path.join(output_dir, "reranker_score_by_type.csv"))
        
        # Win count table
        win_pivot = df.pivot_table(index="chunking", columns="embedding", values="wins")
        win_pivot.to_csv(os.path.join(output_dir, "reranker_wins_by_type.csv"))
        
        # Mean rank table
        rank_pivot = df.pivot_table(index="chunking", columns="embedding", values="mean_rank")
        rank_pivot.to_csv(os.path.join(output_dir, "reranker_rank_by_type.csv"))

# Add this function to your script
def load_answers_by_retriever(answers_dir):
    """Load answers from all retrievers into a structured dictionary."""
    answers_by_retriever = {}
    
    # List all answer files
    answer_files = [f for f in os.listdir(answers_dir) if f.endswith("_answers.json")]
    
    for file_name in answer_files:
        file_path = os.path.join(answers_dir, file_name)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            retriever_name = data.get("retriever_name", file_name.replace("_answers.json", ""))
            
            # Extract answers
            retriever_answers = {}
            for query_id, query_data in data.get("answers", {}).items():
                if isinstance(query_data, dict) and "answer" in query_data:
                    retriever_answers[query_id] = query_data["answer"]
                else:
                    retriever_answers[query_id] = query_data
            
            answers_by_retriever[retriever_name] = retriever_answers
            logger.info(f"Loaded {len(retriever_answers)} answers for {retriever_name}")
            
        except Exception as e:
            logger.error(f"Error loading {file_name}: {str(e)}")
    
    return answers_by_retriever


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
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()
    
    # Set default output directory if not specified
    if args.output_dir is None:
        args.output_dir = os.path.join(args.answers_dir, "evaluations")
    
    # Set default retrieval directory if not specified
    if args.retrieval_dir is None:
        answers_parent = os.path.dirname(args.answers_dir)
        if os.path.basename(answers_parent).startswith("run_"):
            args.retrieval_dir = answers_parent
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize evaluators
    quality_evaluator = ResponseQualityEvaluator(use_gpu=args.use_gpu)
    reranker_evaluator = RerankerEvaluator()
    
    # Load data
    ground_truth = quality_evaluator.load_ground_truth(args.ground_truth)
    answers = quality_evaluator.load_answers(args.answers_dir)
    retrieval_results = quality_evaluator.load_retrieval_results(args.retrieval_dir)
    
    # Group answers by retriever for the reranker
    answers_by_retriever = load_answers_by_retriever(args.answers_dir)
    
    # Check if we have data
    if not ground_truth:
        logger.error("No ground truth data loaded")
        return
    
    if not answers:
        logger.error("No answer data loaded")
        return
    
    # Evaluate each retriever for standard metrics
    evaluation_results = {}
    
    for retriever_name, retriever_answers in answers.items():
        # Skip if missing retrieval results
        if retriever_name not in retrieval_results:
            logger.warning(f"Skipping {retriever_name} - no retrieval results found")
            continue
        
        # Evaluate
        results = quality_evaluator.evaluate_retriever(
            retriever_name=retriever_name,
            answers=retriever_answers,
            retrieval_results=retrieval_results[retriever_name],
            ground_truth=ground_truth,
            top_k=args.top_k
        )
        
        # Store results
        evaluation_results[retriever_name] = results
        
        # Save individual results
        output_path = os.path.join(args.output_dir, f"{retriever_name.replace(' ', '_')}_evaluation.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved evaluation results for {retriever_name} to {output_path}")
    
    # Compare retrievers based on accuracy/faithfulness
    comparison = quality_evaluator.compare_retrievers(evaluation_results)
    
    # Save comparison
    comparison_path = os.path.join(args.output_dir, "retriever_comparison.json")
    with open(comparison_path, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2)
    
    logger.info(f"Saved retriever comparison to {comparison_path}")
    
    # Create comparison tables
    create_comparison_tables(comparison, args.output_dir)
    
    # Run reranker evaluation
    logger.info("Running reranker-based comparison")
    reranker_results = reranker_evaluator.rerank_answers(answers_by_retriever, ground_truth)
    
    # Save reranker results
    reranker_path = os.path.join(args.output_dir, "reranker_results.json")
    with open(reranker_path, 'w', encoding='utf-8') as f:
        json.dump(reranker_results, f, indent=2)
    
    logger.info(f"Saved reranker evaluation to {reranker_path}")
    
    # Create reranker comparison tables
    create_reranker_comparison_tables(reranker_results, args.output_dir)
    
    # Print summary
    logger.info("Evaluation summary:")
    for retriever, result in evaluation_results.items():
        logger.info(f"  {retriever}:")
        logger.info(f"    Faithfulness: {result['avg_faithfulness']:.4f}")
        logger.info(f"    Accuracy: {result['avg_accuracy']:.4f}")
    
    # Print reranker summary
    logger.info("Reranker comparison summary:")
    for retriever, score in reranker_results["aggregated"]["mean_score"].items():
        wins = reranker_results["aggregated"]["wins"].get(retriever, 0)
        top3 = reranker_results["aggregated"]["top3"].get(retriever, 0)
        logger.info(f"  {retriever}: Score={score:.4f}, Wins={wins}, Top3={top3}")
    
    logger.info("Evaluation completed successfully")


if __name__ == "__main__":
    main()
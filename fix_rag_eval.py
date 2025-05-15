#!/usr/bin/env python3
import os
import json
import argparse
import logging
import re
import numpy as np
from typing import List, Dict, Set, Any, Tuple
from collections import Counter, defaultdict

from gpu_utils import GPUVerifier

# Initialize GPU verification
gpu_verifier = GPUVerifier(require_gpu=True)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RAG-Evaluator-Fix")

# Try to import spaCy
try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False
    logger.warning("spaCy not installed. Using simple tokenization.")

# Try to import BERTScore
try:
    from bert_score import BERTScorer
    HAS_BERTSCORE = True
except ImportError:
    HAS_BERTSCORE = False
    logger.warning("bert-score not installed. Semantic similarity evaluation will be limited.")

def load_ground_truth(ground_truth_path: str) -> Dict[str, str]:
    """Load ground truth answers."""
    if not os.path.exists(ground_truth_path):
        logger.warning(f"Ground truth file not found: {ground_truth_path}")
        return {}
    
    try:
        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        ground_truth = {}
        
        # Handle format: {query_id: {answer: "text"}}
        for query_id, value in data.items():
            if isinstance(value, dict) and "answer" in value:
                ground_truth[query_id] = value["answer"]
            elif isinstance(value, str):
                ground_truth[query_id] = value
        
        logger.info(f"Loaded {len(ground_truth)} ground truth answers")
        return ground_truth
    
    except Exception as e:
        logger.error(f"Error loading ground truth: {str(e)}")
        return {}

def tokenize_text(text: str, nlp=None) -> List[str]:
    """Tokenize text into words/tokens."""
    if not text:
        return []
    
    if nlp:
        # Use spaCy for better tokenization
        doc = nlp(text)
        return [token.text.lower() for token in doc if not token.is_punct and not token.is_space]
    else:
        # Simple tokenization fallback
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)  # Replace punctuation with space
        return [word for word in text.split() if word.strip()]

def calculate_token_overlap(response_tokens: List[str], context_tokens: List[str]) -> float:
    """Calculate token overlap between response and context."""
    if not response_tokens:
        return 0.0
        
    # Count tokens in both texts
    response_counter = Counter(response_tokens)
    context_counter = Counter(context_tokens)
    
    # Calculate intersection
    intersection = sum((response_counter & context_counter).values())
    
    # Calculate faithfulness as portion of response tokens found in context
    return intersection / len(response_tokens)

def calculate_bertscore(bert_scorer, response: str, reference: str) -> Dict[str, float]:
    """Calculate BERTScore between response and reference."""
    if not bert_scorer or not response or not reference:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    try:
        P, R, F1 = bert_scorer.score([response], [reference])
        return {
            "precision": float(P[0]),
            "recall": float(R[0]),
            "f1": float(F1[0])
        }
    except Exception as e:
        logger.error(f"Error calculating BERTScore: {str(e)}")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

def evaluate_answer_file(answer_file: str, retrieval_dir: str, ground_truth: Dict[str, str], 
                         nlp=None, bert_scorer=None, max_chunks: int = 20) -> Dict[str, Any]:
    """Evaluate answers against retrieved chunks and ground truth."""
    logger.info(f"Processing {answer_file}")
    
    # Extract retriever info from filename
    filename = os.path.basename(answer_file)
    retriever_type = "unknown"
    chunking_type = "unknown"
    
    if "__" in filename:
        parts = filename.split("__")
        if len(parts) >= 3:
            retriever_type = parts[0]
            chunking_type = parts[1]
    
    # Load answer file
    try:
        with open(answer_file, 'r', encoding='utf-8') as f:
            answer_data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading answer file {answer_file}: {str(e)}")
        return {}
    
    # Find corresponding retrieval file
    retrieval_files = []
    for fname in os.listdir(retrieval_dir):
        if fname.endswith("_retrieval.json") and retriever_type in fname and chunking_type in fname:
            retrieval_files.append(os.path.join(retrieval_dir, fname))
    
    if not retrieval_files:
        logger.warning(f"No matching retrieval file found for {filename}")
        return {}
    
    # Load retrieval file
    try:
        with open(retrieval_files[0], 'r', encoding='utf-8') as f:
            retrieval_data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading retrieval file {retrieval_files[0]}: {str(e)}")
        return {}
    
    # Extract retrieval chunks by query_id
    chunks_by_query = {}
    for query_id, query_data in retrieval_data.get("queries", {}).items():
        chunks = query_data.get("retrieved_chunks", [])
        chunk_texts = []
        
        # Limit to max_chunks
        for chunk in chunks[:max_chunks]:
            chunk_text = chunk.get("text", "").strip()
            if chunk_text:
                chunk_texts.append(chunk_text)
        
        chunks_by_query[query_id] = chunk_texts
    
    # Process each answer
    results = {
        "retriever_name": retriever_type + "_" + chunking_type,
        "evaluations": {},
        "aggregates": {}
    }
    
    faithfulness_scores = []
    gt_accuracy_scores = []
    bertscore_f1_scores = []
    
    for query_id, answer_item in answer_data.get("answers", {}).items():
        query_text = answer_item.get("query", "")
        answer_text = answer_item.get("answer", "")
        
        # Get chunks for this query
        chunks = chunks_by_query.get(query_id, [])
        combined_chunks = "\n\n".join(chunks)
        
        # Tokenize texts
        answer_tokens = tokenize_text(answer_text, nlp)
        chunks_tokens = tokenize_text(combined_chunks, nlp)
        
        # Calculate faithfulness (token overlap)
        faithfulness = calculate_token_overlap(answer_tokens, chunks_tokens)
        
        # Calculate ground truth metrics if available
        gt_metrics = {}
        if query_id in ground_truth:
            gt_answer = ground_truth[query_id]
            gt_tokens = tokenize_text(gt_answer, nlp)
            
            # Calculate accuracy (token overlap with ground truth)
            gt_accuracy = calculate_token_overlap(answer_tokens, gt_tokens)
            gt_metrics["accuracy"] = gt_accuracy
            gt_accuracy_scores.append(gt_accuracy)
            
            # Calculate BERTScore if available
            if bert_scorer:
                bertscore = calculate_bertscore(bert_scorer, answer_text, gt_answer)
                gt_metrics["bertscore"] = bertscore
                bertscore_f1_scores.append(bertscore["f1"])
        
        # Store results
        results["evaluations"][query_id] = {
            "query": query_text,
            "faithfulness": faithfulness,
            "num_chunks_used": len(chunks),
            "ground_truth": gt_metrics
        }
        
        faithfulness_scores.append(faithfulness)
    
    # Calculate aggregates
    if faithfulness_scores:
        results["aggregates"]["avg_faithfulness"] = np.mean(faithfulness_scores)
        results["aggregates"]["min_faithfulness"] = np.min(faithfulness_scores)
        results["aggregates"]["max_faithfulness"] = np.max(faithfulness_scores)
    
    if gt_accuracy_scores:
        results["aggregates"]["avg_gt_accuracy"] = np.mean(gt_accuracy_scores)
        results["aggregates"]["min_gt_accuracy"] = np.min(gt_accuracy_scores)
        results["aggregates"]["max_gt_accuracy"] = np.max(gt_accuracy_scores)
    
    if bertscore_f1_scores:
        results["aggregates"]["avg_bertscore_f1"] = np.mean(bertscore_f1_scores)
        results["aggregates"]["min_bertscore_f1"] = np.min(bertscore_f1_scores)
        results["aggregates"]["max_bertscore_f1"] = np.max(bertscore_f1_scores)
    
    return results

def create_comparison_table(all_results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Create comparison tables for different metrics."""
    comparison = {
        "faithfulness": defaultdict(dict),
        "gt_accuracy": defaultdict(dict),
        "bertscore_f1": defaultdict(dict)
    }
    
    for retriever_name, results in all_results.items():
        if "__" in retriever_name:
            parts = retriever_name.split("__")
            if len(parts) >= 2:
                retriever = parts[0]
                chunking = parts[1]
                
                # Add to comparison tables
                if "avg_faithfulness" in results.get("aggregates", {}):
                    comparison["faithfulness"][retriever][chunking] = results["aggregates"]["avg_faithfulness"]
                
                if "avg_gt_accuracy" in results.get("aggregates", {}):
                    comparison["gt_accuracy"][retriever][chunking] = results["aggregates"]["avg_gt_accuracy"]
                
                if "avg_bertscore_f1" in results.get("aggregates", {}):
                    comparison["bertscore_f1"][retriever][chunking] = results["aggregates"]["avg_bertscore_f1"]
    
    return comparison

def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG answers with fixed metrics")
    parser.add_argument("--answers_dir", required=True, help="Directory containing answer files")
    parser.add_argument("--retrieval_dir", required=True, help="Directory containing retrieval files")
    parser.add_argument("--ground_truth", default=None, help="Path to ground truth file")
    parser.add_argument("--output_dir", default=None, help="Directory to save results")
    parser.add_argument("--max_chunks", type=int, default=20, help="Maximum number of chunks to use")
    
    args = parser.parse_args()
    
    # Set output directory
    output_dir = args.output_dir or os.path.join(args.answers_dir, "fixed_evaluations")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load ground truth if provided
    ground_truth = {}
    if args.ground_truth:
        ground_truth = load_ground_truth(args.ground_truth)
    
    # Load NLP model if available
    nlp = None
    if HAS_SPACY:
        try:
            nlp = spacy.load("en_core_web_md")
            logger.info("Loaded spaCy model")
        except Exception as e:
            logger.error(f"Error loading spaCy model: {str(e)}")
    
    # Load BERTScore if available
    bert_scorer = None
    if HAS_BERTSCORE:
        try:
            bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
            logger.info("Loaded BERTScore model")
        except Exception as e:
            logger.error(f"Error loading BERTScore model: {str(e)}")
    
    # Find answer files
    answer_files = []
    for filename in os.listdir(args.answers_dir):
        if filename.endswith("_answers.json") and not filename.startswith(("evaluation_", "answers_summary", "fixed_")):
            answer_files.append(os.path.join(args.answers_dir, filename))
    
    logger.info(f"Found {len(answer_files)} answer files")
    
    # Process each answer file
    all_results = {}
    for answer_file in answer_files:
        try:
            file_basename = os.path.basename(answer_file)
            results = evaluate_answer_file(
                answer_file=answer_file,
                retrieval_dir=args.retrieval_dir,
                ground_truth=ground_truth,
                nlp=nlp,
                bert_scorer=bert_scorer,
                max_chunks=args.max_chunks
            )
            
            # Save individual results
            if results:
                output_path = os.path.join(output_dir, f"fixed_{file_basename}")
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2)
                
                logger.info(f"Saved evaluation for {file_basename} to {output_path}")
                
                # Add to all results
                retriever_name = results.get("retriever_name", file_basename)
                all_results[retriever_name] = results
        except Exception as e:
            logger.error(f"Error processing {answer_file}: {str(e)}")
    
    # Create comparison tables
    comparison = create_comparison_table(all_results)
    
    # Save comparison tables
    for metric, data in comparison.items():
        output = []
        
        # Get unique retrievers and chunking types
        retrievers = sorted(data.keys())
        chunking_types = set()
        for retriever_data in data.values():
            chunking_types.update(retriever_data.keys())
        chunking_types = sorted(chunking_types)
        
        # Create header
        header = ["Retriever"] + chunking_types
        output.append(",".join(header))
        
        # Create rows
        for retriever in retrievers:
            row = [retriever]
            for chunking in chunking_types:
                value = data[retriever].get(chunking, "")
                row.append(f"{value:.4f}" if isinstance(value, float) else "")
            output.append(",".join(row))
        
        # Save to file
        output_path = os.path.join(output_dir, f"comparison_{metric}.csv")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(output))

#!/usr/bin/env python3
"""
SimpleRAGEval - A simplified evaluation script for RAG systems

This script evaluates the quality of RAG (Retrieval-Augmented Generation) systems
by comparing generated answers with both retrieval chunks and ground truth.

Usage:
    python simple_rag_eval.py --answers_dir <path> --retrieval_dir <path> [options]

Key metrics:
    - Faithfulness: How well responses use information from retrieved chunks
    - Ground Truth Accuracy: How well responses match reference answers
    - BERTScore: Semantic similarity between responses and references

Author: Created for RAG evaluation with minimal dependencies
"""

import os
import json
import argparse
import re
import sys
from collections import Counter

def setup_argparse():
    """Set up command-line argument parsing."""
    parser = argparse.ArgumentParser(description="Simple RAG Evaluation")
    parser.add_argument("--answers_dir", required=True, help="Directory with answer files")
    parser.add_argument("--retrieval_dir", required=True, help="Directory with retrieval files")
    parser.add_argument("--ground_truth", default=None, help="Path to ground truth file (optional)")
    parser.add_argument("--output_dir", default=None, help="Directory to save results")
    parser.add_argument("--max_chunks", type=int, default=20, help="Maximum chunks to use")
    parser.add_argument("--top_k", type=int, default=5, help="Top K chunks for metrics")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    return parser.parse_args()

def load_ground_truth(filepath):
    """Load ground truth answers from JSON file."""
    print(f"Loading ground truth from: {filepath}")
    if not os.path.exists(filepath):
        print(f"Error: Ground truth file not found: {filepath}")
        return {}
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        ground_truth = {}
        for query_id, item in data.items():
            if isinstance(item, dict) and "answer" in item:
                ground_truth[query_id] = item["answer"]
            elif isinstance(item, str):
                ground_truth[query_id] = item
        
        print(f"Loaded {len(ground_truth)} ground truth answers")
        return ground_truth
    except Exception as e:
        print(f"Error loading ground truth: {str(e)}")
        return {}

def tokenize(text):
    """Simple tokenization function."""
    if not text:
        return []
    # Convert to lowercase
    text = text.lower()
    # Replace punctuation with spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    # Split and filter empty tokens
    return [token for token in text.split() if token.strip()]

def calculate_overlap(response_tokens, reference_tokens):
    """Calculate token overlap between response and reference."""
    if not response_tokens:
        return 0.0
    
    # Count tokens
    response_counter = Counter(response_tokens)
    reference_counter = Counter(reference_tokens)
    
    # Calculate intersection
    intersection = sum((response_counter & reference_counter).values())
    
    # Calculate overlap as portion of response tokens found in reference
    return intersection / len(response_tokens)

def find_matching_retrieval_file(retrieval_dir, retriever_type, chunking_type):
    """Find matching retrieval file for a given retriever and chunking type."""
    for filename in os.listdir(retrieval_dir):
        if not filename.endswith("_retrieval.json"):
            continue
            
        if (retriever_type.lower() in filename.lower() and 
            chunking_type.lower() in filename.lower()):
            return os.path.join(retrieval_dir, filename)
    
    # Try looser matching
    for filename in os.listdir(retrieval_dir):
        if not filename.endswith("_retrieval.json"):
            continue
            
        if retriever_type.lower() in filename.lower():
            return os.path.join(retrieval_dir, filename)
    
    return None

def process_answer_file(filename, answers_dir, retrieval_dir, ground_truth, max_chunks=20, verbose=False):
    """Process a single answer file and calculate metrics."""
    filepath = os.path.join(answers_dir, filename)
    print(f"Processing: {filename}")
    
    # Extract retriever and chunking type from filename
    parts = filename.replace("__", "_").split("_")
    if len(parts) >= 3 and parts[-1] == "answers.json":
        retriever_type = parts[0]
        chunking_type = parts[1]
    else:
        retriever_type = "unknown"
        chunking_type = "unknown"
        
    print(f"  Retriever: {retriever_type}, Chunking: {chunking_type}")
    
    # Load answer file
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            answer_data = json.load(f)
    except Exception as e:
        print(f"  Error loading answer file: {str(e)}")
        return None
    
    # Find matching retrieval file
    retrieval_file = find_matching_retrieval_file(retrieval_dir, retriever_type, chunking_type)
    if not retrieval_file:
        print("  No matching retrieval file found!")
        return None
    
    print(f"  Found retrieval file: {os.path.basename(retrieval_file)}")
    
    # Load retrieval data
    try:
        with open(retrieval_file, 'r', encoding='utf-8') as f:
            retrieval_data = json.load(f)
    except Exception as e:
        print(f"  Error loading retrieval file: {str(e)}")
        return None
    
    # Extract chunks by query ID
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
    
    print(f"  Extracted chunk data for {len(chunks_by_query)} queries")
    
    # Process answers and calculate metrics
    results = {
        "retriever_type": retriever_type,
        "chunking_type": chunking_type,
        "evaluations": {},
        "summary": {}
    }
    
    faithfulness_scores = []
    accuracy_scores = []
    answer_count = 0
    
    for query_id, answer_item in answer_data.get("answers", {}).items():
        answer_text = answer_item.get("answer", "")
        if not answer_text:
            continue
            
        # Get chunks for this query
        chunks = chunks_by_query.get(query_id, [])
        combined_chunks = "\n\n".join(chunks)
        
        # Calculate faithfulness
        answer_tokens = tokenize(answer_text)
        chunks_tokens = tokenize(combined_chunks)
        faithfulness = calculate_overlap(answer_tokens, chunks_tokens)
        
        # Calculate ground truth accuracy if available
        accuracy = None
        if query_id in ground_truth:
            gt_answer = ground_truth[query_id]
            gt_tokens = tokenize(gt_answer)
            accuracy = calculate_overlap(answer_tokens, gt_tokens)
            accuracy_scores.append(accuracy)
        
        # Store results
        results["evaluations"][query_id] = {
            "faithfulness": faithfulness,
            "gt_accuracy": accuracy,
            "chunks_used": len(chunks)
        }
        
        faithfulness_scores.append(faithfulness)
        answer_count += 1
        
        if verbose:
            print(f"  Query {query_id}: Faithfulness={faithfulness:.4f}, Accuracy={accuracy:.4f if accuracy else 'N/A'}")
    
    # Calculate summary metrics
    if faithfulness_scores:
        avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores)
        results["summary"]["avg_faithfulness"] = avg_faithfulness
        print(f"  Average faithfulness: {avg_faithfulness:.4f}")
    
    if accuracy_scores:
        avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)
        results["summary"]["avg_accuracy"] = avg_accuracy
        print(f"  Average accuracy: {avg_accuracy:.4f}")
    
    results["summary"]["answer_count"] = answer_count
    print(f"  Processed {answer_count} answers")
    
    return results

def create_comparison_tables(all_results):
    """Create comparison tables for different metrics and retriever/chunking combinations."""
    metrics = ["faithfulness", "accuracy"]
    tables = {}
    
    for metric in metrics:
        # Get unique retriever and chunking types
        retrievers = sorted({r["retriever_type"] for r in all_results.values()})
        chunking_types = sorted({r["chunking_type"] for r in all_results.values()})
        
        # Create table
        table = []
        header = ["Retriever"] + chunking_types
        table.append(",".join(header))
        
        for retriever in retrievers:
            row = [retriever]
            for chunking in chunking_types:
                # Find matching result
                value = ""
                for result in all_results.values():
                    if (result["retriever_type"] == retriever and 
                        result["chunking_type"] == chunking):
                        metric_key = f"avg_{metric}"
                        if metric_key in result.get("summary", {}):
                            value = f"{result['summary'][metric_key]:.4f}"
                row.append(value)
            table.append(",".join(row))
        
        tables[metric] = table
    
    return tables

def main():
    """Main function to run the evaluation."""
    args = setup_argparse()
    
    print("\n=== Simple RAG Evaluation ===")
    print(f"Answers directory: {args.answers_dir}")
    print(f"Retrieval directory: {args.retrieval_dir}")
    print(f"Max chunks: {args.max_chunks}")
    
    if not os.path.exists(args.answers_dir):
        print(f"Error: Answers directory not found: {args.answers_dir}")
        return
    
    if not os.path.exists(args.retrieval_dir):
        print(f"Error: Retrieval directory not found: {args.retrieval_dir}")
        return
    
    # Set output directory
    output_dir = args.output_dir or os.path.join(args.answers_dir, "simple_eval")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load ground truth if provided
    ground_truth = {}
    if args.ground_truth:
        ground_truth = load_ground_truth(args.ground_truth)
    
    # Find answer files
    answer_files = []
    for filename in os.listdir(args.answers_dir):
        if filename.endswith("_answers.json") and not filename.startswith(("evaluation_", "fixed_", "simple_")):
            answer_files.append(filename)
    
    if not answer_files:
        print("No answer files found!")
        return
    
    print(f"Found {len(answer_files)} answer files")
    
    # Process each answer file
    all_results = {}
    for filename in answer_files:
        try:
            result = process_answer_file(
                filename=filename,
                answers_dir=args.answers_dir,
                retrieval_dir=args.retrieval_dir,
                ground_truth=ground_truth,
                max_chunks=args.max_chunks,
                verbose=args.verbose
            )
            
            if result:
                # Save individual results
                output_path = os.path.join(output_dir, f"simple_eval_{filename}")
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2)
                
                print(f"Saved evaluation to {output_path}")
                all_results[filename] = result
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    # Create and save comparison tables
    if all_results:
        print("\nCreating comparison tables...")
        tables = create_comparison_tables(all_results)
        
        for metric, table in tables.items():
            output_path = os.path.join(output_dir, f"comparison_{metric}.csv")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(table))
            print(f"Saved {metric} comparison table to {output_path}")
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import os
import argparse
import json
import glob
from metadata.metadata_evaluator import MetadataEvaluator
from utils.logger import setup_logger
from gpu_utils import GPUVerifier
from tabulate import tabulate

# Initialize GPU verification
gpu_verifier = GPUVerifier(require_gpu=True)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Metadata Evaluation Tool")
    parser.add_argument("--metadata_dir", type=str, default="metadata_gen_output",
                        help="Directory containing metadata subdirectories")
    parser.add_argument("--evaluation_dir", type=str, default="evaluation",
                        help="Directory to store evaluation results")
    parser.add_argument("--eval_file", type=str, default="evaluation/metadata_evaluation.json",
                        help="Path to evaluation results JSON file")
    parser.add_argument("--run_eval", action="store_true",
                        help="Run metadata evaluation")
    parser.add_argument("--show_results", action="store_true",
                        help="Display results after evaluation")
    return parser.parse_args()

def run_evaluation():
    args = parse_arguments()
    logger = setup_logger("MetadataEvalTool")
    logger.info("Starting standalone metadata evaluation")
    
    # Create evaluator
    evaluator = MetadataEvaluator(args.evaluation_dir)
    
    # Get chunking subdirectories
    chunking_dirs = [d for d in os.listdir(args.metadata_dir) 
                    if os.path.isdir(os.path.join(args.metadata_dir, d)) 
                    and "chunks_metadata" in d]
    
    if not chunking_dirs:
        logger.error(f"No chunking directories found in {args.metadata_dir}")
        return
    
    logger.info(f"Found {len(chunking_dirs)} chunking directories: {', '.join(chunking_dirs)}")
    
    # Evaluate each chunking method
    results = {}
    for chunking_dir in chunking_dirs:
        logger.info(f"Evaluating {chunking_dir}")
        dir_path = os.path.join(args.metadata_dir, chunking_dir)
        dir_results = evaluator._evaluate_file_directory(dir_path)
        results[chunking_dir] = dir_results
    
    # Create aggregate results
    aggregate = evaluator._aggregate_results(results)
    
    # Save combined results
    final_results = {
        "individual_results": results,
        "aggregate_results": aggregate
    }
    
    output_path = os.path.join(args.evaluation_dir, "metadata_evaluation.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2)
    
    # Fix this part at the end of run_evaluation()
    logger.info(f"Saved evaluation results to {output_path}")

    # Generate visualizations
    evaluator._generate_visualizations(aggregate)

    # Fix these lines - accessing the correct metrics path
    if "completeness_metrics" in aggregate and "overall_completeness" in aggregate.get("completeness_metrics", {}):
        logger.info(f"Overall completeness: {aggregate['completeness_metrics']['overall_completeness']:.2f}%")
    else:
        # Check aggregate structure and try alternative paths
        avg_completeness = aggregate.get("avg_completeness", 0)
        logger.info(f"Overall completeness: {avg_completeness:.2f}%")

    if "intent_coverage" in aggregate and "coverage_percentage" in aggregate.get("intent_coverage", {}):
        logger.info(f"Intent coverage: {aggregate['intent_coverage']['coverage_percentage']:.2f}%")
    else:
        # Try alternative path
        avg_intent_coverage = aggregate.get("avg_intent_coverage", 0)
        logger.info(f"Intent coverage: {avg_intent_coverage:.2f}%")

    logger.info("Metadata evaluation completed")


def display_results():
    args = parse_arguments()
    
    # Load evaluation results
    if not os.path.exists(args.eval_file):
        print(f"Error: Evaluation file not found: {args.eval_file}")
        return
    
    with open(args.eval_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # Extract results by chunking method
    individual_results = results.get("individual_results", {})
    
    # Display results for each chunking method
    for chunking_method, method_results in individual_results.items():
        print(f"\n===== {chunking_method.upper()} =====")
        
        # Create a table for file metrics
        table_data = []
        headers = ["File", "Chunks", "Completeness", "Intent Coverage", "Unique Types", "Avg Keywords"]
        
        # Process each file
        for filename, file_results in method_results.items():
            chunks = file_results.get("total_chunks", 0)
            completeness = file_results.get("completeness_metrics", {}).get("overall_completeness", 0)
            intent_coverage = file_results.get("intent_coverage", {}).get("coverage_percentage", 0)
            unique_types = file_results.get("diversity_metrics", {}).get("unique_content_types", 0)
            avg_keywords = file_results.get("keyword_statistics", {}).get("avg_keywords_per_chunk", 0)
            
            table_data.append([
                filename,
                chunks,
                f"{completeness:.2f}%",
                f"{intent_coverage:.2f}%",
                unique_types,
                f"{avg_keywords:.2f}"
            ])
        
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        # Display summary info for this chunking method
        total_chunks = sum(file_results.get("total_chunks", 0) for file_results in method_results.values())
        avg_completeness = sum(file_results.get("completeness_metrics", {}).get("overall_completeness", 0) 
                            for file_results in method_results.values()) / len(method_results) if method_results else 0
        
        print(f"\nTotal chunks: {total_chunks}")
        print(f"Average completeness: {avg_completeness:.2f}%")
        
        # Display top content types
        content_types = {}
        for file_results in method_results.values():
            distribution = file_results.get("diversity_metrics", {}).get("content_type_distribution", {})
            for content_type, count in distribution.items():
                content_types[content_type] = content_types.get(content_type, 0) + count
        
        print("\nTop content types:")
        for content_type, count in sorted(content_types.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"- {content_type}: {count}")    

if __name__ == "__main__":
    run_evaluation()
    display_results()
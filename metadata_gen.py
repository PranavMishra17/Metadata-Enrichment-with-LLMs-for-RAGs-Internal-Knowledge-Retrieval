#!/usr/bin/env python3
import os
import argparse
import time
import json

from metadata.llm_metadata_generator import LLMMetadataGenerator
from metadata.metadata_evaluator import MetadataEvaluator
from utils.logger import setup_logger

from gpu_utils import GPUVerifier
# Initialize GPU verification
gpu_verifier = GPUVerifier(require_gpu=True)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Metadata generation tool")
    
    # Optional arguments
    parser.add_argument("--chunks_dir", type=str, default="chunk_output", 
                        help="Directory containing chunk files")
    parser.add_argument("--output_dir", type=str, default="metadata_gen_output", 
                        help="Directory to store enriched output")
    parser.add_argument("--evaluation_dir", type=str, default="evaluation", 
                        help="Directory to store evaluation results")
    parser.add_argument("--evaluate", action="store_true", 
                        help="Evaluate metadata after generation")
    
    return parser.parse_args()

def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logger
    logger = setup_logger("MetadataGenTool")
    logger.info("Starting metadata generation tool")
    
    # Create output directories
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        logger.info(f"Created output directory: {args.output_dir}")
    
    if not os.path.exists(args.evaluation_dir):
        os.makedirs(args.evaluation_dir)
        logger.info(f"Created evaluation directory: {args.evaluation_dir}")
    
    # Check if chunks directory exists
    if not os.path.exists(args.chunks_dir):
        logger.error(f"Chunks directory does not exist: {args.chunks_dir}")
        return
    
    # Initialize metadata generator
    try:
        start_time = time.time()
        metadata_generator = LLMMetadataGenerator(output_dir=args.output_dir)
        
        # Process chunks
        metadata_generator.process_chunks(chunks_dir=args.chunks_dir)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        logger.info(f"Metadata generation completed in {processing_time:.2f} seconds")
        
        # Save processing metrics
        metrics = {
            "processing_time": processing_time,
            "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "chunks_directory": args.chunks_dir,
            "output_directory": args.output_dir
        }
        
        metrics_path = os.path.join(args.evaluation_dir, "metadata_gen_metrics.json")
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Saved processing metrics to {metrics_path}")
        
        # Run evaluation if requested
        if args.evaluate:
            logger.info("Starting metadata evaluation")
            evaluator = MetadataEvaluator(evaluation_dir=args.evaluation_dir)
            evaluation_results = evaluator.evaluate_metadata(args.output_dir)
            
            logger.info("Metadata evaluation completed")
            logger.info(f"Overall completeness: {evaluation_results.get('avg_completeness', 0):.2f}%")
            logger.info(f"Intent coverage: {evaluation_results.get('avg_intent_coverage', 0):.2f}%")
        
    except Exception as e:
        logger.error(f"Error in metadata generation: {str(e)}")

if __name__ == "__main__":
    main()
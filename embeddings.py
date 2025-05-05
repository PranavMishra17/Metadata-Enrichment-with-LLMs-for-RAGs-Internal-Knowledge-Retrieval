#!/usr/bin/env python3
import os
import argparse
import json
import time
from typing import List, Dict, Any

from embedding.naive_embedder import NaiveEmbedder
from embedding.tfidf_embedder import TfidfEmbedder
from embedding.prefix_embedder import PrefixEmbedder
from utils.logger import setup_logger

from gpu_utils import GPUVerifier

# Initialize GPU verification
gpu_verifier = GPUVerifier(require_gpu=True)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Embedding generation tool")
    
    # Input/output directories
    parser.add_argument("--input_dir", type=str, default="metadata_gen_output",
                        help="Input directory containing enriched chunks")
    parser.add_argument("--output_dir", type=str, default="embeddings_output",
                        help="Output directory for embeddings")
    
    # Chunking types to process
    parser.add_argument("--chunking_types", type=str, nargs="+", 
                      default=["semantic", "naive", "recursive"],
                      help="Chunking types to process")
    
    # Embedding types to generate
    parser.add_argument("--embedding_types", type=str, nargs="+",
                      default=["naive", "tfidf", "prefix"],
                      help="Types of embeddings to generate (naive, tfidf, prefix)")
    
    # Model selection
    parser.add_argument("--model", type=str, default="Snowflake/arctic-embed-s",
                      help="SentenceTransformer model to use")
    
    # TF-IDF weights
    parser.add_argument("--content_weight", type=float, default=0.7,
                      help="Weight for content embeddings in TF-IDF approach")
    parser.add_argument("--tfidf_weight", type=float, default=0.3,
                      help="Weight for TF-IDF embeddings in TF-IDF approach")
    
    # Evaluation
    parser.add_argument("--evaluate", action="store_true",
                      help="Run evaluation after generating embeddings")
    
    return parser.parse_args()

def create_embeddings(args):
    """Create embeddings based on arguments."""
    logger = setup_logger("EmbeddingTool")
    logger.info("Starting embedding generation")
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        logger.error(f"Input directory does not exist: {args.input_dir}")
        return
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        logger.info(f"Created output directory: {args.output_dir}")
    
    # Process each chunking type
    for chunking_type in args.chunking_types:
        chunking_dir = os.path.join(args.input_dir, f"{chunking_type}_chunks_metadata")
        if not os.path.exists(chunking_dir):
            logger.warning(f"Chunking directory does not exist: {chunking_dir}")
            continue
        
        logger.info(f"Processing {chunking_type} chunks")
        
        # Process each embedding type
        for embedding_type in args.embedding_types:
            try:
                start_time = time.time()
                
                if embedding_type.lower() == "naive":
                    logger.info(f"Generating naive embeddings for {chunking_type} chunks")
                    embedder = NaiveEmbedder(
                        input_dir=args.input_dir,
                        output_dir=args.output_dir,
                        chunking_type=chunking_type,
                        model_name=args.model
                    )
                    
                elif embedding_type.lower() == "tfidf":
                    logger.info(f"Generating TF-IDF embeddings for {chunking_type} chunks")
                    embedder = TfidfEmbedder(
                        input_dir=args.input_dir,
                        output_dir=args.output_dir,
                        chunking_type=chunking_type,
                        model_name=args.model,
                        content_weight=args.content_weight,
                        tfidf_weight=args.tfidf_weight
                    )
                    
                elif embedding_type.lower() == "prefix":
                    logger.info(f"Generating prefix-fusion embeddings for {chunking_type} chunks")
                    embedder = PrefixEmbedder(
                        input_dir=args.input_dir,
                        output_dir=args.output_dir,
                        chunking_type=chunking_type,
                        model_name=args.model
                    )
                    
                else:
                    logger.warning(f"Unknown embedding type: {embedding_type}")
                    continue
                
                # Process chunks
                index_info = embedder.process_all_chunks()
                
                # Calculate processing time
                end_time = time.time()
                processing_time = end_time - start_time
                
                logger.info(f"Completed {embedding_type} embeddings for {chunking_type} in {processing_time:.2f} seconds")
                
                # Run evaluation if requested
                if args.evaluate and index_info:
                    logger.info(f"Running evaluation for {embedding_type} embeddings of {chunking_type} chunks")
                    # Import evaluation module only when needed
                    from embedding.evaluator import EmbeddingEvaluator
                    evaluator = EmbeddingEvaluator(args.output_dir)
                    evaluator.evaluate_embeddings(chunking_type, embedding_type)
                
            except Exception as e:
                logger.error(f"Error processing {embedding_type} embeddings for {chunking_type}: {str(e)}")
    
    logger.info("Embedding generation completed")

def main():
    """Main entry point."""
    args = parse_arguments()
    create_embeddings(args)

if __name__ == "__main__":
    main()


'''
# Basic usage - generate all embedding types for all chunking methods
python embeddings.py

# Generate specific embedding types
python embeddings.py --embedding_types naive tfidf

# Generate embeddings for specific chunking methods
python embeddings.py --chunking_types semantic

# Customize embedding model
python embeddings.py --model all-MiniLM-L6-v2

# Customize TF-IDF weights
python embeddings.py --embedding_types tfidf --content_weight 0.6 --tfidf_weight 0.4

# Run with evaluation
python embeddings.py --evaluate

# Available options
--input_dir       Input directory containing enriched chunks (default: metadata_gen_output)
--output_dir      Output directory for embeddings (default: embeddings_output)
--chunking_types  Chunking types to process (default: semantic naive recursive)
--embedding_types Types of embeddings to generate (default: naive tfidf prefix)
--model           SentenceTransformer model to use (default: Snowflake/arctic-embed-s)
--content_weight  Weight for content embeddings in TF-IDF approach (default: 0.7)
--tfidf_weight    Weight for TF-IDF embeddings in TF-IDF approach (default: 0.3)
--evaluate        Run evaluation after generating embeddings


'''
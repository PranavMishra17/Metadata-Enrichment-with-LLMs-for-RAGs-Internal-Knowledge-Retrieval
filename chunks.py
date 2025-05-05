
#!/usr/bin/env python3
import os
import argparse
import json
from typing import Dict, Any
import glob

from chunking.naive_chunker import NaiveChunker
from chunking.recursive_chunker import RecursiveChunker
from chunking.semantic_chunker import SemanticChunker
from chunking.chunk_evaluator import ChunkEvaluator
from utils.logger import setup_logger
from utils.pdf_utils import pdf_to_text

from gpu_utils import GPUVerifier
# Initialize GPU verification
gpu_verifier = GPUVerifier(require_gpu=True)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Document chunking tool")
    
    # Input file - now optional with default
    parser.add_argument("--input_file", type=str, default="input_files",
                        help="Path to input document or directory (default: input_files)")
    
    # Optional arguments
    parser.add_argument("--chunking_method", type=str, default="naive", 
                        choices=["naive", "recursive", "semantic"], 
                        help="Chunking method to use")
    parser.add_argument("--output_dir", type=str, default="chunk_output", 
                        help="Directory to store chunked output")
    
    # Naive chunker parameters
    parser.add_argument("--chunk_by", type=str, default="paragraph", 
                        choices=["paragraph", "sentence"], 
                        help="Method for naive chunking")
    
    # Recursive chunker parameters
    parser.add_argument("--split_method", type=str, default="length", 
                        choices=["length", "delimiter"], 
                        help="Method for recursive splitting")
    parser.add_argument("--max_chunk_length", type=int, default=1000, 
                        help="Maximum length of a chunk in characters (for recursive chunker)")
    
    # Semantic chunker parameters
    parser.add_argument("--sentence_model", type=str, default="all-MiniLM-L6-v2", 
                        help="Sentence transformer model to use for semantic chunking")
    parser.add_argument("--percentile_threshold", type=float, default=95, 
                        help="Percentile threshold for identifying semantic breakpoints")
    parser.add_argument("--context_window", type=int, default=1, 
                        help="Number of sentences to consider for context in semantic chunking")
    
    # Common parameters
    parser.add_argument("--min_chunk_size", type=int, default=2, 
                        help="Minimum number of sentences in a chunk")
    parser.add_argument("--max_chunk_size", type=int, default=10, 
                        help="Maximum number of sentences in a chunk")
    parser.add_argument("--overlap", type=int, default=1, 
                        help="Number of sentences to overlap between chunks")
    
    # Evaluation
    parser.add_argument("--evaluate", action="store_true", 
                        help="Evaluate chunks after creation")
    
    return parser.parse_args()

def get_chunker(args):
    """Get the appropriate chunker based on arguments."""
    if args.chunking_method == "naive":
        return NaiveChunker(
            chunk_by=args.chunk_by,
            min_chunk_size=args.min_chunk_size,
            max_chunk_size=args.max_chunk_size,
            overlap=args.overlap,
            output_dir=args.output_dir
        )
    elif args.chunking_method == "recursive":
        return RecursiveChunker(
            split_method=args.split_method,
            max_chunk_length=args.max_chunk_length,
            min_chunk_size=args.min_chunk_size,
            max_chunk_size=args.max_chunk_size,
            overlap=args.overlap,
            output_dir=args.output_dir
        )
    elif args.chunking_method == "semantic":
        return SemanticChunker(
            sentence_model=args.sentence_model,
            percentile_threshold=args.percentile_threshold,
            context_window=args.context_window,
            min_chunk_size=args.min_chunk_size,
            max_chunk_size=args.max_chunk_size,
            output_dir=args.output_dir
        )
    else:
        raise ValueError(f"Unknown chunking method: {args.chunking_method}")


def process_files(args, logger):
    """Process input files based on arguments."""
    input_path = args.input_file
    files_to_process = []
    
    # Check if input is a directory or a file
    if os.path.isdir(input_path):
        # Process all text and PDF files in directory
        text_files = glob.glob(os.path.join(input_path, "*.txt"))
        pdf_files = glob.glob(os.path.join(input_path, "*.pdf"))
        
        # Convert PDF files to text
        converted_files = []
        for pdf_file in pdf_files:
            try:
                text_file = pdf_to_text(pdf_file)
                converted_files.append(text_file)
                logger.info(f"Converted PDF to text: {pdf_file} -> {text_file}")
            except Exception as e:
                logger.error(f"Failed to convert PDF to text: {pdf_file}. Error: {str(e)}")
        
        # Combine text files and converted PDF files
        files_to_process = text_files + converted_files
        logger.info(f"Found {len(text_files)} text files and converted {len(pdf_files)} PDF files in directory: {input_path}")
    
    elif os.path.isfile(input_path):
        # Process single file (check if it's a PDF and convert if necessary)
        if input_path.lower().endswith('.pdf'):
            try:
                text_file = pdf_to_text(input_path)
                files_to_process = [text_file]
                logger.info(f"Converted PDF to text: {input_path} -> {text_file}")
            except Exception as e:
                logger.error(f"Failed to convert PDF to text: {input_path}. Error: {str(e)}")
                return []
        else:
            files_to_process = [input_path]
            logger.info(f"Processing single file: {input_path}")
    else:
        logger.error(f"Input path does not exist: {input_path}")
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    
    if not files_to_process:
        logger.warning(f"No files to process")
        return []
    
    # Get chunker
    chunker = get_chunker(args)
    logger.info(f"Using {args.chunking_method} chunking method")
    
    # Process each file
    results = []
    for file_path in files_to_process:
        try:
            logger.info(f"Processing file: {file_path}")
            result = chunker.process_document(file_path)
            output_path = chunker.save_chunks(result)
            logger.info(f"Saved chunks to: {output_path}")
            results.append((file_path, output_path, result))
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
    
    return results


def evaluate_chunks(results, logger):
    """Evaluate chunks using the ChunkEvaluator."""
    logger.info("Evaluating chunks...")
    
    evaluator = ChunkEvaluator()
    evaluation_results = {}
    
    for file_path, output_path, result in results:
        file_name = os.path.basename(file_path)
        
        # Evaluate chunks
        metrics = evaluator.evaluate_chunks(result["chunks"])
        
        # Add document metadata to metrics
        metrics["document_name"] = file_name
        metrics["chunking_method"] = result["metadata"]["chunking_method"]
        metrics["total_chunks"] = result["metadata"]["total_chunks"]
        
        # Save evaluation results
        eval_path = output_path.replace(".json", "_evaluation.json")
        with open(eval_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Saved evaluation results to: {eval_path}")
        evaluation_results[file_name] = metrics
    
    # Print summary
    if evaluation_results:
        logger.info("\nEvaluation Summary:")
        for file_name, metrics in evaluation_results.items():
            logger.info(f"File: {file_name}")
            logger.info(f"  Chunking Method: {metrics['chunking_method']}")
            logger.info(f"  Total Chunks: {metrics['total_chunks']}")
            logger.info(f"  Avg Chunk Size (words): {metrics['avg_chunk_size_words']:.2f}")
            logger.info(f"  Coherence Score: {metrics.get('coherence_score', 'N/A')}")
            logger.info(f"  Inter-chunk Similarity: {metrics.get('avg_inter_chunk_similarity', 'N/A'):.2f}")
            logger.info(f"  Intra-chunk Similarity: {metrics.get('avg_intra_chunk_similarity', 'N/A'):.2f}")
            logger.info("")
    
    return evaluation_results


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logger
    logger = setup_logger("ChunkingTool")
    logger.info("Starting document chunking tool")
    
    # Create input directory if it doesn't exist
    input_dir = "input_files"
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
        logger.info(f"Created input directory: {input_dir}")
    
    logger.info(f"Looking for files in: {args.input_file}")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        logger.info(f"Created output directory: {args.output_dir}")
    
    # Process files
    results = process_files(args, logger)
    
    if not results:
        logger.warning("No files were successfully processed")
        return
    
    # Evaluate chunks if requested
    if args.evaluate:
        evaluation_results = evaluate_chunks(results, logger)
    else:
        # Ask user if they want to evaluate chunks
        user_input = input("\nDo you want to evaluate chunks? (y/n): ")
        if user_input.lower() in ["y", "yes"]:
            evaluation_results = evaluate_chunks(results, logger)
    
    logger.info("Document chunking completed")



# Initialize GPU verification
gpu_verifier = GPUVerifier(require_gpu=True)
if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import os
import json
import time
import logging
import argparse
from typing import List, Dict, Any
import concurrent.futures
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RAG-Answerer")

class LLMProcessor:
    """Class to handle LLM interactions."""
    
    def __init__(self, model_name=None, temperature=0.5, max_retries=3, retry_delay=5):
        self.model_name = model_name or config.AZURE_DEPLOYMENT
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.client = None
        self.initialize_llm()
    
    def initialize_llm(self):
        """Initialize the LLM client."""
        try:
            self.client = AzureChatOpenAI(
                azure_deployment=config.AZURE_DEPLOYMENT,
                api_key=config.AZURE_API_KEY,
                api_version=config.AZURE_API_VERSION,
                azure_endpoint=config.AZURE_ENDPOINT,
                temperature=self.temperature
            )
            logger.info("LLM initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            self.client = None
    
    def generate_answer(self, query: str, context: str) -> str:
        """Generate an answer based on query and context."""
        if not self.client:
            logger.error("LLM client not initialized")
            return "Error: LLM not available"
        
        # Create system and user messages
        system_msg = "You are a knowledgeable and confident assistant. Provide accurate answers to queries based on the information available to you. Always offer the best possible response, focusing on being helpful rather than discussing limitations. Make the user feel supported and well-informed."
        user_msg = f"Query: {query}\n\nContext:\n{context}"
        
        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content=user_msg)
        ]
        
        # Try to generate with retries
        for attempt in range(self.max_retries):
            try:
                response = self.client.invoke(messages)
                return response.content
            except Exception as e:
                logger.error(f"Error generating answer (attempt {attempt+1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
        
        return "Error: Failed to generate answer after multiple attempts"

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="RAG Answer Generator")
    
    parser.add_argument("--retrieval_dir", type=str, required=True,
                       help="Directory containing retrieval outputs")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory to store answer outputs")
    parser.add_argument("--top_k", type=int, default=10,
                       help="Number of top chunks to use for context")
    parser.add_argument("--threads", type=int, default=4,
                       help="Number of parallel threads")
    parser.add_argument("--rate_limit", type=float, default=10,
                       help="Maximum requests per minute")
    
    return parser.parse_args()

def load_retrieval_files(retrieval_dir: str) -> List[str]:
    """Load all retrieval result files."""
    retrieval_files = []
    
    for filename in os.listdir(retrieval_dir):
        if filename.endswith("_retrieval.json"):
            retrieval_files.append(os.path.join(retrieval_dir, filename))
    
    return retrieval_files


def process_retrieval_file(file_path: str, top_k: int, llm_processor: LLMProcessor) -> Dict[str, Any]:
    """Process a single retrieval file."""
    logger.info(f"Processing {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        retriever_name = data.get("run_info", {}).get("retriever_name", "unknown")
        answers = {}
        
        for query_id, query_data in data.get("queries", {}).items():
            query_text = query_data.get("query_text", "")
            chunks = query_data.get("retrieved_chunks", [])
            
            logger.info(f"Query {query_id}: {len(chunks)} chunks available in file")
            
            # Count chunks with text
            chunks_with_text = sum(1 for chunk in chunks if chunk.get("text", "").strip())
            logger.info(f"Query {query_id}: {chunks_with_text} chunks have non-empty text")
            
            # Extract text from top-k chunks
            context_texts = []
            for i, chunk in enumerate(chunks[:top_k]):
                chunk_text = chunk.get("text", "").strip()
                if chunk_text:
                    context_texts.append(chunk_text)
                else:
                    logger.warning(f"Chunk {i} (ID: {chunk.get('chunk_id', 'unknown')}) has empty text")
            
            # Create context string
            context = "\n\n".join(context_texts)
            
            logger.info(f"Query {query_id}: Using {len(context_texts)} chunks out of {len(chunks)} available")
            
            # Generate answer
            answer = llm_processor.generate_answer(query_text, context)
            
            # Store result
            answers[query_id] = {
                "query": query_text,
                "answer": answer,
                "num_chunks_used": len(context_texts),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        
        return {
            "retriever_name": retriever_name,
            "answers": answers
        }
    
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return {
            "retriever_name": os.path.basename(file_path),
            "error": str(e),
            "answers": {}
        }


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Set default output directory if not specified
    if args.output_dir is None:
        args.output_dir = os.path.join(args.retrieval_dir, "answers")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize LLM processor
    llm_processor = LLMProcessor()
    
    # Load retrieval files
    retrieval_files = load_retrieval_files(args.retrieval_dir)
    
    if not retrieval_files:
        logger.error(f"No retrieval files found in {args.retrieval_dir}")
        return
    
    # Calculate rate limit delay
    rate_limit_delay = 60.0 / args.rate_limit
    
    # Process files with thread pool
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = []
        
        for file_path in retrieval_files:
            future = executor.submit(process_retrieval_file, file_path, args.top_k, llm_processor)
            futures.append(future)
            # Apply rate limiting
            time.sleep(rate_limit_delay)
        
        # Collect results
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                retriever_name = result.get("retriever_name", "unknown")
                
                # Save individual result
                sanitized_name = retriever_name.replace(" ", "_").replace("(", "_").replace(")", "_")
                output_path = os.path.join(args.output_dir, f"{sanitized_name}_answers.json")
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2)
                
                logger.info(f"Saved answers for {retriever_name} to {output_path}")
            
            except Exception as e:
                logger.error(f"Error processing result: {str(e)}")
    
    # Create summary file
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "retriever_count": len(retrieval_files),
        "top_k": args.top_k,
        "output_dir": args.output_dir
    }
    
    summary_path = os.path.join(args.output_dir, "answers_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Answer generation completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
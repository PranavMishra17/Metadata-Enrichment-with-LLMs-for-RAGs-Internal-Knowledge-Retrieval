import re
from typing import List, Dict, Any, Tuple
import nltk
from nltk.tokenize import sent_tokenize

from chunking.base_chunker import BaseChunker

from gpu_utils import GPUVerifier

# Initialize GPU verification
gpu_verifier = GPUVerifier(require_gpu=True)

class NaiveChunker(BaseChunker):
    """Naive chunking strategy that splits text by paragraphs or fixed number of sentences."""


    
    def __init__(
        self,
        chunk_by: str = "paragraph",
        min_chunk_size: int = 2,
        max_chunk_size: int = 10,
        overlap: int = 1,
        output_dir: str = "chunk_output"
    ):
        """Initialize naive chunker."""
        super().__init__(min_chunk_size, max_chunk_size, overlap, output_dir)
        self.chunk_by = chunk_by
        
        # Download NLTK resources if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            self.logger.info("Downloading NLTK punkt tokenizer")
            nltk.download('punkt')
    
    def _create_chunks(self, text: str, document_id: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Create chunks from text using naive chunking."""
        chunks = []
        
        if self.chunk_by == "paragraph":
            # Split by paragraphs (double newlines)
            paragraphs = re.split(r'\n\s*\n', text)
            
            # Process each paragraph as a chunk
            for i, para in enumerate(paragraphs):
                para = para.strip()
                if not para:
                    continue
                
                # Get paragraph position in original text
                start_idx = text.find(para)
                end_idx = start_idx + len(para)
                
                # Count sentences in paragraph
                sentences = sent_tokenize(para)
                
                chunk = {
                    "chunk_id": f"{document_id}_chunk_{i}",
                    "text": para,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "metadata": {
                        "num_sentences": len(sentences),
                        "num_words": len(para.split()),
                        "source": "paragraph"
                    }
                }
                chunks.append(chunk)
                
        elif self.chunk_by == "sentence":
            # Split by sentences
            sentences = sent_tokenize(text)
            
            # Create chunks of max_chunk_size sentences with overlap
            for i in range(0, len(sentences), self.max_chunk_size - self.overlap):
                chunk_sentences = sentences[i:i + self.max_chunk_size]
                
                if len(chunk_sentences) < self.min_chunk_size:
                    # Skip if too few sentences left
                    continue
                
                chunk_text = " ".join(chunk_sentences)
                
                # Get chunk position in original text
                first_sentence = chunk_sentences[0]
                last_sentence = chunk_sentences[-1]
                start_idx = text.find(first_sentence)
                end_idx = text.find(last_sentence) + len(last_sentence)
                
                chunk = {
                    "chunk_id": f"{document_id}_chunk_{i//(self.max_chunk_size - self.overlap)}",
                    "text": chunk_text,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "metadata": {
                        "num_sentences": len(chunk_sentences),
                        "num_words": len(chunk_text.split()),
                        "source": "fixed_sentence"
                    }
                }
                chunks.append(chunk)
        else:
            self.logger.error(f"Invalid chunking method: {self.chunk_by}")
            raise ValueError(f"Invalid chunking method: {self.chunk_by}. Must be 'paragraph' or 'sentence'.")
        
        # Document metadata
        document_metadata = {
            "total_paragraphs": len(re.split(r'\n\s*\n', text)) if self.chunk_by == "paragraph" else 0,
            "total_sentences": len(sent_tokenize(text)),
            "total_words": len(text.split()),
            "chunking_strategy": f"naive_{self.chunk_by}"
        }
        
        return chunks, document_metadata
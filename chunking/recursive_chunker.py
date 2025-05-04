import re
import nltk
from nltk.tokenize import sent_tokenize
from typing import List, Dict, Any, Tuple

from chunking.base_chunker import BaseChunker

from gpu_utils import GPUVerifier
# Initialize GPU verification
gpu_verifier = GPUVerifier(require_gpu=True)


class RecursiveChunker(BaseChunker):
    """Recursive chunking strategy that recursively splits text into smaller chunks."""

    
    def __init__(
        self,
        split_method: str = "length",  # "length" or "delimiter"
        max_chunk_length: int = 1000,
        delimiters: List[str] = None,
        min_chunk_size: int = 2,
        max_chunk_size: int = 10,
        overlap: int = 1,
        output_dir: str = "chunk_output"
    ):
        """Initialize recursive chunker."""
        super().__init__(min_chunk_size, max_chunk_size, overlap, output_dir)
        self.split_method = split_method
        self.max_chunk_length = max_chunk_length
        self.delimiters = delimiters or ["##", "#", "\\n\\n", "\\n", "\\. ", "; ", ", "]
        
        # Download NLTK resources if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            self.logger.info("Downloading NLTK punkt tokenizer")
            nltk.download('punkt')
    
    def _create_chunks(self, text: str, document_id: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Create chunks from text using recursive chunking."""
        chunks = []
        chunk_counter = 0
        
        # Start recursive chunking
        self._recursive_split(text, document_id, 0, len(text), chunks, chunk_counter)
        
        # Sort chunks by start_idx
        chunks.sort(key=lambda x: x["start_idx"])
        
        # Fix chunk IDs after sorting
        for i, chunk in enumerate(chunks):
            chunk["chunk_id"] = f"{document_id}_chunk_{i}"
        
        # Document metadata
        document_metadata = {
            "total_sentences": len(sent_tokenize(text)),
            "total_words": len(text.split()),
            "chunking_strategy": f"recursive_{self.split_method}",
            "max_chunk_length": self.max_chunk_length if self.split_method == "length" else None,
            "delimiters_used": self.delimiters if self.split_method == "delimiter" else None
        }
        
        return chunks, document_metadata
    
    def _recursive_split(
        self, 
        text: str, 
        document_id: str, 
        start_idx: int, 
        end_idx: int, 
        chunks: List[Dict[str, Any]], 
        chunk_counter: int, 
        depth: int = 0
    ) -> int:
        """Recursively split text into chunks."""
        # Get text segment
        segment = text[start_idx:end_idx]
        
        # Base case: If segment is short enough or can't be split further
        if (self.split_method == "length" and len(segment) <= self.max_chunk_length) or depth > 10:
            sentences = sent_tokenize(segment)
            
            # Check if chunk has minimum number of sentences
            if len(sentences) >= self.min_chunk_size:
                chunk = {
                    "chunk_id": f"{document_id}_chunk_{chunk_counter}",
                    "text": segment,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "metadata": {
                        "num_sentences": len(sentences),
                        "num_words": len(segment.split()),
                        "recursion_depth": depth
                    }
                }
                chunks.append(chunk)
            else:
                self.logger.debug(f"Skipping too small segment at depth {depth} (only {len(sentences)} sentences)")
            
            return chunk_counter + 1
        
        # Recursive case: split and process
        split_index = None
        
        if self.split_method == "length":
            # Split approximately in half
            mid_point = len(segment) // 2
            
            # Find a good splitting point near the middle
            candidates = []
            
            # Try paragraph breaks first
            para_breaks = [m.start() for m in re.finditer(r'\n\s*\n', segment)]
            candidates.extend((p, 4) for p in para_breaks)  # Higher priority
            
            # Try sentence breaks
            sentence_breaks = []
            sentences = sent_tokenize(segment)
            
            current_pos = 0
            for s in sentences:
                current_pos = segment.find(s, current_pos) + len(s)
                sentence_breaks.append(current_pos)
            
            candidates.extend((p, 3) for p in sentence_breaks)  # Medium priority
            
            # Try other punctuation
            for punct, priority in [('. ', 2), ('; ', 1), (', ', 0)]:
                punct_breaks = [m.start() + len(punct) - 1 for m in re.finditer(re.escape(punct), segment)]
                candidates.extend((p, priority) for p in punct_breaks)
            
            # Find closest candidate to mid_point with highest priority
            if candidates:
                # Sort by priority (descending) then by distance to mid_point (ascending)
                candidates.sort(key=lambda x: (-x[1], abs(x[0] - mid_point)))
                split_index = candidates[0][0]
            else:
                # Fall back to exact mid-point if no good candidates
                split_index = mid_point
                
        elif self.split_method == "delimiter":
            # Try each delimiter in order
            for delimiter in self.delimiters:
                # Clean up delimiter for regex
                clean_delimiter = delimiter.replace("\\n", "\n").replace("\\.", ".").replace("\\s", "\\s")
                
                # Find all matches of the delimiter
                matches = list(re.finditer(re.escape(clean_delimiter), segment))
                
                if matches:
                    # Use the last match as the split point
                    split_index = matches[-1].start()
                    break
            
            # If no delimiter found, fall back to middle
            if split_index is None:
                split_index = len(segment) // 2
        
        # Recursively process the two halves
        if split_index and split_index > 0 and split_index < len(segment):
            # Process first half
            chunk_counter = self._recursive_split(
                text, document_id, start_idx, start_idx + split_index, 
                chunks, chunk_counter, depth + 1
            )
            
            # Process second half
            chunk_counter = self._recursive_split(
                text, document_id, start_idx + split_index, end_idx, 
                chunks, chunk_counter, depth + 1
            )
        else:
            # If split point is invalid, treat as base case
            sentences = sent_tokenize(segment)
            
            if len(sentences) >= self.min_chunk_size:
                chunk = {
                    "chunk_id": f"{document_id}_chunk_{chunk_counter}",
                    "text": segment,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "metadata": {
                        "num_sentences": len(sentences),
                        "num_words": len(segment.split()),
                        "recursion_depth": depth
                    }
                }
                chunks.append(chunk)
                chunk_counter += 1
        
        return chunk_counter
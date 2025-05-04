# chunking package
from .naive_chunker import NaiveChunker
from .recursive_chunker import RecursiveChunker
from .semantic_chunker import SemanticChunker
from .chunk_evaluator import ChunkEvaluator

from gpu_utils import GPUVerifier

# Initialize GPU verification
gpu_verifier = GPUVerifier(require_gpu=True)

__all__ = ['NaiveChunker', 'RecursiveChunker', 'SemanticChunker', 'ChunkEvaluator']
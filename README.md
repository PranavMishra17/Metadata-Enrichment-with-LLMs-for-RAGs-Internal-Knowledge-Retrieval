# Metadata Enrichment with LLMs

This project implements an AI-powered chatbot for internal knowledge retrieval, with a focus on enhancing document processing through chunking, metadata enrichment, embedding, retrieval, and evaluation.

## Project Overview

The project is structured as a pipeline with the following components:

1. **Chunking**: Breaking documents into meaningful segments
2. **Metadata Enrichment**: Enhancing chunks with descriptive metadata
3. **Embedding**: Creating vector representations of chunks
4. **Retrieval**: Implementing a retrieval system for chunks
5. **Retrieval Evaluation**: Assessing retrieval quality
6. **Prompting**: Generating appropriate prompts for the LLM
7. **Ground Truth Generation**: Creating test datasets
8. **Evaluation**: Comprehensive system evaluation

## Current Implementation

The current release focuses on the **Chunking** component, with support for:

- **Semantic Chunking**: Groups text by semantic similarity using sentence embeddings
- **Recursive Chunking**: Recursively splits text into smaller chunks using various criteria
- **Naive Chunking**: Simple chunking by paragraphs or sentences

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/metadata-enrichment-llm.git
cd metadata-enrichment-llm

# Install dependencies
pip install -r requirements.txt

# Download required NLTK resources (optional, will be downloaded automatically when needed)
python -c "import nltk; nltk.download('punkt')"
```

## Usage
### Chunking Documents

The system automatically creates an `input_files` directory where you can place your documents. It supports both text files (.txt) and PDF files (.pdf).

```bash
# Basic usage
python chunks.py --input_file path/to/document.txt --chunking_method naive

# Using semantic chunking
python chunks.py --input_file path/to/document.txt --chunking_method semantic --sentence_model paraphrase-MiniLM-L3-v2 --percentile_threshold 95

# Using recursive chunking
python chunks.py --input_file path/to/document.txt --chunking_method recursive --split_method length --max_chunk_length 1000

# Process all text and PDF files in the input_files directory
python chunks.py --chunking_method naive --chunk_by paragraph --evaluate

# Process all text and PDF files in a custom directory
python chunks.py --input_file path/to/documents/ --chunking_method naive --chunk_by paragraph --evaluate
```

When processing PDF files, the system automatically converts them to text files in the same directory before chunking.

### Command Line Arguments

- `--input_file`: Path to input document or directory
- `--chunking_method`: Chunking method to use (`naive`, `recursive`, or `semantic`)
- `--output_dir`: Directory to store chunked output (default: `chunk_output`)

#### Naive Chunker Parameters
- `--chunk_by`: Method for naive chunking (`paragraph` or `sentence`)

#### Recursive Chunker Parameters
- `--split_method`: Method for recursive splitting (`length` or `delimiter`)
- `--max_chunk_length`: Maximum length of a chunk in characters

#### Semantic Chunker Parameters
- `--sentence_model`: Sentence transformer model to use
- `--percentile_threshold`: Percentile threshold for identifying semantic breakpoints (default: 95)
- `--context_window`: Number of sentences to consider for context (default: 1)

#### Common Parameters
- `--min_chunk_size`: Minimum number of sentences in a chunk
- `--max_chunk_size`: Maximum number of sentences in a chunk
- `--overlap`: Number of sentences to overlap between chunks
- `--evaluate`: Evaluate chunks after creation

## Chunking Strategies

### 1. Semantic Chunking

Groups sentences by semantic similarity using sentence embeddings and clustering algorithms. This approach ensures that semantically related content stays together.

**Key Components:**
- Uses `sentence-transformers` models for embedding sentences
- Identifies semantic breakpoints where shifts in meaning occur
- Merges smaller chunks with their most semantically similar neighbors
- Calculates coherence scores for chunks

### 2. Recursive Chunking

Recursively splits text based on various criteria such as length or delimiters, ensuring that the resulting chunks are balanced.

**Key Components:**
- Supports length-based or delimiter-based splitting
- Recursively processes text until chunks meet size requirements
- Uses punctuation and paragraph breaks as potential split points

### 3. Naive Chunking

Simple chunking strategies based on paragraphs or fixed number of sentences.

**Key Components:**
- Paragraph-based chunking follows document structure
- Sentence-based chunking creates fixed-size chunks
- Supports configurable overlap between chunks

## Chunk Evaluation

The system provides evaluation metrics for assessing chunk quality:

- **Size Metrics**: Statistics about chunk sizes (words, sentences)
- **Overlap Metrics**: Measures of content duplication between chunks
- **Coherence Metrics**: 
  - **Inter-chunk similarity**: How similar chunks are to each other (lower is better)
  - **Intra-chunk similarity**: How coherent sentences are within each chunk (higher is better)
- **Visualization**: Optional plots showing similarity distributions

## Project Structure

```
metadata-enrichment-llm/
├── README.md
├── requirements.txt
├── chunks.py
├── input_files/     # Place your PDFs and text files here
├── chunk_output/    # Chunked outputs are saved here
├── logs/            # Log files directory
├── evaluation_plots/ # Evaluation visualizations
├── chunking/
│   ├── __init__.py
│   ├── base_chunker.py
│   ├── semantic_chunker.py
│   ├── recursive_chunker.py
│   ├── naive_chunker.py
│   └── chunk_evaluator.py
└── utils/
    ├── __init__.py
    ├── logger.py
    └── pdf_utils.py # PDF handling utilities
```

## Future Work

The following components will be implemented in future releases:

1. **Metadata Enrichment**: Enhancing chunks with descriptive metadata using LLMs
2. **Embedding**: Creating vector representations of chunks
3. **Retrieval**: Implementing a retrieval system for chunks
4. **Retrieval Evaluation**: Assessing retrieval quality
5. **Prompting**: Generating appropriate prompts for the LLM
6. **Ground Truth Generation**: Creating test datasets
7. **Evaluation**: Comprehensive system evaluation

## License

[MIT License](LICENSE)
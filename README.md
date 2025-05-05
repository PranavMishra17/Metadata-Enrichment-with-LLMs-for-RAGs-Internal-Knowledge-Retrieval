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

# Chunking Strategies

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

# Metadata Enrichment with LLMs - Metadata Generation

This document outlines the metadata generation phase of the Metadata Enrichment with LLMs pipeline, which enhances document chunks with rich semantic and structural metadata to improve retrieval and answer composition.

## Overview

The metadata generation system:

1. Reads chunked documents from the `chunk_output` directory
2. Processes chunks by chunking method (semantic, naive, recursive)
3. Generates rich metadata using LLMs
4. Saves enriched chunks to organized output directories
5. Optionally evaluates metadata quality

## Installation

```bash
# Install required dependencies
pip install -r requirements.txt

# Set up environment variables (.env file in project root)
AZURE_API_KEY=your_key
AZURE_ENDPOINT=your_endpoint
AZURE_DEPLOYMENT=your_deployment
AZURE_API_VERSION=your_api_version
TEMPERATURE=0.0
BATCH_SIZE=10
REQUEST_TIMEOUT=30
RETRY_LIMIT=3
RETRY_DELAY=5
```

## Usage

```bash
# Basic usage
python metadata_gen.py

# Custom paths and evaluation
python metadata_gen.py --chunks_dir path/to/chunks --output_dir path/to/output --evaluate

# Available options
--chunks_dir      Directory containing chunk files (default: chunk_output)
--output_dir      Directory to store enriched output (default: metadata_gen_output)
--evaluation_dir  Directory to store evaluation results (default: evaluation)
--evaluate        Run evaluation after metadata generation
```

## Directory Structure

```
metadata-enrichment-llm/
├── metadata_gen.py               # Main entry point
├── .env                          # Environment variables
├── chunk_output/                 # Input chunks from chunking phase
├── metadata_gen_output/
│   ├── semantic_chunks_metadata/ # Enriched semantic chunks
│   ├── naive_chunks_metadata/    # Enriched naive chunks 
│   └── recursive_chunks_metadata/# Enriched recursive chunks
└── evaluation/                   # Evaluation metrics and visualizations
```

## Metadata Structure

Each chunk is enriched with the following metadata:

```json
{
  "chunk_id": "unique-id",
  "text": "original content",
  "metadata": {
    "content": {
      "content_type": {
        "primary": "Procedural|Conceptual|Reference|Warning|Example",
        "subtypes": ["Setup Guide", "Configuration", ...]
      },
      "keywords": ["term1", "term2", ...],
      "entities": ["Entity1", "Entity2", ...],
      "has_code": true|false
    },
    "technical": {
      "primary_category": "Category",
      "secondary_categories": ["Category1", "Category2"],
      "mentioned_services": ["Service1", "Service2", ...],
      "mentioned_tools": ["Tool1", "Tool2", ...]
    },
    "semantic": {
      "summary": "Concise summary of content",
      "intents": ["How-To", "Debug", "Compare", "Reference"],
      "potential_questions": ["Question1?", "Question2?", ...]
    }
  },
  "embedding_enhancement": {
    "contextual_prefix": "[ContentType] [Category]",
    "tf_idf_keywords": ["keyword1", "keyword2", ...]
  }
}
```

## Metadata Components

### Content Metadata
- **Content Type**: Categorizes content as Procedural, Conceptual, Reference, Warning, or Example
- **Keywords**: Important technical terms and concepts
- **Entities**: Named entities (products, services, tools)
- **Code Detection**: Identifies presence of code examples

### Technical Metadata
- **Primary Category**: Main technical category
- **Secondary Categories**: Related technical categories
- **Mentioned Services**: Specific services referenced
- **Mentioned Tools**: Development tools mentioned

### Semantic Metadata
- **Summary**: Concise abstract of the content
- **Intents**: User intents this content addresses (How-To, Debug, Compare, Reference)
- **Potential Questions**: Questions this content can answer

### Embedding Enhancement
- **Contextual Prefix**: Prepended text for embedding enhancement
- **TF-IDF Keywords**: Keywords for TF-IDF vector enhancement (30% weighting)

## Evaluation

The metadata evaluation system assesses:

1. **Completeness**: Percentage of fields that are properly populated
2. **Diversity**: Variety of content types, categories, and intents
3. **Intent Coverage**: Coverage of standard user intents (How-To, Debug, Compare, Reference)
4. **Keyword Statistics**: Analysis of keywords and their distribution

Evaluation results are saved to the evaluation directory and include visualizations of:
- Field completeness
- Content type distribution
- Category distribution
- Intent distribution
- Keyword distribution

## Implementation Details

### LLM Integration

The system uses Azure OpenAI services for metadata generation with:
- Rate limit handling
- Batch processing
- Retry mechanism

### Error Handling

The implementation includes:
- Proper error logging
- Graceful handling of rate limits
- Fallbacks for failed metadata generation

### Performance Considerations

- Processes chunks in batches to respect API limits
- Uses checkpoints to resume processing
- Keeps different chunking methods separate for clean evaluation






# Embedding System

This document outlines the embedding generation phase of the Metadata Enrichment with LLMs pipeline, implementing a dual-embedding approach for technical documentation with rich metadata.

## Overview

The embedding system generates three types of vector representations for each chunking method:

1. **Naive Embeddings**: Basic content-only embeddings
2. **TF-IDF Weighted Embeddings**: Combines content (70%) and metadata keywords (30%)
3. **Prefix-Fusion Embeddings**: Injects formatted metadata prefixes into content before embedding

All embeddings are stored in FAISS indexes for efficient similarity search and retrieval.

## Installation

```bash
# Install required dependencies
pip install -r requirements.txt

# Required packages include:
# - sentence-transformers
# - faiss-cpu (or faiss-gpu)
# - scikit-learn
# - matplotlib
# - numpy
```

## Directory Structure

```
metadata-enrichment-llm/
├── embeddings.py                   # Main entry point
├── embeddings/                     # Embedding modules
│   ├── base_embedder.py            # Base embedding class
│   ├── naive_embedder.py           # Content-only embeddings
│   ├── tfidf_embedder.py           # TF-IDF weighted embeddings
│   ├── prefix_embedder.py          # Prefix-injection embeddings
│   └── evaluator.py                # Embedding evaluation utilities
├── metadata_gen_output/            # Input enriched chunks
│   ├── semantic_chunks_metadata/   # Semantic chunking results
│   ├── naive_chunks_metadata/      # Naive chunking results
│   └── recursive_chunks_metadata/  # Recursive chunking results
└── embeddings_output/              # Generated embeddings
    ├── semantic/                   # Semantic chunking embeddings
    │   ├── naive_embedding/        # Naive embeddings
    │   ├── tfidf_embedding/        # TF-IDF embeddings
    │   └── prefix_fusion_embedding/# Prefix embeddings
    ├── naive/                      # [Similar structure]
    └── recursive/                  # [Similar structure]
```

## Usage

```bash
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
```

## Embedding Approaches

### 1. Naive Embeddings

Simple content-only embeddings that serve as a baseline:

- Uses raw chunk text without metadata
- No special preprocessing or weighting
- Fastest to generate but less retrieval-focused

### 2. TF-IDF Weighted Embeddings

Combines content and metadata in vector space:

- Content embeddings (70% weight)
- TF-IDF vectors from metadata (30% weight)
- Metadata sources:
  - Technical keywords (40%)
  - Named entities (25%)
  - Technical categories (20%)
  - Question keywords (15%)

### 3. Prefix-Fusion Embeddings

Injects structured metadata prefixes into text:

- Intent prefixes (25%): `[Intent:HowTo]`
- Service context (20%): `[Service:S3|IAM]`
- Content type (15%): `[Procedural]`
- Technical category (10%): `[CloudStorage]`
- Code presence (10%): `[Code:Python]`
- Potential questions (20%): `[Q:howDoIConfigureS3Versioning]`

## FAISS Index Structure

Each embedding type generates a FAISS index with:

- `index.faiss`: The FAISS vector index for similarity search
- `id_mapping.pkl`: Mappings between chunk IDs and index positions
- `metadata.json`: Essential chunk metadata for retrieval
- `document_list.json`: List of processed documents

## Evaluation

When run with the `--evaluate` flag, the system evaluates:

1. **Metadata Consistency**: How well the embedding space preserves metadata relationships
2. **Nearest Neighbor Statistics**: Distribution of distances in the embedding space

Evaluation results are saved to `evaluation/{chunking_type}/{embedding_type}_evaluation.json` with visualizations in `evaluation/{chunking_type}/visualizations/`.

## Implementation Notes

- All models use embeddings of dimension 384 by default
- Dimension matching is done automatically for TF-IDF vectors
- Metadata is normalized and formatted consistently across embedding types
- FAISS uses inner product (dot product) for cosine similarity

## Best Practices

- **Model Selection**: The default `Snowflake/arctic-embed-s` model works well for technical content, but you can substitute other SentenceTransformer models
- **Chunking Method**: Semantic chunking typically works best with prefix-fusion embeddings
- **TF-IDF Weights**: The default 70/30 split balances content and metadata well, but you can adjust based on your retrieval needs
- **Evaluation**: Always evaluate embeddings to understand their characteristics before using in production

## Example Workflow

A typical workflow might look like:

1. Process documents with semantic chunking
2. Generate metadata for chunks
3. Create all three embedding types
4. Evaluate and compare the embedding types
5. Select the best performing embedding type for your retrieval system

The system is designed to be modular, allowing you to experiment with different combinations of chunking methods and embedding strategies.

# Retriever System

This section outlines the retrieval phase of the Metadata Enrichment with LLMs pipeline, implementing multiple retrieval strategies with comprehensive evaluation capabilities.

## Overview

The retrieval system provides three main retrieval approaches:

1. **Content-Only Retrieval**: Basic text similarity using naive embeddings
2. **Content + Metadata Retrieval**: 
   - TF-IDF Weighted: Combines content (70%) and metadata (30%) vectors
   - Prefix-Fusion: Uses metadata-injected embeddings 
3. **Content + Metadata + Reranker**: Adds cross-encoder reranking for improved relevance

Each approach can be used with any of the chunking methods (semantic, naive, recursive).

## Installation

```bash
# Install required dependencies
pip install -r requirements.txt

# Required packages include:
# - sentence-transformers
# - faiss-cpu (or faiss-gpu)
# - scikit-learn
# - numpy
# - scipy
# - matplotlib
```

## Directory Structure

```
metadata-enrichment-llm/
├── retriever.py                   # Main entry point
├── retrieval/                     # Retrieval modules
│   ├── base_retriever.py          # Base retriever class
│   ├── content_retriever.py       # Content-only retriever
│   ├── tfidf_retriever.py         # TF-IDF weighted retriever
│   ├── prefix_retriever.py        # Prefix-fusion retriever
│   ├── reranker_retriever.py      # Reranker wrapper
│   └── evaluator.py               # Evaluation utilities
├── embeddings_output/             # Input embeddings
│   ├── semantic/                  # Semantic chunking embeddings
│   ├── naive/                     # Naive chunking embeddings
│   └── recursive/                 # Recursive chunking embeddings
└── retrieval_output/              # Generated results
    └── run_[timestamp]/           # Run-specific outputs
        ├── Content_(semantic)_results.json
        ├── TF-IDF_(naive)_results.json
        ├── Prefix-Fusion_(recursive)_results.json
        ├── Reranker_(semantic)_results.json
        ├── run_[id]_content_semantic_evaluation.json
        ├── run_[id]_tfidf_naive_evaluation.json
        ├── run_[id]_prefix_recursive_evaluation.json
        ├── run_[id]_reranker_semantic_evaluation.json
        └── retriever_comparison.json
```

## Usage

### Basic Retrieval
```bash
# Basic usage - run all retrievers with default parameters
python retriever.py

# Run with custom queries file  (4 threads)
python retriever.py --queries_file sample_queries.json

# Run with more threads for faster processing
python retriever.py --queries_file questions.json --threads 8

# Run specific retrievers and chunking types
python retriever.py --retrievers content prefix reranker --chunking_types semantic

# Customize results parameters
python retriever.py --top_k 10 --reranker_k 30


# Available options
--embedding_dir     Directory containing embeddings (default: embeddings_output)
--output_dir        Directory to store retrieval results (default: retrieval_output)
--queries_file      JSON file containing queries and relevance judgments
--retrievers        Retrievers to use (content, tfidf, prefix, reranker)
--chunking_types    Chunking types to use (semantic, naive, recursive)
--top_k             Number of results to retrieve (default: 5)
--reranker_k        Number of initial results for reranker (default: 20)
--model             Embedding model to use (default: Snowflake/arctic-embed-s)
--reranker_model    Reranker model name (default: cross-encoder/ms-marco-MiniLM-L-6-v2)
--run_id            Unique ID for this evaluation run
--threads           Number of parallel threads to use (default: 4)
```
### Custom Retrieval Configuration
```bash
# Run specific retrievers with high parallelism
python retriever.py --queries_file questions.json --retrievers prefix reranker --threads 12

# Run on specific chunking types with custom top-k
python retriever.py --queries_file questions.json --chunking_types semantic --top_k 10 --threads 6
```

### Evaluation
```bash
# After running retrieval, evaluate the results:
python retriever_eval.py --input_dir retrieval_output/run_12345678

# Evaluate with relevance judgments:
python retriever_eval.py --input_dir retrieval_output/run_12345678 --relevance_file relevance.json
```

## Sample Questions JSON Format

The system accepts queries in JSON format with the following structure:

```json
[
  {"id": "q1", "query": "How do I create a bucket in S3?"},
  {"id": "q2", "query": "What is Amazon S3 Glacier?"},
  {"id": "q3", "query": "How to upload files to S3?"}
]
```

The multithreading option `--threads` controls how many retrievers run in parallel. Higher values will process queries faster but use more system resources. The optimal value depends on your machine's capabilities and the number of retrievers you're running.

## Retriever Types

### 1. Content Retriever

Simple content-based retrieval using naive embeddings:

- Uses raw chunk text without metadata
- Provides a baseline for comparison
- Available with all chunking methods

### 2. TF-IDF Retriever

Combines content and metadata using a weighted approach:

- Content embedding (70% weight)
- TF-IDF vector from metadata (30% weight)
- Preserves semantic meaning while adding keyword focus
- Uses pre-computed tf-idf embeddings

### 3. Prefix-Fusion Retriever

Uses embeddings that incorporate metadata as prefixes:

- Automatically detects query intent and injects appropriate prefixes
- Formats prefixes similar to those used during embedding
- Strengthens content-metadata connection

### 4. Reranker Retriever

Adds cross-encoder reranking on top of another retriever:

- First retrieves a larger candidate set (default: 20 results)
- Then reranks using cross-encoder model
- Returns the top-k reranked results (default: 5)
- Uses MS-MARCO-MiniLM-L-6-v2 by default

## Evaluation Metrics

The system evaluates retrieval performance using:

### Core IR Metrics

- **Contextual Precision (CP@K)**: Proportion of relevant results in top-k
- **Mean Reciprocal Rank (MRR)**: Position of first relevant result
- **Normalized Discounted Cumulative Gain (NDCG@K)**: Relevance-weighted ranking quality
- **Recall@K**: Proportion of all relevant documents retrieved in top-k

### AWS-Specific Metrics

- **Chunk Utilization Rate**: Diversity of chunks in results
- **API Element Recall**: Coverage of API elements mentioned in query
- **Metadata Consistency Score**: Consistency of metadata across results

## Retriever Comparison

After evaluation, the system generates a comparison report identifying:

- Performance of each retriever on key metrics
- Best retriever for each metric
- Overall best retriever based on average ranking

## Resuming Interrupted Runs

The system uses checkpoints to save progress during retrieval:

- Results are saved after every 5 queries
- If interrupted, running again with the same run_id will resume
- Use `--eval_only` to evaluate existing results without re-running retrieval

## Performance Considerations

- Multithreaded retrieval for improved performance
- Configurable number of parallel threads
- Progress reporting with estimated time remaining

## Next Steps

The retrieval results generated by this system can be used as input for the answer generation phase, where an LLM will formulate coherent answers based on the retrieved chunks.

# Future Work





# License

[MIT License](LICENSE)
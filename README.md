# Metadata Enrichment with LLMs for RAG Systems

This repository implements a comprehensive framework for enhancing Retrieval-Augmented Generation (RAG) systems through metadata enrichment, advanced chunking strategies, and neural retrieval techniques.

<p align="center">
  <img src="assets/pipeline_overview.png" alt="Pipeline Overview" width="800"/>
</p>

## Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Project Components](#project-components)
  - [1. Chunking](#1-chunking)
  - [2. Metadata Enrichment](#2-metadata-enrichment)
  - [3. Embedding Generation](#3-embedding-generation)
  - [4. Retrieval System](#4-retrieval-system)
  - [5. Ground Truth Generation](#5-ground-truth-generation)
  - [6. Retrieval Evaluation](#6-retrieval-evaluation)
  - [7. Answer Generation](#7-answer-generation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Glossary](#glossary)

## Introduction

Retrieval-Augmented Generation (RAG) systems enhance LLM outputs by retrieving relevant context from external knowledge sources. This project focuses on improving the retrieval component through metadata enrichment, implementing a pipeline that:

1. Breaks documents into meaningful chunks using three distinct strategies
2. Enriches chunks with LLM-generated semantic metadata
3. Creates vector embeddings using multiple approaches
4. Implements and evaluates various retrieval methodologies
5. Generates ground truth for objective evaluation
6. Produces answers based on retrieved contexts

## Requirements

To install requirements:

```bash
# Clone the repository
git clone https://github.com/username/metadata-enrichment-llm.git
cd metadata-enrichment-llm

# Install dependencies
pip install -r requirements.txt

# Download required NLTK resources (optional)
python -c "import nltk; nltk.download('punkt')"

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

## Project Components

### 1. Chunking

Implements three distinct chunking strategies to process documents into meaningful segments:

- **Semantic Chunking**: Groups text by semantic similarity using sentence embeddings
- **Recursive Chunking**: Recursively splits text based on length or delimiters
- **Naive Chunking**: Simple chunking by paragraphs or sentences

#### Usage

```bash
# Basic usage
python chunks.py --input_file path/to/document.txt --chunking_method naive

# Using semantic chunking
python chunks.py --input_file path/to/document.txt --chunking_method semantic --sentence_model paraphrase-MiniLM-L3-v2 --percentile_threshold 95

# Using recursive chunking
python chunks.py --input_file path/to/document.txt --chunking_method recursive --split_method length --max_chunk_length 1000

# Process all files in a directory with evaluation
python chunks.py --input_file path/to/documents/ --chunking_method naive --chunk_by paragraph --evaluate
```

#### Command Line Arguments

- `--input_file`: Path to input document or directory
- `--chunking_method`: Chunking method to use (`naive`, `recursive`, or `semantic`)
- `--output_dir`: Directory to store chunked output (default: `chunk_output`)

##### Naive Chunker Parameters
- `--chunk_by`: Method for naive chunking (`paragraph` or `sentence`)

##### Recursive Chunker Parameters
- `--split_method`: Method for recursive splitting (`length` or `delimiter`)
- `--max_chunk_length`: Maximum length of a chunk in characters

##### Semantic Chunker Parameters
- `--sentence_model`: Sentence transformer model to use
- `--percentile_threshold`: Percentile threshold for identifying semantic breakpoints (default: 95)
- `--context_window`: Number of sentences to consider for context (default: 1)

#### Evaluation

The `chunk_eval.py` script provides comprehensive evaluation of chunking quality:

```bash
# Evaluate all chunks in the output directory
python chunk_eval.py --chunks_dir chunk_output

# Evaluate specific chunking method
python chunk_eval.py --chunks_dir chunk_output --chunking_method semantic

# Generate visualizations
python chunk_eval.py --chunks_dir chunk_output --visualize
```

##### Evaluation Arguments
- `--chunks_dir`: Directory containing chunks to evaluate (default: `chunk_output`)
- `--chunking_method`: Specific chunking method to evaluate (evaluates all if not specified)
- `--output_dir`: Directory to store evaluation results (default: `evaluation/chunking`)
- `--visualize`: Generate visualization plots
- `--min_sentences`: Minimum acceptable sentences per chunk for metrics (default: 3)
- `--max_sentences`: Maximum acceptable sentences per chunk for metrics (default: 20)

##### Metrics Calculated
- Size distribution (sentences, tokens, characters)
- Coherence scores (intra-chunk similarity)
- Overlap assessment
- Content coverage

### 2. Metadata Enrichment

Enhances chunks with rich semantic and structural metadata using LLMs:

- **Content Metadata**: Content type, keywords, entities, code detection
- **Technical Metadata**: Categories, services, tools
- **Semantic Metadata**: Summary, intents, potential questions

#### Usage

```bash
# Basic usage
python metadata_gen.py

# Custom paths and evaluation
python metadata_gen.py --chunks_dir path/to/chunks --output_dir path/to/output --evaluate

# Process specific chunking method
python metadata_gen.py --chunks_dir chunk_output --chunking_method semantic
```

#### Command Line Arguments

- `--chunks_dir`: Directory containing chunk files (default: `chunk_output`)
- `--output_dir`: Directory to store enriched output (default: `metadata_gen_output`)
- `--chunking_method`: Specific chunking method to process (processes all if not specified)
- `--batch_size`: Number of chunks to process in one batch (default: 10)
- `--retry_limit`: Number of retries for failed API calls (default: 3)
- `--evaluate`: Run evaluation after metadata generation

#### Evaluation

The `metadata_eval.py` script provides detailed evaluation of metadata quality:

```bash
# Evaluate all metadata in the output directory
python metadata_eval.py --metadata_dir metadata_gen_output

# Evaluate specific chunking method
python metadata_eval.py --metadata_dir metadata_gen_output --chunking_method semantic

# Generate detailed visualizations
python metadata_eval.py --metadata_dir metadata_gen_output --visualize --detailed_report
```

##### Evaluation Arguments
- `--metadata_dir`: Directory containing metadata to evaluate (default: `metadata_gen_output`)
- `--chunking_method`: Specific chunking method to evaluate (evaluates all if not specified)
- `--output_dir`: Directory to store evaluation results (default: `evaluation/metadata`)
- `--visualize`: Generate visualization plots
- `--detailed_report`: Generate detailed per-field analysis
- `--sample_size`: Number of chunks to sample for in-depth analysis (default: 100)

##### Metrics Calculated
- Metadata completeness (percentage of fields populated)
- Field consistency (variation across chunks)
- Content type distribution
- Intent coverage
- Keyword relevance
- Category distribution

### 3. Embedding Generation

Generates vector representations using three distinct approaches:

- **Naive Embeddings**: Basic content-only embeddings
- **TF-IDF Weighted**: Combines content (70%) and metadata (30%)
- **Prefix-Fusion**: Injects formatted metadata prefixes into content

#### Usage

```bash
# Basic usage - generate all embedding types for all chunking methods
python embeddings.py

# Generate specific embedding types
python embeddings.py --embedding_types naive tfidf

# Generate embeddings for specific chunking methods
python embeddings.py --chunking_types semantic

# Customize embedding model
python embeddings.py --model all-MiniLM-L6-v2
```

#### Command Line Arguments

- `--input_dir`: Input directory containing enriched chunks (default: `metadata_gen_output`)
- `--output_dir`: Output directory for embeddings (default: `embeddings_output`)
- `--chunking_types`: Chunking types to process (default: `semantic naive recursive`)
- `--embedding_types`: Types of embeddings to generate (default: `naive tfidf prefix`)
- `--model`: SentenceTransformer model to use (default: `Snowflake/arctic-embed-s`)
- `--content_weight`: Weight for content embeddings in TF-IDF approach (default: 0.7)
- `--tfidf_weight`: Weight for TF-IDF embeddings in TF-IDF approach (default: 0.3)
- `--evaluate`: Run evaluation after generating embeddings

#### Evaluation

The `embedding_eval.py` script analyzes embedding quality and characteristics:

```bash
# Evaluate all embeddings
python embedding_eval.py --embeddings_dir embeddings_output

# Evaluate specific chunking and embedding type
python embedding_eval.py --embeddings_dir embeddings_output --chunking_type semantic --embedding_type prefix_fusion

# Run in-depth nearest neighbor analysis
python embedding_eval.py --embeddings_dir embeddings_output --nn_analysis --top_k 50
```

##### Evaluation Arguments
- `--embeddings_dir`: Directory containing embeddings to evaluate (default: `embeddings_output`)
- `--chunking_type`: Specific chunking type to evaluate (evaluates all if not specified)
- `--embedding_type`: Specific embedding type to evaluate (evaluates all if not specified)
- `--output_dir`: Directory to store evaluation results (default: `evaluation/embeddings`)
- `--nn_analysis`: Perform nearest neighbor analysis
- `--top_k`: Number of nearest neighbors to analyze (default: 20)
- `--visualize`: Generate visualization plots
- `--sample_size`: Number of embeddings to sample for analysis (default: 100)

##### Metrics Calculated
- Embedding distribution statistics
- Clustering tendencies
- Semantic consistency with metadata
- Category separation
- Nearest neighbor relevance

### 4. Retrieval System

Implements multiple retrieval strategies with comprehensive evaluation:

- **Content-Only**: Basic text similarity using naive embeddings
- **Content+Metadata**: TF-IDF weighted or Prefix-Fusion retrievers
- **Reranking**: Adds cross-encoder reranking for improved relevance

#### Usage

```bash
# Basic usage - run all retrievers with default parameters
python retriever.py

# Run with custom queries file
python retriever.py --queries_file sample_queries.json

# Run specific retrievers and chunking types
python retriever.py --retrievers content prefix reranker --chunking_types semantic

# Customize results parameters
python retriever.py --top_k 10 --reranker_k 30

# Run with evaluation
python retriever.py --queries_file sample_queries.json --evaluate
```

#### Command Line Arguments

- `--embedding_dir`: Directory containing embeddings (default: `embeddings_output`)
- `--output_dir`: Directory to store retrieval results (default: `retrieval_output`)
- `--queries_file`: JSON file containing queries and relevance judgments
- `--retrievers`: Retrievers to use (`content`, `tfidf`, `prefix`, `reranker`)
- `--chunking_types`: Chunking types to use (`semantic`, `naive`, `recursive`)
- `--top_k`: Number of results to retrieve (default: 5)
- `--reranker_k`: Number of initial results for reranker (default: 20)
- `--model`: Embedding model to use (default: `Snowflake/arctic-embed-s`)
- `--reranker_model`: Reranker model name (default: `cross-encoder/ms-marco-MiniLM-L-6-v2`)
- `--run_id`: Unique ID for this evaluation run
- `--threads`: Number of parallel threads to use (default: 4)
- `--evaluate`: Run preliminary evaluation after retrieval
- `--eval_output`: Directory to store preliminary evaluation (default: `retrieval_output/run_ID/eval`)

### 5. Ground Truth Generation

Creates evaluation datasets using reranker-based assessment:

- Evaluates retrieved chunks against queries using a neural reranker
- Generates ground truth rankings for objective comparison
- Calculates statistics like percentiles and rank changes

#### Usage

```bash
# Basic usage
python retriever_gt.py --input_dir retrieval_output/run_ID --queries_file sample_q.json

# Custom output directory
python retriever_gt.py --input_dir retrieval_output/run_ID --output_dir custom/gt/path --queries_file sample_q.json

# Adjust evaluation parameters
python retriever_gt.py --input_dir retrieval_output/run_ID --top_k 50 --threads 8
```

#### Command Line Arguments

- `--input_dir`: Directory with retrieval output (required)
- `--output_dir`: Ground truth output directory (default: `input_dir/ground_truth`)
- `--queries_file`: JSON file with queries (default: `sample_q.json`)
- `--top_k`: Number of chunks to evaluate (default: 25)
- `--threads`: Number of parallel threads (default: 4)
- `--rate_limit`: Rate limit for model API calls (default: 2.0)

### 6. Retrieval Evaluation

Evaluates retriever performance against ground truth:

- Calculates IR metrics like Precision@K, Recall@K, MRR, NDCG@K, Hit Rate@K
- Generates comparative visualizations and tables
- Analyzes performance by chunking method and retriever type

#### Usage

```bash
# Basic usage
python retriever_eval.py --retrieval_dir retrieval_output/run_ID --ground_truth_dir retrieval_output/run_ID/ground_truth

# Custom output directory
python retriever_eval.py --retrieval_dir retrieval_output/run_ID --ground_truth_dir retrieval_output/run_ID/ground_truth --output_dir custom/eval/path

# Customize K values for evaluation
python retriever_eval.py --retrieval_dir retrieval_output/run_ID --ground_truth_dir retrieval_output/run_ID/ground_truth --k_values 1 3 5 10 25 50
```

#### Command Line Arguments

- `--retrieval_dir`: Directory containing retrieval results
- `--ground_truth_dir`: Directory with ground truth data
- `--output_dir`: Directory to store evaluation results (default: `retrieval_dir/evaluation`)
- `--k_values`: K values for metrics (default: `1 3 5 10 20`)
- `--threads`: Number of parallel threads (default: 4)
- `--detailed_report`: Generate detailed per-query analysis (default: False)

#### Metrics Calculated
- **Precision@K**: Proportion of relevant results in top-k
- **Recall@K**: Proportion of all relevant documents retrieved in top-k
- **MRR (Mean Reciprocal Rank)**: Position of first relevant result
- **NDCG@K**: Relevance-weighted ranking quality
- **Hit Rate@K**: Proportion of highly relevant documents in top-k
- **Metadata Consistency**: Consistency of document categories/types

### 7. Answer Generation

Generates answers using LLMs based on retrieved contexts:

- Extracts top-k chunks from retrieval outputs
- Formulates prompts with query and context
- Generates and stores LLM-produced answers

#### Usage

```bash
python prompt.py --retrieval_dir retrieval_output/run_ID --top_k 10
```

#### Command Line Arguments

- `--retrieval_dir`: Directory containing retrieval outputs
- `--output_dir`: Directory to store answer outputs
- `--top_k`: Number of top chunks to use for context (default: 10)
- `--threads`: Number of parallel threads (default: 4)
- `--rate_limit`: Maximum requests per minute (default: 10)

## Project Structure

```
metadata-enrichment-llm/
├── chunks.py                     # Chunking entry point
├── metadata_gen.py               # Metadata generation entry point
├── embeddings.py                 # Embedding generation entry point
├── retriever.py                  # Retrieval system entry point
├── retriever_gt.py               # Ground truth generation
├── retriever_eval.py             # Retrieval evaluation
├── prompt.py                     # Answer generation
├── requirements.txt              # Dependencies
├── .env                          # Environment variables
├── chunking/                     # Chunking modules
├── metadata/                     # Metadata modules
├── embeddings/                   # Embedding modules
├── retrieval/                    # Retrieval modules
├── evaluation/                   # Evaluation utilities
├── utils/                        # Shared utilities
├── input_files/                  # Raw document inputs
├── chunk_output/                 # Chunked outputs
├── metadata_gen_output/          # Metadata-enriched outputs
├── embeddings_output/            # Generated embeddings
└── retrieval_output/             # Retrieval results
    └── run_[timestamp]/          # Run-specific outputs
        ├── ground_truth/         # Ground truth data
        ├── evaluation/           # Evaluation results
        │   ├── tables/           # Comparison tables
        │   └── visualizations/   # Visualizations
        └── answers/              # Generated answers
```

## Results

Our evaluation shows the impact of different chunking and retrieval strategies on RAG system performance:

| Retriever Configuration | Precision@3 | Recall@5 | NDCG@10 | MRR | Hit Rate@5 |
|-------------------------|------------|----------|---------|-----|------------|
| Content (naive)         | 0.65       | 0.48     | 0.72    | 0.78| 0.67       |
| Content (semantic)      | 0.71       | 0.53     | 0.76    | 0.81| 0.73       |
| TF-IDF (semantic)       | 0.78       | 0.61     | 0.82    | 0.85| 0.79       |
| Prefix (semantic)       | 0.82       | 0.64     | 0.85    | 0.88| 0.82       |
| Reranker-Prefix (semantic) | 0.89    | 0.71     | 0.91    | 0.93| 0.88       |

Key findings:
- Semantic chunking outperforms naive and recursive methods across all retrievers
- Metadata-enhanced retrievers (TF-IDF, Prefix) significantly outperform content-only retrieval
- Reranking provides substantial improvements, especially for precision metrics
- The combination of semantic chunking, prefix-fusion embeddings, and reranking delivers the best overall performance

# Response Quality Evaluation for RAG Systems

This README describes the Response Quality Evaluation module, which measures the quality of responses generated by a RAG (Retrieval-Augmented Generation) system.

## Overview

The Response Quality Evaluation module (`response_quality_eval.py`) assesses three critical aspects of RAG system responses:

1. **Completeness**: How well the response covers entities present in the retrieved documents
2. **Faithfulness**: How accurately the response represents information from the documents
3. **Hallucination**: The degree to which the response contains information not present in the documents
4. **Ground Truth Accuracy** (optional): How well the response matches ground truth answers, if provided

## Key Metrics

### Completeness Score

The completeness score measures how well the entities in the response are covered by the entities in the retrieved documents:

```
C = |R ∩ D| / |R|
```

Where:
- `R` is the set of entities in the response
- `D` is the set of entities in the retrieved documents
- `|R ∩ D|` is the number of entities that appear in both the response and documents
- `|R|` is the total number of entities in the response

A higher completeness score indicates that more of the response's content is supported by the retrieved documents.

### Faithfulness Score

The faithfulness score measures how accurately the response reflects the information in the retrieved documents:

```
F = Number of supported facts / Total facts in response
```

A fact is considered supported if at least 50% of its key entities are present in the retrieved documents.

### Hallucination Score

The hallucination score is the inverse of the faithfulness score:

```
H = 1 - F
```

A higher hallucination score indicates more unsupported information in the response.

### Ground Truth Accuracy (optional)

If ground truth answers are provided, the evaluator also calculates a ground truth accuracy score:

```
GT = |R ∩ G| / |G|
```

Where:
- `R` is the set of entities in the response
- `G` is the set of entities in the ground truth answer
- `|R ∩ G|` is the number of entities that appear in both the response and ground truth
- `|G|` is the total number of entities in the ground truth answer

## Prerequisites

The evaluator requires the following Python packages:

- spaCy
- numpy
- pandas
- matplotlib
- seaborn
- tqdm
- (spaCy language model: en_core_web_md)

Install prerequisites with:

```bash
pip install spacy numpy pandas matplotlib seaborn tqdm
python -m spacy download en_core_web_md
```

## Usage

### Basic Usage

Run the evaluator on a directory containing answer files:

```bash
python response_quality_eval.py --answers_dir retrieval_output/run_1747232219/answers --ground_truth gt.json
```

### Full Options

```bash
python response_quality_eval.py \
  --answers_dir path/to/answers \
  --output_dir path/to/output \
  --retrieval_dir path/to/retrieval \
  --ground_truth path/to/ground_truth.json \
  --threads 4 \
  --use_gpu
```

### Parameters

- `--answers_dir`: Directory containing answer JSON files (required)
- `--output_dir`: Directory to store evaluation results (defaults to `answers_dir/evaluations`)
- `--retrieval_dir`: Directory containing retrieval results (optional)
- `--ground_truth`: Path to ground truth JSON file (optional)
- `--threads`: Number of parallel threads to use (default: 4)
- `--use_gpu`: Use GPU for NLP processing if available


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

[MIT License](LICENSE)

## Glossary

- **Chunking**: Process of breaking documents into smaller segments for better retrieval
- **Content Type**: Classification of text as Procedural, Conceptual, Reference, Warning, or Example
- **Embedding**: Vector representation of text for semantic search
- **Ground Truth**: Reference dataset used for evaluation
- **Hit Rate@k**: Proportion of highly relevant docs found in top-k results
- **Metadata**: Descriptive information about chunks that enhances retrieval
- **NDCG (Normalized Discounted Cumulative Gain)**: Metric measuring ranking quality
- **Prefix-Fusion**: Method that injects metadata as text prefixes before embedding
- **RAG (Retrieval-Augmented Generation)**: LLM technique that retrieves external knowledge
- **Reranker**: Model that improves ranking by reassessing initial retrieval results
- **Semantic Chunking**: Creating chunks based on semantic relationships between sentences
- **TF-IDF (Term Frequency-Inverse Document Frequency)**: Statistical measure for term importance
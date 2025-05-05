#!/usr/bin/env python3
import os
import json
import faiss
import numpy as np
import pickle
import matplotlib.pyplot as plt
import time
from tabulate import tabulate
from collections import defaultdict

def evaluate_embeddings(embedding_dir="embeddings_output", eval_dir="evaluation/embeddings"):
    """Evaluate all embeddings and generate a pairwise comparison."""
    print("Starting embedding evaluation...")
    
    # Create evaluation directory
    os.makedirs(eval_dir, exist_ok=True)
    
    # Find all chunking methods
    chunking_methods = [d for d in os.listdir(embedding_dir) 
                      if os.path.isdir(os.path.join(embedding_dir, d))]
    
    # Store all results
    all_results = {}
    summary_data = []
    
    # Process each chunking method
    for chunking_method in chunking_methods:
        chunking_dir = os.path.join(embedding_dir, chunking_method)
        
        # Find all embedding types
        embedding_types = [d for d in os.listdir(chunking_dir)
                         if os.path.isdir(os.path.join(chunking_dir, d))]
        
        method_results = {}
        
        # Process each embedding type
        for embedding_type in embedding_types:
            embedding_dir_path = os.path.join(chunking_dir, embedding_type)
            
            # Load FAISS index
            index_path = os.path.join(embedding_dir_path, "index.faiss")
            if not os.path.exists(index_path):
                print(f"Warning: No index found at {index_path}")
                continue
                
            try:
                # Load index
                index = faiss.read_index(index_path)
                
                # Load document list
                doc_list_path = os.path.join(embedding_dir_path, "document_list.json")
                if os.path.exists(doc_list_path):
                    with open(doc_list_path, 'r', encoding='utf-8') as f:
                        doc_list = json.load(f)
                else:
                    doc_list = {"total_chunks": index.ntotal}
                
                # Load ID mappings
                mapping_path = os.path.join(embedding_dir_path, "id_mapping.pkl")
                with open(mapping_path, "rb") as f:
                    mappings = pickle.load(f)
                
                # Extract basic metrics
                metrics = {
                    "total_vectors": index.ntotal,
                    "dimension": index.d,
                    "index_size_bytes": os.path.getsize(index_path),
                    "documents": doc_list.get("documents", []),
                    "document_count": len(doc_list.get("documents", [])),
                }
                
                # Calculate advanced metrics if possible
                try:
                    # Random sample query
                    sample_size = min(100, index.ntotal)
                    sample_indices = np.random.choice(index.ntotal, sample_size, replace=False)
                    
                    # For each sample, perform a search
                    k = 10  # Number of neighbors to retrieve
                    avg_distances = []
                    
                    for idx in sample_indices:
                        # Get vector for this index
                        if hasattr(index, "reconstruct"):
                            query_vector = index.reconstruct(int(idx)).reshape(1, -1)
                        else:
                            continue
                            
                        # Search
                        D, I = index.search(query_vector, k+1)  # +1 to skip self
                        
                        # Calculate average distance (skip first which is self)
                        if len(D[0]) > 1:
                            avg_distances.append(np.mean(D[0][1:]))
                    
                    if avg_distances:
                        metrics["avg_neighbor_distance"] = float(np.mean(avg_distances))
                        metrics["min_neighbor_distance"] = float(np.min(avg_distances))
                        metrics["max_neighbor_distance"] = float(np.max(avg_distances))
                except Exception as e:
                    print(f"  Warning: Could not calculate advanced metrics: {str(e)}")
                
                method_results[embedding_type] = metrics
                
                # Add to summary table
                summary_data.append([
                    chunking_method,
                    embedding_type,
                    metrics["total_vectors"],
                    metrics["dimension"],
                    f"{metrics['index_size_bytes'] / (1024*1024):.2f} MB",
                    metrics.get("avg_neighbor_distance", "N/A")
                ])
                
            except Exception as e:
                print(f"Error evaluating {chunking_method}/{embedding_type}: {str(e)}")
        
        all_results[chunking_method] = method_results
    
    # Save results
    output_path = os.path.join(eval_dir, "embedding_evaluation.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    
    # Display results
    print("\n===== EMBEDDING EVALUATION SUMMARY =====\n")
    
    # Print summary table
    headers = ["Chunking Method", "Embedding Type", "Vectors", "Dimension", "Index Size", "Avg NN Distance"]
    print(tabulate(summary_data, headers=headers, tablefmt="grid"))
    
    # Print chunking method comparison
    print("\n===== CHUNKING METHOD COMPARISON =====\n")
    
    # Group by chunking method
    for chunking_method, results in all_results.items():
        print(f"\n{chunking_method.upper()}")
        
        # Create comparison table for this method
        if results:
            method_data = []
            for embedding_type, metrics in results.items():
                method_data.append([
                    embedding_type,
                    metrics["total_vectors"],
                    metrics.get("avg_neighbor_distance", "N/A")
                ])
            
            method_headers = ["Embedding Type", "Vectors", "Avg NN Distance"]
            print(tabulate(method_data, headers=method_headers, tablefmt="simple"))
    
    # Print embedding type comparison
    print("\n===== EMBEDDING TYPE COMPARISON =====\n")
    
    # Group by embedding type
    embedding_type_results = defaultdict(list)
    for chunking_method, results in all_results.items():
        for embedding_type, metrics in results.items():
            embedding_type_results[embedding_type].append({
                "chunking_method": chunking_method,
                "metrics": metrics
            })
    
    for embedding_type, results in embedding_type_results.items():
        print(f"\n{embedding_type.upper()}")
        
        if results:
            type_data = []
            for result in results:
                type_data.append([
                    result["chunking_method"],
                    result["metrics"]["total_vectors"],
                    result["metrics"].get("avg_neighbor_distance", "N/A")
                ])
            
            type_headers = ["Chunking Method", "Vectors", "Avg NN Distance"]
            print(tabulate(type_data, headers=type_headers, tablefmt="simple"))
    
    # Generate visualization
    try:
        plt.figure(figsize=(12, 8))
        
        # Prepare data
        chunking_methods = list(all_results.keys())
        embedding_types = set()
        for results in all_results.values():
            embedding_types.update(results.keys())
        
        embedding_types = list(embedding_types)
        
        # Create grouped bar chart for vector counts
        x = np.arange(len(chunking_methods))
        width = 0.8 / len(embedding_types)
        
        for i, embedding_type in enumerate(embedding_types):
            counts = []
            for method in chunking_methods:
                if embedding_type in all_results[method]:
                    counts.append(all_results[method][embedding_type]["total_vectors"])
                else:
                    counts.append(0)
            
            plt.bar(x + i*width - 0.4 + width/2, counts, width, label=embedding_type)
        
        plt.xlabel('Chunking Method')
        plt.ylabel('Vector Count')
        plt.title('Vector Count by Chunking Method and Embedding Type')
        plt.xticks(x, chunking_methods)
        plt.legend()
        
        # Save the plot
        os.makedirs(os.path.join(eval_dir, "visualizations"), exist_ok=True)
        plt.savefig(os.path.join(eval_dir, "visualizations", "vector_counts.png"))
        
        print(f"\nSaved visualization to {os.path.join(eval_dir, 'visualizations', 'vector_counts.png')}")
    except Exception as e:
        print(f"Error generating visualization: {str(e)}")
    
    print(f"\nEvaluation complete. Results saved to {output_path}")
    return all_results

if __name__ == "__main__":
    evaluate_embeddings()
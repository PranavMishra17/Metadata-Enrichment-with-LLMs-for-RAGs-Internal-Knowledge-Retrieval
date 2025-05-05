import json
import os
import glob
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

from utils.logger import setup_logger

from gpu_utils import GPUVerifier

# Initialize GPU verification
gpu_verifier = GPUVerifier(require_gpu=True)

class MetadataEvaluator:
    """Evaluator for assessing metadata quality."""
    
    def __init__(self, evaluation_dir="evaluation"):
        """Initialize the metadata evaluator."""
        self.evaluation_dir = evaluation_dir
        self.logger = setup_logger("MetadataEvaluator")
        
        # Create evaluation directory if it doesn't exist
        if not os.path.exists(evaluation_dir):
            os.makedirs(evaluation_dir)
            self.logger.info(f"Created evaluation directory: {evaluation_dir}")
    
    def evaluate_metadata(self, metadata_dir):
        """Evaluate metadata quality for all files in a directory."""
        self.logger.info(f"Evaluating metadata in {metadata_dir}")
        
        # Find all enriched JSON files
        metadata_files = glob.glob(os.path.join(metadata_dir, "*_enriched_chunks.json"))
        self.logger.info(f"Found {len(metadata_files)} metadata files")
        
        # Process each file
        results = {}
        for file_path in metadata_files:
            try:
                file_results = self._evaluate_file(file_path)
                file_name = os.path.basename(file_path)
                results[file_name] = file_results
            except Exception as e:
                self.logger.error(f"Error evaluating {file_path}: {str(e)}")
        
        # Aggregate results
        aggregate_results = self._aggregate_results(results)
        
        # Generate visualizations
        self._generate_visualizations(aggregate_results)
        
        # Save results
        output_path = os.path.join(self.evaluation_dir, "metadata_evaluation.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "individual_results": results,
                "aggregate_results": aggregate_results
            }, f, indent=2)
        
        self.logger.info(f"Saved evaluation results to {output_path}")
        
        return aggregate_results
    
    def _evaluate_file(self, file_path):
        """Evaluate metadata quality for a single file."""
        self.logger.info(f"Evaluating file: {file_path}")
        
        # Read metadata file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        chunks = data.get("chunks", [])
        self.logger.info(f"Found {len(chunks)} chunks in file")
        
        # Calculate statistics
        metrics = {
            "total_chunks": len(chunks),
            "completeness_metrics": self._calculate_completeness(chunks),
            "diversity_metrics": self._calculate_diversity(chunks),
            "intent_coverage": self._calculate_intent_coverage(chunks),
            "keyword_statistics": self._calculate_keyword_statistics(chunks)
        }
        
        return metrics
    
    def _calculate_completeness(self, chunks):
        """Calculate completeness metrics for metadata fields."""
        fields = {
            "content_type": 0,
            "keywords": 0,
            "entities": 0,
            "primary_category": 0,
            "secondary_categories": 0,
            "summary": 0,
            "intents": 0,
            "potential_questions": 0,
            "contextual_prefix": 0,
            "tf_idf_keywords": 0
        }
        
        # Count filled fields
        for chunk in chunks:
            if "metadata" in chunk:
                metadata = chunk["metadata"]
                
                # Content fields
                if "content" in metadata:
                    content = metadata["content"]
                    if content.get("content_type", {}).get("primary"):
                        fields["content_type"] += 1
                    if content.get("keywords") and len(content["keywords"]) > 0:
                        fields["keywords"] += 1
                    if content.get("entities") and len(content["entities"]) > 0:
                        fields["entities"] += 1
                
                # Technical fields
                if "technical" in metadata:
                    technical = metadata["technical"]
                    if technical.get("primary_category"):
                        fields["primary_category"] += 1
                    if technical.get("secondary_categories") and len(technical["secondary_categories"]) > 0:
                        fields["secondary_categories"] += 1
                
                # Semantic fields
                if "semantic" in metadata:
                    semantic = metadata["semantic"]
                    if semantic.get("summary"):
                        fields["summary"] += 1
                    if semantic.get("intents") and len(semantic["intents"]) > 0:
                        fields["intents"] += 1
                    if semantic.get("potential_questions") and len(semantic["potential_questions"]) > 0:
                        fields["potential_questions"] += 1
            
            # Embedding enhancement fields
            if "embedding_enhancement" in chunk:
                enhancement = chunk["embedding_enhancement"]
                if enhancement.get("contextual_prefix"):
                    fields["contextual_prefix"] += 1
                if enhancement.get("tf_idf_keywords") and len(enhancement["tf_idf_keywords"]) > 0:
                    fields["tf_idf_keywords"] += 1
        
        # Calculate percentages
        total = len(chunks)
        completeness = {field: (count / total) * 100 if total > 0 else 0 
                       for field, count in fields.items()}
        
        # Calculate overall completeness
        overall = sum(completeness.values()) / len(completeness) if completeness else 0
        
        return {
            "field_completeness": completeness,
            "overall_completeness": overall
        }
    
    def _calculate_diversity(self, chunks):
        """Calculate diversity metrics for metadata values."""
        # Collect values for different fields
        content_types = []
        primary_categories = []
        intents = []
        
        for chunk in chunks:
            if "metadata" in chunk:
                metadata = chunk["metadata"]
                
                # Content type
                if "content" in metadata and "content_type" in metadata["content"]:
                    primary = metadata["content"].get("content_type", {}).get("primary")
                    if primary:
                        content_types.append(primary)
                
                # Primary category
                if "technical" in metadata:
                    primary = metadata["technical"].get("primary_category")
                    if primary:
                        primary_categories.append(primary)
                
                # Intents
                if "semantic" in metadata:
                    chunk_intents = metadata["semantic"].get("intents", [])
                    intents.extend(chunk_intents)
        
        # Calculate diversity (unique values)
        unique_content_types = len(set(content_types))
        unique_primary_categories = len(set(primary_categories))
        unique_intents = len(set(intents))
        
        # Calculate frequency distributions
        content_type_dist = dict(Counter(content_types))
        primary_category_dist = dict(Counter(primary_categories))
        intents_dist = dict(Counter(intents))
        
        return {
            "unique_content_types": unique_content_types,
            "unique_primary_categories": unique_primary_categories,
            "unique_intents": unique_intents,
            "content_type_distribution": content_type_dist,
            "primary_category_distribution": primary_category_dist,
            "intents_distribution": intents_dist
        }
    
    def _calculate_intent_coverage(self, chunks):
        """Calculate intent coverage metrics."""
        # Define standard intents
        standard_intents = {"How-To", "Debug", "Compare", "Reference"}
        
        # Collect all intents
        all_intents = set()
        for chunk in chunks:
            if "metadata" in chunk and "semantic" in chunk["metadata"]:
                intents = chunk["metadata"]["semantic"].get("intents", [])
                all_intents.update(intents)
        
        # Calculate coverage
        covered_intents = standard_intents.intersection(all_intents)
        coverage_percentage = (len(covered_intents) / len(standard_intents)) * 100 if standard_intents else 0
        
        return {
            "standard_intents": list(standard_intents),
            "covered_intents": list(covered_intents),
            "coverage_percentage": coverage_percentage
        }
    
    def _calculate_keyword_statistics(self, chunks):
        """Calculate statistics for keywords."""
        # Collect all keywords
        all_keywords = []
        for chunk in chunks:
            # Content keywords
            if "metadata" in chunk and "content" in chunk["metadata"]:
                keywords = chunk["metadata"]["content"].get("keywords", [])
                all_keywords.extend(keywords)
            
            # TF-IDF keywords
            if "embedding_enhancement" in chunk:
                tf_idf_keywords = chunk["embedding_enhancement"].get("tf_idf_keywords", [])
                all_keywords.extend(tf_idf_keywords)
        
        # Calculate statistics
        total_keywords = len(all_keywords)
        unique_keywords = len(set(all_keywords))
        avg_per_chunk = total_keywords / len(chunks) if chunks else 0
        
        # Get most common keywords
        keyword_counts = Counter(all_keywords)
        most_common = keyword_counts.most_common(10)
        
        return {
            "total_keywords": total_keywords,
            "unique_keywords": unique_keywords,
            "avg_keywords_per_chunk": avg_per_chunk,
            "most_common_keywords": most_common
        }
    
    def _aggregate_results(self, results):
        """Aggregate results from multiple files."""
        if not results:
            return {}
        
        # Initialize aggregate metrics
        aggregate = {
            "total_files": len(results),
            "total_chunks": 0,
            "avg_completeness": 0,
            "avg_intent_coverage": 0,
            "unique_content_types": set(),
            "unique_primary_categories": set(),
            "unique_intents": set(),
            "content_type_distribution": Counter(),
            "primary_category_distribution": Counter(),
            "intents_distribution": Counter(),
            "most_common_keywords": Counter()
        }
        
        # Accumulate metrics
        for file_name, file_results in results.items():
            aggregate["total_chunks"] += file_results.get("total_chunks", 0)
            
            # Completeness
            completeness = file_results.get("completeness_metrics", {}).get("overall_completeness", 0)
            aggregate["avg_completeness"] += completeness
            
            # Intent coverage
            coverage = file_results.get("intent_coverage", {}).get("coverage_percentage", 0)
            aggregate["avg_intent_coverage"] += coverage
            
            # Diversity metrics
            diversity = file_results.get("diversity_metrics", {})
            
            # Update unique sets
            content_types = set(diversity.get("content_type_distribution", {}).keys())
            aggregate["unique_content_types"].update(content_types)
            
            categories = set(diversity.get("primary_category_distribution", {}).keys())
            aggregate["unique_primary_categories"].update(categories)
            
            intents = set(diversity.get("intents_distribution", {}).keys())
            aggregate["unique_intents"].update(intents)
            
            # Update distributions
            for content_type, count in diversity.get("content_type_distribution", {}).items():
                aggregate["content_type_distribution"][content_type] += count
            
            for category, count in diversity.get("primary_category_distribution", {}).items():
                aggregate["primary_category_distribution"][category] += count
            
            for intent, count in diversity.get("intents_distribution", {}).items():
                aggregate["intents_distribution"][intent] += count
            
            # Update keyword counts
            keyword_stats = file_results.get("keyword_statistics", {})
            for keyword, count in keyword_stats.get("most_common_keywords", []):
                aggregate["most_common_keywords"][keyword] += count
        
        # Calculate averages
        num_files = len(results)
        if num_files > 0:
            aggregate["avg_completeness"] /= num_files
            aggregate["avg_intent_coverage"] /= num_files
        
        # Convert sets to lists for JSON serialization
        aggregate["unique_content_types"] = list(aggregate["unique_content_types"])
        aggregate["unique_primary_categories"] = list(aggregate["unique_primary_categories"])
        aggregate["unique_intents"] = list(aggregate["unique_intents"])
        
        # Convert counters to dicts for JSON serialization
        aggregate["content_type_distribution"] = dict(aggregate["content_type_distribution"])
        aggregate["primary_category_distribution"] = dict(aggregate["primary_category_distribution"])
        aggregate["intents_distribution"] = dict(aggregate["intents_distribution"])
        aggregate["most_common_keywords"] = dict(aggregate["most_common_keywords"])
        
        return aggregate
    
    def _generate_visualizations(self, aggregate_results):
        """Generate visualizations for evaluation results."""
        try:
            # Create visualizations directory
            viz_dir = os.path.join(self.evaluation_dir, "visualizations")
            if not os.path.exists(viz_dir):
                os.makedirs(viz_dir)
            
            # Plot field completeness
            self._plot_field_completeness(aggregate_results, viz_dir)
            
            # Plot content type distribution
            self._plot_distribution(
                aggregate_results.get("content_type_distribution", {}),
                "Content Type Distribution",
                "content_type_distribution.png",
                viz_dir
            )
            
            # Plot primary category distribution
            self._plot_distribution(
                aggregate_results.get("primary_category_distribution", {}),
                "Primary Category Distribution",
                "primary_category_distribution.png",
                viz_dir,
                top_n=10
            )
            
            # Plot intent distribution
            self._plot_distribution(
                aggregate_results.get("intents_distribution", {}),
                "Intent Distribution",
                "intent_distribution.png",
                viz_dir
            )
            
            # Plot keyword distribution
            self._plot_distribution(
                aggregate_results.get("most_common_keywords", {}),
                "Most Common Keywords",
                "keyword_distribution.png",
                viz_dir,
                top_n=15
            )
            
            self.logger.info(f"Generated visualizations in {viz_dir}")
            
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {str(e)}")
    
    def _plot_field_completeness(self, aggregate_results, output_dir):
        """Plot field completeness metrics."""
        # Extract completeness data from first file (as example)
        for file_name, file_results in aggregate_results.get("individual_results", {}).items():
            completeness = file_results.get("completeness_metrics", {}).get("field_completeness", {})
            if completeness:
                break
        else:
            # No completeness data found
            return
        
        # Sort fields by completeness
        sorted_fields = sorted(completeness.items(), key=lambda x: x[1], reverse=True)
        fields = [item[0] for item in sorted_fields]
        values = [item[1] for item in sorted_fields]
        
        # Create horizontal bar chart
        plt.figure(figsize=(10, 8))
        bars = plt.barh(fields, values, color='skyblue')
        
        # Add data labels
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%',
                     ha='left', va='center')
        
        plt.xlabel('Completeness (%)')
        plt.title('Metadata Field Completeness')
        plt.xlim(0, 105)  # Add some margin for labels
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(output_dir, "field_completeness.png")
        plt.savefig(output_path)
        plt.close()
    
    def _plot_distribution(self, distribution, title, filename, output_dir, top_n=None):

        """Plot distribution of values."""
        if not distribution:
            return
        
        # Sort by count (descending)
        sorted_items = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
        
        # Take top N if specified
        if top_n and len(sorted_items) > top_n:
            sorted_items = sorted_items[:top_n]
            other_count = sum(count for _, count in sorted_items[top_n:])
            if other_count > 0:
                sorted_items.append(("Other", other_count))
        
        labels = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]
        
        # Create horizontal bar chart
        plt.figure(figsize=(10, max(6, len(labels) * 0.4)))
        bars = plt.barh(labels, values, color='lightgreen')
        
        # Add data labels
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, f'{width}',
                     ha='left', va='center')
        
        plt.xlabel('Count')
        plt.title(title)
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path)
        plt.close()


    def _evaluate_file_directory(self, directory_path):
        """Evaluate all metadata files in a directory."""
        self.logger.info(f"Evaluating files in directory: {directory_path}")
        
        # Find all enriched JSON files
        metadata_files = glob.glob(os.path.join(directory_path, "*_enriched_chunks.json"))
        self.logger.info(f"Found {len(metadata_files)} metadata files")
        
        # Process each file
        file_results = {}
        for file_path in metadata_files:
            try:
                result = self._evaluate_file(file_path)
                file_name = os.path.basename(file_path)
                file_results[file_name] = result
            except Exception as e:
                self.logger.error(f"Error evaluating {file_path}: {str(e)}")
        
        return file_results
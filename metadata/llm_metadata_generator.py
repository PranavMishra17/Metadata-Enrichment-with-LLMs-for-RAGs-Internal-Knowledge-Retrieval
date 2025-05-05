import json
import time
import os
from typing import Dict, List, Any, Optional
import random
from langchain_openai import AzureChatOpenAI

import config
from metadata.base_metadata_generator import BaseMetadataGenerator

class LLMMetadataGenerator(BaseMetadataGenerator):
    """Metadata generator using LLM."""
    
    def __init__(self, output_dir="metadata_gen_output"):
        """Initialize the LLM metadata generator."""
        super().__init__(output_dir)
        
        # Initialize LLM
        try:
            self.client = AzureChatOpenAI(
                azure_deployment=config.AZURE_DEPLOYMENT,
                api_key=config.AZURE_API_KEY,
                api_version=config.AZURE_API_VERSION,
                azure_endpoint=config.AZURE_ENDPOINT,
                temperature=config.TEMPERATURE
            )
            self.logger.info("LLM initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM: {str(e)}")
            self.client = None
    
    def _enrich_chunks(self, chunk_data, method):
        """Enrich chunks with metadata using LLM."""
        # Check if LLM is initialized
        if not self.client:
            self.logger.error("LLM not initialized, cannot enrich chunks")
            return chunk_data
        
        # Extract chunks
        chunks = chunk_data.get("chunks", [])
        self.logger.info(f"Enriching {len(chunks)} chunks")
        
        # Process in batches to handle rate limits
        enriched_chunks = []
        batch_size = config.BATCH_SIZE
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            self.logger.info(f"Processing batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
            
            # Process each chunk in the batch
            for chunk in batch:
                try:
                    enriched_chunk = self._enrich_single_chunk(chunk, method)
                    enriched_chunks.append(enriched_chunk)
                except Exception as e:
                    self.logger.error(f"Error enriching chunk {chunk.get('chunk_id', 'unknown')}: {str(e)}")
                    enriched_chunks.append(chunk)  # Keep original chunk on error
                
                # Random delay to avoid rate limits
                time.sleep(random.uniform(0.5, 1.5))
            
            # Delay between batches
            if i + batch_size < len(chunks):
                self.logger.info(f"Pausing between batches to avoid rate limits")
                time.sleep(random.uniform(1.0, 3.0))
        
        # Update chunk data
        result = chunk_data.copy()
        result["chunks"] = enriched_chunks
        
        # Add metadata about the enrichment process
        result["metadata"]["enrichment"] = {
            "method": "llm",
            "model": config.AZURE_DEPLOYMENT,
            "enriched_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return result
    
    def _enrich_single_chunk(self, chunk, method):
        """Enrich a single chunk with metadata."""
        # Extract text content
        text = chunk.get("text", "")
        
        # Generate metadata using LLM
        content_metadata = self._generate_content_metadata(text)
        technical_metadata = self._generate_technical_metadata(text)
        semantic_metadata = self._generate_semantic_metadata(text)
        
        # Create enhanced embedding data
        embedding_enhancement = self._generate_embedding_enhancement(
            text, content_metadata, technical_metadata, semantic_metadata
        )
        
        # Create enriched chunk
        enriched_chunk = chunk.copy()
        
        # Remove redundant metadata if present
        if "metadata" in enriched_chunk:
            metadata = enriched_chunk["metadata"]
            # Remove metadata that won't help with retrieval
            keys_to_remove = ["page_range", "code_lines", "processing_time"]
            for key in keys_to_remove:
                if key in metadata:
                    del metadata[key]
        else:
            enriched_chunk["metadata"] = {}
        
        # Add new metadata
        enriched_chunk["metadata"]["content"] = content_metadata
        enriched_chunk["metadata"]["technical"] = technical_metadata
        enriched_chunk["metadata"]["semantic"] = semantic_metadata
        enriched_chunk["embedding_enhancement"] = embedding_enhancement
        
        return enriched_chunk
    
    def _generate_content_metadata(self, text):
        """Generate content-based metadata using LLM."""
        try:
            prompt = f"""Analyze this technical documentation chunk and extract metadata.
            
            TEXT:
            {text}
            
            OUTPUT JSON with these fields:
            - content_type: object with "primary" (Conceptual/Procedural/Reference/Warning/Example) and "subtypes" array
            - keywords: array of important technical terms (max 10)
            - entities: array of technical named entities (max 5)
            - has_code: boolean if it contains code snippets
            
            Return ONLY the JSON object, nothing else.
            """
            
            # Call LLM with retries for rate limits
            for attempt in range(config.RETRY_LIMIT):
                try:
                    response = self.client.invoke(prompt)
                    content = response.content
                    
                    # Extract JSON from response
                    metadata = json.loads(content)
                    return metadata
                except Exception as e:
                    if "rate limit" in str(e).lower() and attempt < config.RETRY_LIMIT - 1:
                        self.logger.warning(f"Rate limit hit, retrying in {config.RETRY_DELAY} seconds")
                        time.sleep(config.RETRY_DELAY)
                    else:
                        raise
            
            # Fallback if all retries fail
            return {
                "content_type": {"primary": "Unknown", "subtypes": []},
                "keywords": [],
                "entities": [],
                "has_code": False
            }
        except Exception as e:
            self.logger.error(f"Error generating content metadata: {str(e)}")
            # Return default metadata on error
            return {
                "content_type": {"primary": "Unknown", "subtypes": []},
                "keywords": [],
                "entities": [],
                "has_code": False
            }
    
    def _generate_technical_metadata(self, text):
        """Generate technical metadata using LLM."""
        try:
            prompt = f"""Analyze this technical documentation chunk and extract technical categories.
            
            TEXT:
            {text}
            
            OUTPUT JSON with these fields:
            - primary_category: single most relevant technical category
            - secondary_categories: array of related categories (max 2)
            - mentioned_services: specific services referenced (max 3)
            - mentioned_tools: development tools mentioned (max 3)
            
            Return ONLY the JSON object, nothing else.
            """
            
            # Call LLM with retries
            for attempt in range(config.RETRY_LIMIT):
                try:
                    response = self.client.invoke(prompt)
                    content = response.content
                    
                    # Extract JSON from response
                    metadata = json.loads(content)
                    return metadata
                except Exception as e:
                    if "rate limit" in str(e).lower() and attempt < config.RETRY_LIMIT - 1:
                        self.logger.warning(f"Rate limit hit, retrying in {config.RETRY_DELAY} seconds")
                        time.sleep(config.RETRY_DELAY)
                    else:
                        raise
            
            # Fallback
            return {
                "primary_category": "Unknown",
                "secondary_categories": [],
                "mentioned_services": [],
                "mentioned_tools": []
            }
        except Exception as e:
            self.logger.error(f"Error generating technical metadata: {str(e)}")
            return {
                "primary_category": "Unknown",
                "secondary_categories": [],
                "mentioned_services": [],
                "mentioned_tools": []
            }
    
    def _generate_semantic_metadata(self, text):
        """Generate semantic metadata using LLM."""
        try:
            prompt = f"""Analyze this technical documentation chunk and create semantic metadata.
            
            TEXT:
            {text}
            
            OUTPUT JSON with these fields:
            - summary: concise 1-2 sentence summary
            - intents: array of user intents (How-To, Debug, Compare, Reference)
            - potential_questions: 2-3 specific questions this content answers
            
            Return ONLY the JSON object, nothing else.
            """
            
            # Call LLM with retries
            for attempt in range(config.RETRY_LIMIT):
                try:
                    response = self.client.invoke(prompt)
                    content = response.content
                    
                    # Extract JSON from response
                    metadata = json.loads(content)
                    return metadata
                except Exception as e:
                    if "rate limit" in str(e).lower() and attempt < config.RETRY_LIMIT - 1:
                        self.logger.warning(f"Rate limit hit, retrying in {config.RETRY_DELAY} seconds")
                        time.sleep(config.RETRY_DELAY)
                    else:
                        raise
            
            # Fallback
            return {
                "summary": "Technical documentation content",
                "intents": [],
                "potential_questions": []
            }
        except Exception as e:
            self.logger.error(f"Error generating semantic metadata: {str(e)}")
            return {
                "summary": "Technical documentation content",
                "intents": [],
                "potential_questions": []
            }
    
    def _generate_embedding_enhancement(self, text, content_metadata, technical_metadata, semantic_metadata):
        """Generate embedding enhancement fields."""
        try:
            # Create contextual prefix
            prefixes = []
            if "content_type" in content_metadata and "primary" in content_metadata["content_type"]:
                prefixes.append(f"[{content_metadata['content_type']['primary']}]")
            
            if "primary_category" in technical_metadata:
                prefixes.append(f"[{technical_metadata['primary_category']}]")
            
            # Collect keywords for TF-IDF enhancement
            keywords = []
            if "keywords" in content_metadata:
                keywords.extend(content_metadata["keywords"])
            
            if "entities" in content_metadata:
                keywords.extend(content_metadata["entities"])
            
            if "mentioned_services" in technical_metadata:
                keywords.extend(technical_metadata["mentioned_services"])
            
            # Remove duplicates and limit to 15 keywords
            unique_keywords = list(set(keywords))[:15]
            
            return {
                "contextual_prefix": " ".join(prefixes),
                "tf_idf_keywords": unique_keywords
            }
        except Exception as e:
            self.logger.error(f"Error generating embedding enhancement: {str(e)}")
            return {
                "contextual_prefix": "",
                "tf_idf_keywords": []
            }
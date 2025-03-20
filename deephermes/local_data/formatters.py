"""
Data formatters for RAG and fine-tuning.

This module provides functionality for formatting document chunks
for use in RAG and fine-tuning workflows.
"""

from typing import List, Dict, Any, Optional, Union
import json
import hashlib
import uuid

class RAGFormatter:
    """Format document chunks for RAG (Retrieval-Augmented Generation).
    
    This class prepares document chunks for storage in a vector database
    and retrieval during RAG workflows.
    """
    
    def __init__(self, id_field: str = "id"):
        """Initialize RAG formatter.
        
        Args:
            id_field: Field name to use for document IDs
        """
        self.id_field = id_field
    
    def format_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Format a single document chunk for RAG.
        
        Args:
            chunk: Document chunk with content and metadata
            
        Returns:
            Formatted document chunk for vector storage
        """
        # Ensure chunk has content
        if 'content' not in chunk:
            raise ValueError("Chunk must have a 'content' key")
        
        # Generate a deterministic ID if not present
        if self.id_field not in chunk:
            content_hash = hashlib.md5(chunk['content'].encode()).hexdigest()
            chunk_id = f"chunk_{content_hash}"
            
            # Add source info to ID if available
            if 'metadata' in chunk and 'source_document' in chunk['metadata']:
                source = chunk['metadata']['source_document']
                chunk_id = f"{source}_{chunk_id}"
        else:
            chunk_id = chunk[self.id_field]
        
        # Format for vector storage
        return {
            self.id_field: chunk_id,
            "text": chunk['content'],
            "metadata": chunk.get('metadata', {})
        }
    
    def format_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format multiple document chunks for RAG.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            List of formatted document chunks for vector storage
        """
        return [self.format_chunk(chunk) for chunk in chunks]


class FineTuningFormatter:
    """Format document chunks for fine-tuning.
    
    This class prepares document chunks for use in fine-tuning workflows,
    converting them to instruction-response pairs.
    """
    
    def __init__(self, 
                instruction_template: str = "Please summarize the following text:\n\n{content}",
                response_template: str = "Here is a summary of the text:\n\n{summary}",
                output_format: str = "jsonl"):
        """Initialize fine-tuning formatter.
        
        Args:
            instruction_template: Template for instruction format
            response_template: Template for response format
            output_format: Output format ('jsonl' or 'csv')
        """
        self.instruction_template = instruction_template
        self.response_template = response_template
        self.output_format = output_format
        
        if output_format not in ['jsonl', 'csv']:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def format_for_summarization(self, 
                               chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format document chunks for summarization fine-tuning.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            List of instruction-response pairs for fine-tuning
        """
        formatted_data = []
        
        for chunk in chunks:
            content = chunk.get('content', '')
            
            # Skip empty chunks
            if not content.strip():
                continue
            
            # Create instruction-response pair
            # For summarization, we use the content as both instruction and response
            # In a real scenario, you would need human-generated summaries
            formatted_data.append({
                "instruction": self.instruction_template.format(content=content),
                "response": content  # In real use, this would be a human-written summary
            })
        
        return formatted_data
    
    def format_for_qa(self, 
                    chunks: List[Dict[str, Any]],
                    qa_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format document chunks for question-answering fine-tuning.
        
        Args:
            chunks: List of document chunks
            qa_pairs: List of question-answer pairs
            
        Returns:
            List of instruction-response pairs for fine-tuning
        """
        formatted_data = []
        
        # Create a mapping from chunk content to QA pairs
        chunk_to_qa = {}
        for qa in qa_pairs:
            chunk_id = qa.get('chunk_id')
            if chunk_id:
                if chunk_id not in chunk_to_qa:
                    chunk_to_qa[chunk_id] = []
                chunk_to_qa[chunk_id].append(qa)
        
        # Format each chunk with its associated QA pairs
        for chunk in chunks:
            content = chunk.get('content', '')
            chunk_id = chunk.get('id')
            
            # Skip empty chunks
            if not content.strip() or not chunk_id:
                continue
            
            # Get QA pairs for this chunk
            pairs = chunk_to_qa.get(chunk_id, [])
            
            # Create instruction-response pairs
            for pair in pairs:
                question = pair.get('question', '')
                answer = pair.get('answer', '')
                
                if not question or not answer:
                    continue
                
                formatted_data.append({
                    "instruction": f"Context:\n{content}\n\nQuestion: {question}",
                    "response": answer
                })
        
        return formatted_data
    
    def save_to_file(self, 
                   data: List[Dict[str, Any]], 
                   output_path: str) -> None:
        """Save formatted data to a file.
        
        Args:
            data: List of formatted data
            output_path: Path to output file
        """
        if self.output_format == 'jsonl':
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item) + '\n')
        elif self.output_format == 'csv':
            import csv
            with open(output_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['instruction', 'response'])
                for item in data:
                    writer.writerow([item.get('instruction', ''), 
                                    item.get('response', '')])
    
    def set_templates(self, 
                    instruction_template: str,
                    response_template: str) -> None:
        """Set new templates for instruction and response.
        
        Args:
            instruction_template: New instruction template
            response_template: New response template
        """
        self.instruction_template = instruction_template
        self.response_template = response_template

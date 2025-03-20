"""
RAG workflow integration with the adaptive workflow system.

This module provides integration between the RAG workflow and the
existing adaptive workflow system, including a specialized workflow
template for RAG use cases.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
import os
import logging
from pathlib import Path

from ..adaptive_workflow.workflow_templates import WorkflowTemplate
from ..adaptive_workflow.workflow_runner import WorkflowRunner
from .rag_workflow import RAGWorkflow
from .prompt_templates import get_rag_system_prompt


# RAG workflow template
RAG_WORKFLOW = WorkflowTemplate(
    name="rag",
    description="Retrieval-Augmented Generation with local documents",
    system_prompt=get_rag_system_prompt(),
    reasoning_enabled=True,
    reasoning_depth="deep",
    model_config_adjustments={
        # No specific adjustments needed
    },
    fine_tuning_config_adjustments={
        # RAG doesn't require specific fine-tuning adjustments
    },
    serving_config_adjustments={
        "temperature": 0.3,  # Lower temperature for more factual responses
        "top_p": 0.9,
        "max_tokens_per_request": 4096,  # Allow longer responses for detailed answers
    },
    integration_config_adjustments={
        "rag_enabled": True,
        "max_chunks_to_retrieve": 5,
    }
)


class RAGWorkflowRunner(WorkflowRunner):
    """Extended workflow runner with RAG capabilities.
    
    This class extends the standard workflow runner with RAG-specific
    functionality, including document retrieval and context augmentation.
    """
    
    def __init__(self,
                 workflow_type: str = "rag",
                 config_path: Optional[str] = None,
                 vector_store_path: Optional[str] = None,
                 embedding_model: Optional[str] = None,
                 collection_name: str = "default",
                 prioritize_speed: bool = False,
                 prioritize_quality: bool = False,
                 max_memory_usage_pct: float = 80.0,
                 verbose: bool = True):
        """Initialize the RAG workflow runner.
        
        Args:
            workflow_type: Type of workflow to run (default: "rag")
            config_path: Path to a saved configuration file
            vector_store_path: Path to the vector store directory
            embedding_model: Name of the embedding model to use
            collection_name: Name of the collection to query
            prioritize_speed: Whether to prioritize speed over quality
            prioritize_quality: Whether to prioritize quality over speed
            max_memory_usage_pct: Maximum percentage of memory to use
            verbose: Whether to print verbose output
        """
        # Register RAG workflow template if not already registered
        from ..adaptive_workflow.workflow_templates import WORKFLOW_TEMPLATES
        if "rag" not in WORKFLOW_TEMPLATES:
            WORKFLOW_TEMPLATES["rag"] = RAG_WORKFLOW
        
        # Initialize parent class
        super().__init__(
            workflow_type=workflow_type,
            config_path=config_path,
            prioritize_speed=prioritize_speed,
            prioritize_quality=prioritize_quality,
            max_memory_usage_pct=max_memory_usage_pct,
            verbose=verbose
        )
        
        # Initialize RAG workflow
        self.rag_workflow = RAGWorkflow(
            vector_store_path=vector_store_path,
            embedding_model=embedding_model,
            use_adaptive_model_selection=True,
            max_chunks_to_retrieve=self.config.integration_config.get("max_chunks_to_retrieve", 5)
        )
        
        self.collection_name = collection_name
        self.logger = logging.getLogger("deephermes.rag")
    
    def generate_response(self, prompt: str) -> str:
        """Generate a response to a prompt with RAG augmentation.
        
        Args:
            prompt: The prompt to generate a response for
            
        Returns:
            The generated response
        """
        # Initialize if not already initialized
        if not self.model or not self.tokenizer:
            self.initialize()
        
        # Get system prompt from workflow template
        system_prompt = self.workflow_template.get_full_system_prompt()
        
        # Initialize messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add user message
        messages.append({"role": "user", "content": prompt})
        
        # Augment with RAG
        augmented_messages, retrieved_docs = self.rag_workflow.augment_messages(
            messages=messages,
            collection_name=self.collection_name
        )
        
        if self.verbose and retrieved_docs:
            self.logger.info(f"Retrieved {len(retrieved_docs)} relevant documents")
        
        # Generate response using augmented messages
        from ..core.inference import run_inference
        response = run_inference(
            model=self.model,
            tokenizer=self.tokenizer,
            messages=augmented_messages,
            max_tokens=self.config.serving_config.get("max_tokens_per_request", 4096),
            temperature=self.config.serving_config.get("temperature", 0.3),
            top_p=self.config.serving_config.get("top_p", 0.9)
        )
        
        return response
    
    def chat(self) -> None:
        """Run an interactive chat session with RAG capabilities."""
        if not self.model or not self.tokenizer:
            self.initialize()
        
        # Get system prompt from workflow template
        system_prompt = self.workflow_template.get_full_system_prompt()
        
        # Initialize messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        print(f"\n=== DeepHermes {self.workflow_template.name.capitalize()} ===")
        print(f"Workflow: {self.workflow_template.description}")
        print("Type 'exit' to quit, 'help' for more commands.")
        
        from ..core.inference import run_inference
        
        while True:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            # Check for special commands
            if user_input.lower() == "exit":
                print("Exiting chat...")
                break
            elif user_input.lower() == "help":
                self._print_help()
                continue
            elif user_input.lower() == "clear":
                # Clear chat history but keep system prompt
                if messages and messages[0]["role"] == "system":
                    messages = [messages[0]]
                else:
                    messages = []
                print("Chat history cleared.")
                continue
            elif user_input.lower().startswith("add "):
                # Add a document or directory to the RAG system
                path = user_input[4:].strip()
                if os.path.exists(path):
                    if os.path.isfile(path):
                        chunks = self.rag_workflow.add_document(
                            file_path=path,
                            collection_name=self.collection_name
                        )
                        print(f"Added document with {chunks} chunks to collection '{self.collection_name}'")
                    elif os.path.isdir(path):
                        chunks = self.rag_workflow.add_directory(
                            directory_path=path,
                            collection_name=self.collection_name
                        )
                        print(f"Added directory with {chunks} total chunks to collection '{self.collection_name}'")
                else:
                    print(f"Path not found: {path}")
                continue
            elif user_input.lower().startswith("collection "):
                # Switch collection
                new_collection = user_input[11:].strip()
                self.collection_name = new_collection
                print(f"Switched to collection: {self.collection_name}")
                continue
            elif user_input.lower() == "collections":
                # List collections
                collections = self.rag_workflow.list_collections()
                if collections:
                    print("Available collections:")
                    for collection in collections:
                        print(f"- {collection}")
                else:
                    print("No collections found.")
                continue
            
            # Add user message to history
            messages.append({"role": "user", "content": user_input})
            
            # Augment with RAG
            augmented_messages, retrieved_docs = self.rag_workflow.augment_messages(
                messages=messages,
                collection_name=self.collection_name
            )
            
            if self.verbose and retrieved_docs:
                print(f"[Retrieved {len(retrieved_docs)} relevant documents]")
            
            # Generate response
            print("\nDeepHermes: ", end="", flush=True)
            response = run_inference(
                model=self.model,
                tokenizer=self.tokenizer,
                messages=augmented_messages,
                max_tokens=self.config.serving_config.get("max_tokens_per_request", 4096),
                temperature=self.config.serving_config.get("temperature", 0.3),
                top_p=self.config.serving_config.get("top_p", 0.9)
            )
            
            # Add assistant message to history (use original messages, not augmented)
            messages.append({"role": "assistant", "content": response})
    
    def _print_help(self) -> None:
        """Print help information for chat commands."""
        super()._print_help()  # Call parent class method
        
        # Add RAG-specific commands
        print("\nRAG-specific commands:")
        print("  add <path>           - Add a document or directory to the RAG system")
        print("  collection <name>    - Switch to a different collection")
        print("  collections          - List available collections")

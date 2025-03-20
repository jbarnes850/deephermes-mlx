"""
Command-line interface for the RAG functionality.

This module provides a command-line interface for using the RAG
(Retrieval-Augmented Generation) functionality of DeepHermes MLX.
"""

import os
import sys
import argparse
import logging
from typing import List, Dict, Any, Optional, Union

from ..rag import RAGWorkflow, RAGWorkflowRunner


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration.
    
    Args:
        verbose: Whether to enable verbose logging
    """
    log_level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def add_documents_command(args: argparse.Namespace) -> None:
    """Add documents to the RAG system.
    
    Args:
        args: Command-line arguments
    """
    setup_logging(args.verbose)
    
    # Initialize RAG workflow
    rag = RAGWorkflow(
        vector_store_path=args.vector_store_path,
        embedding_model=args.embedding_model,
        use_adaptive_model_selection=not args.disable_adaptive_model
    )
    
    # Process paths
    total_chunks = 0
    for path in args.paths:
        if os.path.isfile(path):
            print(f"Processing file: {path}")
            chunks = rag.add_document(
                file_path=path,
                collection_name=args.collection,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap
            )
            total_chunks += chunks
            print(f"Added {chunks} chunks from {path}")
        elif os.path.isdir(path):
            print(f"Processing directory: {path}")
            chunks = rag.add_directory(
                directory_path=path,
                collection_name=args.collection,
                recursive=args.recursive,
                file_extensions=args.extensions.split(',') if args.extensions else None,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap
            )
            total_chunks += chunks
            print(f"Added {chunks} chunks from directory {path}")
        else:
            print(f"Path not found: {path}")
    
    print(f"Total chunks added to collection '{args.collection}': {total_chunks}")


def query_command(args: argparse.Namespace) -> None:
    """Query the RAG system.
    
    Args:
        args: Command-line arguments
    """
    setup_logging(args.verbose)
    
    # Initialize RAG workflow runner
    runner = RAGWorkflowRunner(
        vector_store_path=args.vector_store_path,
        embedding_model=args.embedding_model,
        collection_name=args.collection,
        verbose=args.verbose
    )
    
    if args.interactive:
        # Run interactive chat
        runner.chat()
    else:
        # Generate response for a single query
        response = runner.generate_response(args.query)
        print(response)


def list_collections_command(args: argparse.Namespace) -> None:
    """List collections in the vector store.
    
    Args:
        args: Command-line arguments
    """
    setup_logging(args.verbose)
    
    # Initialize RAG workflow
    rag = RAGWorkflow(
        vector_store_path=args.vector_store_path,
        embedding_model=args.embedding_model,
        use_adaptive_model_selection=not args.disable_adaptive_model
    )
    
    # List collections
    collections = rag.list_collections()
    if collections:
        print("Available collections:")
        for collection in collections:
            print(f"- {collection}")
            
            # Show collection info if verbose
            if args.verbose:
                info = rag.get_collection_info(collection)
                print(f"  Documents: {info.get('document_count', 'N/A')}")
                print(f"  Last updated: {info.get('last_updated', 'N/A')}")
    else:
        print("No collections found.")


def delete_collection_command(args: argparse.Namespace) -> None:
    """Delete a collection from the vector store.
    
    Args:
        args: Command-line arguments
    """
    setup_logging(args.verbose)
    
    # Initialize RAG workflow
    rag = RAGWorkflow(
        vector_store_path=args.vector_store_path,
        embedding_model=args.embedding_model,
        use_adaptive_model_selection=not args.disable_adaptive_model
    )
    
    # Confirm deletion
    if not args.force:
        confirm = input(f"Are you sure you want to delete collection '{args.collection}'? (y/N): ")
        if confirm.lower() != 'y':
            print("Deletion cancelled.")
            return
    
    # Delete collection
    success = rag.delete_collection(args.collection)
    if success:
        print(f"Collection '{args.collection}' deleted successfully.")
    else:
        print(f"Failed to delete collection '{args.collection}'.")


def main() -> None:
    """Main entry point for the RAG CLI."""
    parser = argparse.ArgumentParser(
        description="DeepHermes MLX RAG (Retrieval-Augmented Generation) CLI"
    )
    
    # Common arguments
    parser.add_argument(
        "--vector-store-path",
        help="Path to the vector store directory",
        default=os.path.expanduser("~/.deephermes/vector_store")
    )
    parser.add_argument(
        "--embedding-model",
        help="Name of the embedding model to use",
        default="e5-mistral-7b"
    )
    parser.add_argument(
        "--disable-adaptive-model",
        help="Disable adaptive model selection based on available memory",
        action="store_true"
    )
    parser.add_argument(
        "--verbose", "-v",
        help="Enable verbose output",
        action="store_true"
    )
    
    # Subparsers for commands
    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        help="Command to execute"
    )
    
    # Add documents command
    add_parser = subparsers.add_parser(
        "add",
        help="Add documents to the RAG system"
    )
    add_parser.add_argument(
        "paths",
        help="Paths to documents or directories to add",
        nargs="+"
    )
    add_parser.add_argument(
        "--collection", "-c",
        help="Name of the collection to add documents to",
        default="default"
    )
    add_parser.add_argument(
        "--chunk-size",
        help="Size of chunks to split documents into",
        type=int,
        default=1000
    )
    add_parser.add_argument(
        "--chunk-overlap",
        help="Overlap between chunks",
        type=int,
        default=200
    )
    add_parser.add_argument(
        "--recursive", "-r",
        help="Recursively process subdirectories",
        action="store_true"
    )
    add_parser.add_argument(
        "--extensions",
        help="Comma-separated list of file extensions to process (e.g., .txt,.pdf)",
        default=None
    )
    add_parser.set_defaults(func=add_documents_command)
    
    # Query command
    query_parser = subparsers.add_parser(
        "query",
        help="Query the RAG system"
    )
    query_parser.add_argument(
        "query",
        help="Query text",
        nargs="?"
    )
    query_parser.add_argument(
        "--collection", "-c",
        help="Name of the collection to query",
        default="default"
    )
    query_parser.add_argument(
        "--interactive", "-i",
        help="Run in interactive chat mode",
        action="store_true"
    )
    query_parser.set_defaults(func=query_command)
    
    # List collections command
    list_parser = subparsers.add_parser(
        "list",
        help="List collections in the vector store"
    )
    list_parser.set_defaults(func=list_collections_command)
    
    # Delete collection command
    delete_parser = subparsers.add_parser(
        "delete",
        help="Delete a collection from the vector store"
    )
    delete_parser.add_argument(
        "collection",
        help="Name of the collection to delete"
    )
    delete_parser.add_argument(
        "--force", "-f",
        help="Force deletion without confirmation",
        action="store_true"
    )
    delete_parser.set_defaults(func=delete_collection_command)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

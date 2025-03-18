#!/usr/bin/env python
"""
Command-line interface for DeepHermes MLX integrations.

This module provides a CLI for interacting with DeepHermes MLX models
through various integration frameworks like LangChain.
"""

import argparse
import logging
import sys
from typing import List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from deephermes.integrate.langchain import DeepHermesLLM, ChatDeepHermes


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration.
    
    Args:
        verbose: Whether to enable verbose logging
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def generate_text(args: argparse.Namespace) -> None:
    """Generate text using the DeepHermesLLM.
    
    Args:
        args: Command-line arguments
    """
    llm = DeepHermesLLM(
        model_path=args.model_path,
        host=args.host,
        port=args.port,
        start_server=args.start_server,
        adapter_path=args.adapter_path,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    
    if args.prompt_file:
        with open(args.prompt_file, "r") as f:
            prompt = f.read()
    else:
        prompt = args.prompt
    
    print(f"\nGenerating text with prompt: {prompt}\n")
    print("-" * 80)
    
    if args.stream:
        for chunk in llm.stream(prompt):
            print(chunk.text, end="", flush=True)
        print()
    else:
        result = llm.invoke(prompt)
        print(result)
    
    print("-" * 80)


def run_chain(args: argparse.Namespace) -> None:
    """Run a simple LangChain chain with DeepHermesLLM.
    
    Args:
        args: Command-line arguments
    """
    llm = DeepHermesLLM(
        model_path=args.model_path,
        host=args.host,
        port=args.port,
        start_server=args.start_server,
        adapter_path=args.adapter_path,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    
    template = """
    You are an expert in {topic}.
    
    Explain the following concept in simple terms: {concept}
    
    Your explanation should be:
    - Clear and concise
    - Understandable by a beginner
    - Include 3 key points
    """
    
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    
    print(f"\nRunning chain for topic: {args.topic}, concept: {args.concept}\n")
    print("-" * 80)
    
    result = chain.invoke({"topic": args.topic, "concept": args.concept})
    print(result)
    
    print("-" * 80)


def chat(args: argparse.Namespace) -> None:
    """Start an interactive chat session with ChatDeepHermes.
    
    Args:
        args: Command-line arguments
    """
    chat_model = ChatDeepHermes(
        model_path=args.model_path,
        host=args.host,
        port=args.port,
        start_server=args.start_server,
        adapter_path=args.adapter_path,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    
    messages = []
    
    if args.system:
        messages.append(SystemMessage(content=args.system))
        print(f"System: {args.system}")
    
    print("\nChat session started. Type 'exit' or 'quit' to end the session.\n")
    print("-" * 80)
    
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            
            messages.append(HumanMessage(content=user_input))
            
            if args.stream:
                print("Assistant: ", end="", flush=True)
                response_content = ""
                for chunk in chat_model.stream(messages):
                    print(chunk.content, end="", flush=True)
                    response_content += chunk.content
                print()
                messages.append(SystemMessage(content=response_content))
            else:
                response = chat_model.invoke(messages)
                print(f"Assistant: {response.content}")
                messages.append(response)
        
        except KeyboardInterrupt:
            print("\nExiting chat session.")
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("-" * 80)


def main(args: Optional[List[str]] = None) -> int:
    """Main entry point for the CLI.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code
    """
    parser = argparse.ArgumentParser(
        description="DeepHermes MLX Integration CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    
    # Server configuration
    server_group = parser.add_argument_group("Server Configuration")
    server_group.add_argument(
        "--model-path", help="Path to the model directory"
    )
    server_group.add_argument(
        "--host", default="127.0.0.1", help="Server host"
    )
    server_group.add_argument(
        "--port", type=int, default=8080, help="Server port"
    )
    server_group.add_argument(
        "--start-server", action="store_true", help="Start a server if not running"
    )
    server_group.add_argument(
        "--adapter-path", help="Path to adapter weights"
    )
    
    # Generation parameters
    gen_group = parser.add_argument_group("Generation Parameters")
    gen_group.add_argument(
        "--max-tokens", type=int, default=512, help="Maximum tokens to generate"
    )
    gen_group.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature"
    )
    gen_group.add_argument(
        "--top-p", type=float, default=0.9, help="Top-p sampling parameter"
    )
    gen_group.add_argument(
        "--stream", action="store_true", help="Stream the output"
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Generate text command
    generate_parser = subparsers.add_parser(
        "generate", help="Generate text using DeepHermesLLM"
    )
    generate_parser.add_argument(
        "--prompt", default="Tell me about quantum computing.", help="Prompt for generation"
    )
    generate_parser.add_argument(
        "--prompt-file", help="File containing the prompt"
    )
    
    # Run chain command
    chain_parser = subparsers.add_parser(
        "chain", help="Run a simple LangChain chain"
    )
    chain_parser.add_argument(
        "--topic", default="artificial intelligence", help="Topic for the chain"
    )
    chain_parser.add_argument(
        "--concept", default="neural networks", help="Concept to explain"
    )
    
    # Chat command
    chat_parser = subparsers.add_parser(
        "chat", help="Start an interactive chat session"
    )
    chat_parser.add_argument(
        "--system", default="You are a helpful assistant.", help="System message"
    )
    
    # Parse arguments
    args = parser.parse_args(args)
    
    # Set up logging
    setup_logging(args.verbose)
    
    # Run the appropriate command
    try:
        if args.command == "generate":
            generate_text(args)
        elif args.command == "chain":
            run_chain(args)
        elif args.command == "chat":
            chat(args)
        else:
            parser.print_help()
            return 1
    except Exception as e:
        logging.error(f"Error: {e}", exc_info=args.verbose)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

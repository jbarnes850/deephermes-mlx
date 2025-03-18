#!/usr/bin/env python
"""
Example script demonstrating the DeepHermes MLX LangChain integration.

This script shows how to use the DeepHermesLLM and ChatDeepHermes classes
for text generation and chat interactions.
"""

import logging
import argparse
from typing import List, Optional

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage

from deephermes.integrate.langchain import DeepHermesLLM, ChatDeepHermes


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def run_llm_example(
    model_path: Optional[str] = None,
    host: str = "127.0.0.1",
    port: int = 8080,
    start_server: bool = False,
    prompt: str = "Explain quantum computing in simple terms.",
    max_tokens: int = 512,
    temperature: float = 0.7,
    stream: bool = False,
) -> None:
    """Run a simple example using DeepHermesLLM."""
    print("\n=== DeepHermesLLM Example ===\n")
    
    # Create the LLM
    llm = DeepHermesLLM(
        model_path=model_path,
        host=host,
        port=port,
        start_server=start_server,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    
    print(f"Prompt: {prompt}\n")
    
    if stream:
        print("Response (streaming):")
        for chunk in llm.stream(prompt):
            print(chunk.text, end="", flush=True)
        print("\n")
    else:
        print("Response:")
        response = llm.invoke(prompt)
        print(response)
        print()


def run_chain_example(
    model_path: Optional[str] = None,
    host: str = "127.0.0.1",
    port: int = 8080,
    start_server: bool = False,
    topic: str = "artificial intelligence",
    concept: str = "neural networks",
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> None:
    """Run a simple chain example using DeepHermesLLM."""
    print("\n=== LangChain Chain Example ===\n")
    
    # Create the LLM
    llm = DeepHermesLLM(
        model_path=model_path,
        host=host,
        port=port,
        start_server=start_server,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    
    # Create a simple chain
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
    
    print(f"Topic: {topic}")
    print(f"Concept: {concept}\n")
    
    print("Response:")
    result = chain.invoke({"topic": topic, "concept": concept})
    print(result)
    print()


def run_chat_example(
    model_path: Optional[str] = None,
    host: str = "127.0.0.1",
    port: int = 8080,
    start_server: bool = False,
    system_message: str = "You are a helpful assistant.",
    user_message: str = "Explain quantum entanglement in simple terms.",
    max_tokens: int = 512,
    temperature: float = 0.7,
    stream: bool = False,
) -> None:
    """Run a simple chat example using ChatDeepHermes."""
    print("\n=== ChatDeepHermes Example ===\n")
    
    # Create the chat model
    chat = ChatDeepHermes(
        model_path=model_path,
        host=host,
        port=port,
        start_server=start_server,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    
    # Create messages
    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=user_message),
    ]
    
    print(f"System: {system_message}")
    print(f"User: {user_message}\n")
    
    if stream:
        print("Assistant (streaming):")
        for chunk in chat.stream(messages):
            print(chunk.content, end="", flush=True)
        print("\n")
    else:
        print("Assistant:")
        response = chat.invoke(messages)
        print(response.content)
        print()


def main() -> None:
    """Main entry point for the example script."""
    parser = argparse.ArgumentParser(
        description="DeepHermes MLX LangChain Integration Example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Server configuration
    parser.add_argument(
        "--model-path", help="Path to the model directory"
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="Server host"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Server port"
    )
    parser.add_argument(
        "--start-server", action="store_true", help="Start a server if not running"
    )
    
    # Example configuration
    parser.add_argument(
        "--example", choices=["llm", "chain", "chat", "all"], default="all",
        help="Example to run"
    )
    parser.add_argument(
        "--prompt", default="Explain quantum computing in simple terms.",
        help="Prompt for LLM example"
    )
    parser.add_argument(
        "--topic", default="artificial intelligence",
        help="Topic for chain example"
    )
    parser.add_argument(
        "--concept", default="neural networks",
        help="Concept for chain example"
    )
    parser.add_argument(
        "--system-message", default="You are a helpful assistant.",
        help="System message for chat example"
    )
    parser.add_argument(
        "--user-message", default="Explain quantum entanglement in simple terms.",
        help="User message for chat example"
    )
    
    # Generation parameters
    parser.add_argument(
        "--max-tokens", type=int, default=512,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--stream", action="store_true",
        help="Stream the output"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    
    # Run the selected examples
    if args.example in ["llm", "all"]:
        run_llm_example(
            model_path=args.model_path,
            host=args.host,
            port=args.port,
            start_server=args.start_server,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            stream=args.stream,
        )
    
    if args.example in ["chain", "all"]:
        run_chain_example(
            model_path=args.model_path,
            host=args.host,
            port=args.port,
            start_server=args.start_server,
            topic=args.topic,
            concept=args.concept,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    
    if args.example in ["chat", "all"]:
        run_chat_example(
            model_path=args.model_path,
            host=args.host,
            port=args.port,
            start_server=args.start_server,
            system_message=args.system_message,
            user_message=args.user_message,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            stream=args.stream,
        )


if __name__ == "__main__":
    main()

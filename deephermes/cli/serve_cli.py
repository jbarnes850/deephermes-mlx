#!/usr/bin/env python3
"""
Command-line interface for serving DeepHermes MLX models.

This module provides a CLI for starting, stopping, and managing
the MLX-LM server for DeepHermes models.
"""

import os
import sys
import time
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from deephermes.serve.config import ServerConfig
from deephermes.serve.utils import (
    start_server,
    stop_server,
    check_server_status,
    is_port_in_use,
    generate_text
)
from deephermes.export.validator import validate_exported_model_directory


def setup_logging(log_level: str) -> None:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def validate_model(model_path: str) -> bool:
    """
    Validate that the model directory contains all required files.
    
    Args:
        model_path: Path to the model directory
        
    Returns:
        True if the model is valid, False otherwise
    """
    logging.info(f"Validating model in {model_path}...")
    
    if not os.path.exists(model_path):
        logging.error(f"Model path does not exist: {model_path}")
        return False
    
    valid, missing_files = validate_exported_model_directory(model_path)
    if not valid:
        logging.error("Model validation failed. Missing required files:")
        for file in missing_files:
            logging.error(f"  - {file}")
        return False
    
    logging.info("Model validation passed!")
    return True


def save_server_info(config: ServerConfig, pid: int, output_file: str) -> None:
    """
    Save server information to a file.
    
    Args:
        config: Server configuration
        pid: Process ID of the server
        output_file: Path to the output file
    """
    server_info = {
        "pid": pid,
        "config": config.to_dict(),
        "start_time": time.time(),
        "url": f"http://{config.host}:{config.port}"
    }
    
    with open(output_file, "w") as f:
        json.dump(server_info, f, indent=2)
    
    logging.info(f"Server information saved to {output_file}")


def load_server_info(info_file: str) -> Dict[str, Any]:
    """
    Load server information from a file.
    
    Args:
        info_file: Path to the server info file
        
    Returns:
        Dictionary with server information
    """
    try:
        with open(info_file, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Error loading server info: {e}")
        return {}


def start_command(args: argparse.Namespace) -> int:
    """
    Start the MLX-LM server.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code
    """
    # Validate model
    if not validate_model(args.model):
        return 1
    
    # Check if port is already in use
    if is_port_in_use(args.port):
        logging.error(f"Port {args.port} is already in use. Choose a different port or stop the existing server.")
        return 1
    
    # Create server config
    config = ServerConfig(
        model_path=args.model,
        host=args.host,
        port=args.port,
        adapter_path=args.adapter_path,
        cache_limit_gb=args.cache_limit_gb,
        log_level=args.log_level,
        trust_remote_code=args.trust_remote_code,
        use_default_chat_template=args.use_default_chat_template,
        chat_template=args.chat_template
    )
    
    # Start server
    process = start_server(
        config=config,
        background=True,
        wait_for_ready=True,
        timeout=args.timeout
    )
    
    if process is None:
        logging.error("Failed to start server")
        return 1
    
    # Save server info
    if args.info_file:
        save_server_info(config, process.pid, args.info_file)
    
    logging.info(f"Server started successfully at http://{args.host}:{args.port}")
    logging.info(f"Server process ID: {process.pid}")
    
    if not args.detach:
        logging.info("Press Ctrl+C to stop the server")
        try:
            # Wait for the process to finish
            process.wait()
        except KeyboardInterrupt:
            logging.info("Stopping server...")
            stop_server(process)
    
    return 0


def stop_command(args: argparse.Namespace) -> int:
    """
    Stop the MLX-LM server.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code
    """
    if args.info_file and os.path.exists(args.info_file):
        # Load server info
        server_info = load_server_info(args.info_file)
        if server_info and "pid" in server_info:
            pid = server_info["pid"]
            logging.info(f"Stopping server with PID {pid}")
            try:
                os.kill(pid, 0)  # Check if process exists
                os.kill(pid, 15)  # SIGTERM
                logging.info("Server stopped successfully")
                
                # Remove info file
                os.remove(args.info_file)
                return 0
            except OSError:
                logging.warning(f"Process with PID {pid} not found")
    
    # If we get here, either no info file was provided or the process wasn't found
    if args.port:
        logging.info(f"Attempting to stop server on port {args.port}")
        if stop_server(port=args.port):
            logging.info("Server stopped successfully")
            return 0
    
    logging.error("No server found to stop")
    return 1


def status_command(args: argparse.Namespace) -> int:
    """
    Check the status of the MLX-LM server.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code
    """
    host = args.host
    port = args.port
    
    # If info file is provided, use it to get host and port
    if args.info_file and os.path.exists(args.info_file):
        server_info = load_server_info(args.info_file)
        if server_info and "config" in server_info:
            config = server_info["config"]
            host = config.get("host", host)
            port = config.get("port", port)
    
    status = check_server_status(host, port)
    
    if status["running"]:
        print(f"Server is running at {status['url']}")
        if status["models"]:
            print("Available models:")
            for model in status["models"]:
                print(f"  - {model}")
        return 0
    else:
        print(f"Server is not running at {status['url']}")
        if status["error"]:
            print(f"Error: {status['error']}")
        return 1


def test_command(args: argparse.Namespace) -> int:
    """
    Test the MLX-LM server by generating text.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code
    """
    host = args.host
    port = args.port
    
    # If info file is provided, use it to get host and port
    if args.info_file and os.path.exists(args.info_file):
        server_info = load_server_info(args.info_file)
        if server_info and "config" in server_info:
            config = server_info["config"]
            host = config.get("host", host)
            port = config.get("port", port)
    
    # Check if server is running
    status = check_server_status(host, port)
    if not status["running"]:
        print(f"Server is not running at {status['url']}")
        if status["error"]:
            print(f"Error: {status['error']}")
        return 1
    
    # Generate text
    print(f"Generating text with prompt: {args.prompt}")
    try:
        if args.stream:
            print("Streaming response:")
            chunks = generate_text(
                prompt=args.prompt,
                host=host,
                port=port,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                stream=True
            )
            for chunk in chunks:
                print(chunk, end="", flush=True)
            print()
        else:
            response = generate_text(
                prompt=args.prompt,
                host=host,
                port=port,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                stream=False
            )
            print("\nResponse:")
            print(response)
        
        return 0
    except Exception as e:
        print(f"Error generating text: {e}")
        return 1


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="DeepHermes MLX Server CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Start command
    start_parser = subparsers.add_parser("start", help="Start the MLX-LM server")
    start_parser.add_argument("--model", required=True, help="Path to the exported model directory")
    start_parser.add_argument("--host", default="127.0.0.1", help="Host to bind the server to")
    start_parser.add_argument("--port", type=int, default=8080, help="Port to bind the server to")
    start_parser.add_argument("--adapter-path", help="Optional path to adapter weights")
    start_parser.add_argument("--cache-limit-gb", type=int, help="Memory cache limit in GB")
    start_parser.add_argument("--trust-remote-code", action="store_true", help="Trust remote code for tokenizer")
    start_parser.add_argument("--use-default-chat-template", action="store_true", help="Use the default chat template")
    start_parser.add_argument("--chat-template", default="", help="Custom chat template to use")
    start_parser.add_argument("--timeout", type=int, default=30, help="Timeout in seconds for waiting for the server to be ready")
    start_parser.add_argument("--detach", action="store_true", help="Run the server in the background")
    start_parser.add_argument("--info-file", default=".server_info.json", help="File to store server information")
    
    # Stop command
    stop_parser = subparsers.add_parser("stop", help="Stop the MLX-LM server")
    stop_parser.add_argument("--info-file", default=".server_info.json", help="File with server information")
    stop_parser.add_argument("--port", type=int, help="Port the server is running on")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Check the status of the MLX-LM server")
    status_parser.add_argument("--host", default="127.0.0.1", help="Server host")
    status_parser.add_argument("--port", type=int, default=8080, help="Server port")
    status_parser.add_argument("--info-file", default=".server_info.json", help="File with server information")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test the MLX-LM server by generating text")
    test_parser.add_argument("--prompt", required=True, help="Text prompt")
    test_parser.add_argument("--host", default="127.0.0.1", help="Server host")
    test_parser.add_argument("--port", type=int, default=8080, help="Server port")
    test_parser.add_argument("--max-tokens", type=int, default=512, help="Maximum number of tokens to generate")
    test_parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    test_parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling parameter")
    test_parser.add_argument("--stream", action="store_true", help="Stream the response")
    test_parser.add_argument("--info-file", default=".server_info.json", help="File with server information")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    return args


def main() -> int:
    """
    Main entry point for the CLI.
    
    Returns:
        Exit code
    """
    args = parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    
    # Run the appropriate command
    if args.command == "start":
        return start_command(args)
    elif args.command == "stop":
        return stop_command(args)
    elif args.command == "status":
        return status_command(args)
    elif args.command == "test":
        return test_command(args)
    else:
        logging.error(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

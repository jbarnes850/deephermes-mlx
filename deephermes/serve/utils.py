"""
Utility functions for managing the DeepHermes MLX server.
"""

import os
import signal
import subprocess
import time
import logging
import requests
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Union

from .config import ServerConfig


def start_server(
    config: ServerConfig,
    background: bool = True,
    wait_for_ready: bool = True,
    timeout: int = 30
) -> Optional[subprocess.Popen]:
    """
    Start the MLX-LM server with the given configuration.
    
    Args:
        config: Server configuration
        background: Whether to run the server in the background
        wait_for_ready: Whether to wait for the server to be ready
        timeout: Timeout in seconds for waiting for the server to be ready
        
    Returns:
        Subprocess object if running in the background, None otherwise
    """
    logging.info(f"Starting MLX-LM server with model: {config.model_path}")
    
    # Check if server is already running on the specified port
    if is_port_in_use(config.port):
        logging.warning(f"Port {config.port} is already in use. Server may already be running.")
        return None
    
    # Build command
    cmd = ["python", "-m", "mlx_lm.server"] + config.to_args()
    
    # Start server
    if background:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if wait_for_ready:
            start_time = time.time()
            while time.time() - start_time < timeout:
                if is_server_ready(config.host, config.port):
                    logging.info(f"Server is ready at http://{config.host}:{config.port}")
                    return process
                time.sleep(1)
            
            logging.warning(f"Server did not become ready within {timeout} seconds")
        
        return process
    else:
        # Run in foreground
        subprocess.run(cmd)
        return None


def stop_server(process: Optional[subprocess.Popen] = None, port: Optional[int] = None) -> bool:
    """
    Stop the MLX-LM server.
    
    Args:
        process: Subprocess object returned by start_server
        port: Port the server is running on (used if process is None)
        
    Returns:
        True if the server was stopped, False otherwise
    """
    if process is not None:
        logging.info("Stopping MLX-LM server")
        process.terminate()
        try:
            process.wait(timeout=5)
            return True
        except subprocess.TimeoutExpired:
            logging.warning("Server did not terminate gracefully, killing")
            process.kill()
            return True
    elif port is not None:
        # Try to connect to the port and send a termination signal
        # This is a simplified approach that won't work in all cases
        # but avoids permission issues with psutil
        logging.info(f"Attempting to stop server on port {port}")
        try:
            # Try to make a request to the server to check if it's running
            requests.get(f"http://localhost:{port}/v1/models", timeout=1)
            logging.info(f"Server found on port {port}, but cannot be stopped programmatically without process ID")
            logging.info("Please manually stop the server process")
            return False
        except requests.RequestException:
            logging.info(f"No server found running on port {port}")
            return False
    
    logging.warning("No server process found to stop")
    return False


def is_port_in_use(port: int) -> bool:
    """
    Check if a port is in use.
    
    Args:
        port: Port to check
        
    Returns:
        True if the port is in use, False otherwise
    """
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def is_server_ready(host: str, port: int) -> bool:
    """
    Check if the server is ready to accept requests.
    
    Args:
        host: Server host
        port: Server port
        
    Returns:
        True if the server is ready, False otherwise
    """
    try:
        response = requests.get(f"http://{host}:{port}/v1/models", timeout=1)
        return response.status_code == 200
    except requests.RequestException:
        return False


def check_server_status(host: str, port: int) -> Dict[str, Any]:
    """
    Check the status of the MLX-LM server.
    
    Args:
        host: Server host
        port: Server port
        
    Returns:
        Dictionary with server status information
    """
    status = {
        "running": False,
        "url": f"http://{host}:{port}",
        "models": [],
        "error": None
    }
    
    try:
        response = requests.get(f"http://{host}:{port}/v1/models", timeout=2)
        if response.status_code == 200:
            status["running"] = True
            models_data = response.json()
            status["models"] = [model["id"] for model in models_data.get("data", [])]
    except requests.RequestException as e:
        status["error"] = str(e)
    
    return status


def generate_text(
    prompt: str,
    host: str = "127.0.0.1",
    port: int = 8080,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    stream: bool = False,
    **kwargs
) -> Union[str, List[str]]:
    """
    Generate text using the MLX-LM server.
    
    Args:
        prompt: Text prompt
        host: Server host
        port: Server port
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        stream: Whether to stream the response
        **kwargs: Additional parameters to pass to the API
        
    Returns:
        Generated text or list of text chunks if streaming
    """
    url = f"http://{host}:{port}/v1/completions"
    
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": stream,
        **kwargs
    }
    
    if stream:
        chunks = []
        response = requests.post(url, json=payload, stream=True)
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = line[6:]  # Remove 'data: ' prefix
                    if data != '[DONE]':
                        try:
                            chunk = json.loads(data)
                            text = chunk.get('choices', [{}])[0].get('text', '')
                            chunks.append(text)
                        except json.JSONDecodeError:
                            pass
        return chunks
    else:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        return result.get('choices', [{}])[0].get('text', '')

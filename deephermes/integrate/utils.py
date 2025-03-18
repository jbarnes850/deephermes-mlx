"""
Utility functions for DeepHermes MLX integrations.

This module provides shared utilities for integrating DeepHermes MLX models
with various frameworks like LangChain and LlamaIndex.
"""

from typing import Dict, List, Optional, Union
import logging
import requests
import time

logger = logging.getLogger(__name__)


def format_chat_messages(messages: List[Dict[str, str]]) -> str:
    """Format chat messages into a prompt string.
    
    This is a utility function for formatting chat messages in a consistent way
    across different integration frameworks.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        
    Returns:
        Formatted prompt string
    """
    prompt_parts = []
    
    for message in messages:
        role = message.get("role", "").capitalize()
        content = message.get("content", "")
        
        if role and content:
            prompt_parts.append(f"{role}: {content}")
        elif content:
            prompt_parts.append(content)
    
    # Add the assistant prefix for the response
    prompt_parts.append("Assistant:")
    
    return "\n\n".join(prompt_parts)


def wait_for_server(host: str, port: int, timeout: int = 30, retry_interval: float = 0.5) -> bool:
    """Wait for the server to be ready.
    
    Args:
        host: Server host
        port: Server port
        timeout: Maximum time to wait in seconds
        retry_interval: Time between retries in seconds
        
    Returns:
        True if server is ready, False otherwise
    """
    url = f"http://{host}:{port}/v1/models"
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                logger.info(f"Server at {host}:{port} is ready")
                return True
        except requests.exceptions.RequestException:
            pass
        
        time.sleep(retry_interval)
    
    logger.warning(f"Timed out waiting for server at {host}:{port}")
    return False


def get_model_info(host: str, port: int) -> Optional[Dict]:
    """Get information about the model loaded on the server.
    
    Args:
        host: Server host
        port: Server port
        
    Returns:
        Dictionary with model information or None if request fails
    """
    url = f"http://{host}:{port}/v1/models"
    
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error getting model info: {e}")
        return None

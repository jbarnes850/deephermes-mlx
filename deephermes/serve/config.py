"""
Configuration utilities for the DeepHermes MLX server.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class ServerConfig:
    """
    Configuration for the DeepHermes MLX server.
    
    This class wraps the configuration options for the MLX-LM server,
    providing a convenient interface for setting up and managing server instances.
    
    Attributes:
        model_path: Path to the exported model directory
        host: Host to bind the server to
        port: Port to bind the server to
        adapter_path: Optional path to adapter weights
        cache_limit_gb: Optional memory cache limit in GB
        log_level: Logging level for the server
        trust_remote_code: Whether to trust remote code for tokenizer
        use_default_chat_template: Whether to use the default chat template
        chat_template: Custom chat template to use
    """
    model_path: str
    host: str = "127.0.0.1"
    port: int = 8080
    adapter_path: Optional[str] = None
    cache_limit_gb: Optional[int] = None
    log_level: str = "INFO"
    trust_remote_code: bool = False
    use_default_chat_template: bool = True
    chat_template: str = ""
    
    def to_args(self) -> List[str]:
        """
        Convert the configuration to command-line arguments for the MLX-LM server.
        
        Returns:
            List of command-line arguments
        """
        args = [
            "--model", self.model_path,
            "--host", self.host,
            "--port", str(self.port),
            "--log-level", self.log_level,
        ]
        
        if self.adapter_path:
            args.extend(["--adapter-path", self.adapter_path])
        
        if self.cache_limit_gb is not None:
            args.extend(["--cache-limit-gb", str(self.cache_limit_gb)])
        
        if self.trust_remote_code:
            args.append("--trust-remote-code")
        
        if self.use_default_chat_template:
            args.append("--use-default-chat-template")
        
        if self.chat_template:
            args.extend(["--chat-template", self.chat_template])
        
        return args
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration to a dictionary.
        
        Returns:
            Dictionary representation of the configuration
        """
        return {
            "model_path": self.model_path,
            "host": self.host,
            "port": self.port,
            "adapter_path": self.adapter_path,
            "cache_limit_gb": self.cache_limit_gb,
            "log_level": self.log_level,
            "trust_remote_code": self.trust_remote_code,
            "use_default_chat_template": self.use_default_chat_template,
            "chat_template": self.chat_template,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ServerConfig":
        """
        Create a ServerConfig from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration values
            
        Returns:
            ServerConfig instance
        """
        return cls(**config_dict)

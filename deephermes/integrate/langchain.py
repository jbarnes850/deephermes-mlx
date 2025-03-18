"""
LangChain integration for DeepHermes MLX models.

This module provides classes for integrating DeepHermes MLX models with LangChain,
enabling seamless use of these models with LangChain's ecosystem of tools and frameworks.
"""

from typing import Any, Dict, Iterator, List, Mapping, Optional, Union
import json
import logging
import requests
import subprocess
from pydantic import Field, root_validator

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult, GenerationChunk, ChatGenerationChunk

from deephermes.serve.config import ServerConfig
from deephermes.serve.utils import start_server, stop_server, is_server_ready, check_server_status

logger = logging.getLogger(__name__)


class DeepHermesLLM(LLM):
    """LangChain integration for DeepHermes MLX models.
    
    This class allows DeepHermes MLX models to be used with LangChain,
    either by connecting to an existing server or starting a new one.
    
    Example:
        .. code-block:: python
        
            from deephermes.integrate.langchain import DeepHermesLLM
            
            # Connect to an existing server
            llm = DeepHermesLLM(host="127.0.0.1", port=8080)
            
            # Or start a new server with a model
            llm = DeepHermesLLM(model_path="/path/to/model", start_server=True)
            
            # Generate text
            result = llm("What is quantum computing?")
    """
    
    # Server configuration
    model_path: Optional[str] = None
    """Path to the model directory."""
    
    host: str = "127.0.0.1"
    """Host where the server is running."""
    
    port: int = 8080
    """Port where the server is running."""
    
    start_server: bool = False
    """Whether to start a server if one is not already running."""
    
    adapter_path: Optional[str] = None
    """Optional path to adapter weights."""
    
    cache_limit_gb: Optional[int] = None
    """Optional memory cache limit in GB."""
    
    trust_remote_code: bool = False
    """Whether to trust remote code for tokenizer."""
    
    use_default_chat_template: bool = True
    """Whether to use the default chat template."""
    
    chat_template: str = ""
    """Custom chat template to use."""
    
    # Generation parameters
    max_tokens: int = 512
    """Maximum number of tokens to generate."""
    
    temperature: float = 0.7
    """Sampling temperature."""
    
    top_p: float = 0.9
    """Top-p sampling parameter."""
    
    # Internal state
    _server_process: Optional[subprocess.Popen] = None
    
    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True
    
    @root_validator(skip_on_failure=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the server is running or can be started."""
        host = values.get("host")
        port = values.get("port")
        start_server_flag = values.get("start_server")
        model_path = values.get("model_path")
        
        # Check if server is already running
        if not is_server_ready(host, port):
            if start_server_flag:
                if not model_path:
                    raise ValueError(
                        "model_path must be provided when start_server=True and no server is running"
                    )
                
                # Configure and start server
                config = ServerConfig(
                    model_path=model_path,
                    host=host,
                    port=port,
                    adapter_path=values.get("adapter_path"),
                    cache_limit_gb=values.get("cache_limit_gb"),
                    trust_remote_code=values.get("trust_remote_code"),
                    use_default_chat_template=values.get("use_default_chat_template"),
                    chat_template=values.get("chat_template"),
                )
                
                logger.info(f"Starting MLX-LM server with model: {model_path}")
                values["_server_process"] = start_server(
                    config=config,
                    background=True,
                    wait_for_ready=True
                )
                
                if not values["_server_process"]:
                    raise RuntimeError(
                        f"Failed to start server at {host}:{port}. Check logs for details."
                    )
                
                logger.info(f"Server started at http://{host}:{port}")
            else:
                raise ValueError(
                    f"No server found at {host}:{port}. Set start_server=True to start one."
                )
        else:
            logger.info(f"Using existing server at http://{host}:{port}")
            
            # Check server status
            try:
                status = check_server_status(host, port)
                logger.info(f"Server status: {status}")
            except Exception as e:
                logger.warning(f"Failed to check server status: {e}")
        
        return values
    
    def __del__(self):
        """Clean up resources when the object is deleted."""
        if self._server_process:
            logger.info("Stopping MLX-LM server")
            stop_server(self._server_process)
    
    @property
    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return "deephermes_mlx"
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_path": self.model_path,
            "host": self.host,
            "port": self.port,
            "adapter_path": self.adapter_path,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Generate text using the model."""
        url = f"http://{self.host}:{self.port}/v1/completions"
        
        # Prepare the payload
        payload = {
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "stream": False,
        }
        
        if stop:
            payload["stop"] = stop
        
        # Make the request
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            
            return result.get('choices', [{}])[0].get('text', '')
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling MLX-LM server: {e}")
            raise RuntimeError(f"Error calling MLX-LM server: {e}")
    
    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Stream generated text from the model."""
        url = f"http://{self.host}:{self.port}/v1/completions"
        
        # Prepare the payload
        payload = {
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "stream": True,
        }
        
        if stop:
            payload["stop"] = stop
        
        # Make the streaming request
        try:
            response = requests.post(url, json=payload, stream=True)
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]  # Remove 'data: ' prefix
                        if data != '[DONE]':
                            try:
                                chunk_data = json.loads(data)
                                text = chunk_data.get('choices', [{}])[0].get('text', '')
                                chunk = GenerationChunk(text=text)
                                if run_manager:
                                    run_manager.on_llm_new_token(chunk.text)
                                yield chunk
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to decode JSON: {data}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error streaming from MLX-LM server: {e}")
            raise RuntimeError(f"Error streaming from MLX-LM server: {e}")


class ChatDeepHermes(BaseChatModel):
    """Chat model implementation for DeepHermes MLX.
    
    This class allows using DeepHermes MLX models with LangChain's chat interfaces.
    
    Example:
        .. code-block:: python
        
            from deephermes.integrate.langchain import ChatDeepHermes
            from langchain_core.messages import HumanMessage, SystemMessage
            
            chat = ChatDeepHermes(model_path="/path/to/model", start_server=True)
            
            messages = [
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content="What is quantum computing?")
            ]
            
            response = chat.invoke(messages)
            print(response.content)
    """
    
    llm: DeepHermesLLM
    """The underlying LLM to use."""
    
    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        host: str = "127.0.0.1",
        port: int = 8080,
        start_server: bool = False,
        adapter_path: Optional[str] = None,
        cache_limit_gb: Optional[int] = None,
        trust_remote_code: bool = False,
        use_default_chat_template: bool = True,
        chat_template: str = "",
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs: Any,
    ):
        """Initialize the chat model."""
        llm = DeepHermesLLM(
            model_path=model_path,
            host=host,
            port=port,
            start_server=start_server,
            adapter_path=adapter_path,
            cache_limit_gb=cache_limit_gb,
            trust_remote_code=trust_remote_code,
            use_default_chat_template=use_default_chat_template,
            chat_template=chat_template,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        
        super().__init__(llm=llm, **kwargs)
    
    @property
    def _llm_type(self) -> str:
        """Return the type of chat model."""
        return "deephermes_chat"
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a chat response."""
        prompt = self._to_chat_prompt(messages)
        
        response = self.llm._call(
            prompt=prompt,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )
        
        message = AIMessage(content=response)
        
        return ChatResult(generations=[ChatGeneration(message=message)])
    
    def _to_chat_prompt(self, messages: List[BaseMessage]) -> str:
        """Convert chat messages to a prompt string.
        
        This method formats the messages in a way that the model can understand.
        If the model has a specific chat template, the MLX-LM server will use it.
        This is a fallback for models without a chat template.
        """
        prompt_parts = []
        
        for message in messages:
            if isinstance(message, SystemMessage):
                prompt_parts.append(f"System: {message.content}")
            elif isinstance(message, HumanMessage):
                prompt_parts.append(f"Human: {message.content}")
            elif isinstance(message, AIMessage):
                prompt_parts.append(f"Assistant: {message.content}")
            elif isinstance(message, ChatMessage):
                role = message.role.capitalize()
                prompt_parts.append(f"{role}: {message.content}")
            else:
                prompt_parts.append(message.content)
        
        # Add the assistant prefix for the response
        prompt_parts.append("Assistant:")
        
        return "\n\n".join(prompt_parts)
    
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream chat generations."""
        prompt = self._to_chat_prompt(messages)
        
        for chunk in self.llm._stream(
            prompt=prompt,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        ):
            yield ChatGenerationChunk(message=AIMessage(content=chunk.text))

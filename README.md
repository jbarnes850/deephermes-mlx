# DeepHermes MLX Inference

This repository contains a Python implementation for running inference with DeepHermes models on Apple Silicon using the MLX framework. It supports memory-efficient loading options and enhanced reasoning capabilities.

### Features

- Run inference with DeepHermes-3-Llama-3-8B model on Apple Silicon
- Interactive chat mode with history management
- Memory-efficient options (quantization, lazy loading)
- Enhanced reasoning capabilities with DeepHermes's specialized thinking process
- Streaming text generation

### Installation

1. Clone this repository:
```bash
git clone https://github.com/jbarnes850/mlx-deephermes.git
cd mlx-deephermes
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

#### Interactive Chat

The easiest way to interact with the model is through the chat interface:

```bash
python chat.py
```

With memory optimization options:

```bash
python chat.py --quantize 4bit --lazy-load
```

With enhanced reasoning capabilities:

```bash
python chat.py --reasoning
```

#### Single Prompt Inference

For single prompt inference, use the main script:

```bash
python main.py --prompt "Explain quantum computing in simple terms."
```

With memory optimization and reasoning:

```bash
python main.py --prompt "Explain quantum computing in simple terms." --quantize 4bit --reasoning
```

### Command Line Options

#### Model Options
- `--model`: Model path or Hugging Face repo ID (default: "mlx-community/DeepHermes-3-Llama-3-8B-Preview-bf16")
- `--trust-remote-code`: Trust remote code in tokenizer

#### Generation Options
- `--prompt`: Text prompt for generation
- `--system-prompt`: System prompt to use (default: "You are DeepHermes, a helpful AI assistant.")
- `--max-tokens`: Maximum number of tokens to generate (default: 1024)
- `--temperature`: Sampling temperature (default: 0.7)
- `--top-p`: Top-p sampling parameter (default: 0.9)
- `--no-stream`: Disable streaming output
- `--max-kv-size`: Maximum KV cache size for long context

#### Reasoning Options
- `--reasoning`: Enable DeepHermes reasoning mode

#### Memory Optimization Options
- `--quantize`: Quantize model to reduce memory usage (choices: "4bit", "8bit")
- `--lazy-load`: Load model weights lazily to reduce memory usage

### Chat Commands

During an interactive chat session, you can use the following commands:

- `exit`: Quit the chat session
- `clear`: Clear chat history
- `system <prompt>`: Change the system prompt
- `reasoning <on|off>`: Toggle reasoning mode on or off

### DeepHermes Reasoning Capabilities

DeepHermes models are designed with enhanced reasoning capabilities. When the reasoning mode is enabled, the model uses the following specialized prompt:

```bash
You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem.
```

This prompting technique enables the model to:
1. Develop long chains of thought
2. Deliberate with itself through systematic reasoning
3. Enclose its internal thinking process in `<think>` tags
4. Provide a well-reasoned solution after thorough consideration

### Performance Considerations

The DeepHermes-3-Llama-3-8B model is optimized for Apple Silicon and should run efficiently on most modern Macs. For best performance:

1. Use 4-bit quantization (`--quantize 4bit`) to significantly reduce memory usage
2. Enable lazy loading (`--lazy-load`) to load weights on demand
3. For older or memory-constrained devices, consider limiting the maximum tokens generated

For very large models or long contexts, you may need to increase the system wired memory limit:

```bash
sudo sysctl iogpu.wired_limit_mb=32000
```

### License

This project is provided as-is under the terms of the license of the underlying models and libraries.

### Acknowledgments

- [MLX Team at Apple](https://github.com/ml-explore/mlx) for the MLX framework
- [MLX-LM](https://github.com/ml-explore/mlx-lm) for the LLM infrastructure
- [DeepHermes Team](https://huggingface.co/mlx-community/DeepHermes-3-Llama-3-8B-Preview-bf16) for the model weights

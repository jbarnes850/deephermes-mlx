# DeepHermes-3-Mistral-24B MLX Inference

This repository contains code to run inference on the DeepHermes-3-Mistral-24B model locally using Apple's MLX framework on Apple Silicon devices.

## Overview

This project provides a simple and efficient way to run the DeepHermes-3-Mistral-24B model locally on Apple Silicon devices using the MLX framework. The implementation is modular and follows best practices for Python code organization.

## Requirements

- macOS running on Apple Silicon (M1/M2/M3)
- Python 3.8+
- MLX and MLX-LM libraries

## Installation

1. Clone this repository:
```bash
git clone https://github.com/jbarnes850/mlx-deephermes.git
cd mlx-deephermes
```

2. Create a virtual environment and install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Basic Inference

To run a simple inference with the model:

```bash
python main.py --prompt "Tell me about quantum computing" --reasoning
```

### Interactive Chat

For an interactive chat experience:

```bash
python chat.py --reasoning
```

### Command Line Options

Both scripts support the following command line options:

- `--model`: Model path on Hugging Face Hub (default: "Jarrodbarnes/DeepHermes-3-Mistral-24B-Preview-mlx-fp16")
- `--prompt`: Prompt to use for generation (for main.py)
- `--system-prompt`: System prompt to use (default: "You are DeepHermes, a helpful AI assistant.")
- `--max-tokens`: Maximum number of tokens to generate (default: 1024)
- `--temperature`: Sampling temperature (default: 0.7)
- `--top-p`: Top-p sampling parameter (default: 0.9)
- `--trust-remote-code`: Trust remote code in tokenizer
- `--max-kv-size`: Maximum KV cache size for long context
- `--reasoning`: Enable reasoning mode (adds reasoning instruction to system prompt)
- `--no-stream`: Disable streaming output (for main.py)

### Chat Commands

In the interactive chat mode, you can use the following commands:

- `exit`: Quit the chat
- `clear`: Clear chat history
- `system <prompt>`: Change the system prompt

## Project Structure

- `model.py`: Model loading and configuration
- `inference.py`: Inference functionality
- `main.py`: Script for running single inferences
- `chat.py`: Interactive chat interface
- `requirements.txt`: Required Python packages

## Original Model Information

- Original model: [NousResearch/DeepHermes-3-Mistral-24B-Preview](https://huggingface.co/NousResearch/DeepHermes-3-Mistral-24B-Preview)
- MLX version: [Jarrodbarnes/DeepHermes-3-Mistral-24B-Preview-mlx-fp16](https://huggingface.co/Jarrodbarnes/DeepHermes-3-Mistral-24B-Preview-mlx-fp16)

## Performance Considerations

For large models like DeepHermes-3-Mistral-24B, you may need to increase the system wired memory limit to improve performance:

```bash
sudo sysctl iogpu.wired_limit_mb=N
```

Where N should be larger than the size of the model in megabytes but smaller than the memory size of your machine.

## License

This project is released under the MIT License.

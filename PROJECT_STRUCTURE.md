# DeepHermes Project Structure

This document outlines the organization of the DeepHermes MLX project, which has been structured to balance modularity and ease of use.

## Directory Structure

```bash
mlx-deephermes/
├── deephermes/                  # Main package directory
│   ├── __init__.py              # Package initialization and exports
│   ├── cli.py                   # Command-line interface
│   ├── chat.py                  # Interactive chat interface
│   ├── core/                    # Core functionality
│   │   ├── __init__.py
│   │   ├── model.py             # Model loading functions
│   │   ├── inference.py         # Inference functions
│   │   └── utils.py             # Utility functions
│   └── model_selector/          # Model selection based on hardware
│       ├── __init__.py
│       ├── cli.py               # Model selector CLI
│       ├── hardware_detection.py # Hardware detection
│       ├── integration.py       # Integration with main package
│       └── model_recommender.py # Model recommendation logic
├── benchmarks/                  # Benchmarking tools
│   ├── __init__.py
│   ├── benchmark_runner.py      # Core benchmarking logic
│   ├── run_benchmark.py         # Benchmark runner script
│   ├── datasets/                # Dataset handling
│   ├── metrics/                 # Evaluation metrics
│   └── visualizations/          # Result visualization
├── examples/                    # Example scripts and notebooks
│   └── examples.md              # Example usage patterns
├── scripts/                     # Utility scripts
│   └── quickstart.sh            # Quick start script
├── tests/                       # Unit tests
│   └── __init__.py
├── main.py                      # Entry point for inference
├── chat.py                      # Entry point for chat
├── setup.py                     # Package installation
├── requirements.txt             # Dependencies
└── README.md                    # Project documentation
```

## Key Components

### Entry Points

We provide two simple entry points in the root directory for easy access:

- **main.py**: Run inference with a prompt
- **chat.py**: Start an interactive chat session

### Core Package (`deephermes/core/`)

The core functionality is organized into three main modules:

- **model.py**: Functions for loading models with various configurations
- **inference.py**: Inference logic for running models on prompts
- **utils.py**: Utility functions for reasoning prompts and other helpers

### User Interfaces

- **cli.py**: Command-line interface for running inference
- **chat.py**: Interactive chat interface with special commands

### Model Selector

The model selector automatically recommends the optimal model configuration based on your hardware:

- Detects available memory and CPU/GPU capabilities
- Recommends the appropriate model size (3B, 8B, or 24B)
- Suggests quantization level if needed (4-bit or 8-bit)

### Benchmarking Suite

The benchmarking tools allow you to evaluate model performance:

- Compare different DeepHermes models
- Measure inference speed, memory usage, and reasoning quality
- Test on standard datasets like MMLU and GSM8K

## Getting Started

The quickest way to get started is to run the quickstart script:

```bash
bash scripts/quickstart.sh
```

This will:
1. Set up a virtual environment
2. Install all dependencies
3. Download the default model
4. Show you the basic commands to get started

## Usage Examples

After running the quickstart script, you can:

```bash
# Start an interactive chat
python chat.py --reasoning

# Run inference with a prompt
python main.py --reasoning --prompt "Explain quantum computing"

# Use a memory-efficient configuration
python chat.py --reasoning --quantize 4bit
```

For more examples, see the `examples/examples.md` file.

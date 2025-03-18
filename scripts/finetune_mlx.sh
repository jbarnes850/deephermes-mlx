#!/bin/bash
# Fine-tune language models with MLX-LM on Apple Silicon

# Default parameters
MODEL="mlx-community/DeepHermes-3-Llama-3-8B-Preview-bf16"
DATASET_NAME="tatsu-lab/alpaca"
DATA_PATH="./data/alpaca"
ADAPTER_PATH="./adapters"
OUTPUT_DIR="./fine_tuned_model"
NUM_EXAMPLES=100
BATCH_SIZE=1
LEARNING_RATE=1e-4
ITERS=100
LORA_RANK=8
LORA_ALPHA=16
LORA_LAYERS=4
MASK_PROMPT=false
PROMPT=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --dataset-name)
      DATASET_NAME="$2"
      shift 2
      ;;
    --data-path)
      DATA_PATH="$2"
      shift 2
      ;;
    --adapter-path)
      ADAPTER_PATH="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --num-examples)
      NUM_EXAMPLES="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --learning-rate|--lr)
      LEARNING_RATE="$2"
      shift 2
      ;;
    --iters)
      ITERS="$2"
      shift 2
      ;;
    --lora-rank)
      LORA_RANK="$2"
      shift 2
      ;;
    --lora-alpha)
      LORA_ALPHA="$2"
      shift 2
      ;;
    --lora-layers)
      LORA_LAYERS="$2"
      shift 2
      ;;
    --mask-prompt)
      MASK_PROMPT=true
      shift
      ;;
    --prompt)
      PROMPT="$2"
      shift 2
      ;;
    --prepare-data)
      PREPARE_DATA=true
      shift
      ;;
    --train)
      TRAIN=true
      shift
      ;;
    --evaluate)
      EVALUATE=true
      shift
      ;;
    --fuse)
      FUSE=true
      shift
      ;;
    --generate)
      GENERATE=true
      shift
      ;;
    --help)
      echo "Usage: ./finetune_mlx.sh [options] [actions]"
      echo "Options:"
      echo "  --model MODEL           Model ID or path (default: mlx-community/DeepHermes-3-Llama-3-8B-Preview-bf16)"
      echo "  --dataset-name NAME     HuggingFace dataset name (default: tatsu-lab/alpaca)"
      echo "  --data-path PATH        Path to store or load the dataset (default: ./data/alpaca)"
      echo "  --adapter-path PATH     Path to save adapter weights (default: ./adapters)"
      echo "  --output-dir DIR        Output directory for fused model (default: ./fine_tuned_model)"
      echo "  --num-examples NUM      Number of examples to use (default: 100)"
      echo "  --batch-size SIZE       Batch size for training (default: 1)"
      echo "  --learning-rate RATE    Learning rate (default: 1e-4)"
      echo "  --iters NUM             Number of training iterations (default: 100)"
      echo "  --lora-layers NUM       Number of layers to apply LoRA to (default: 4)"
      echo "  --mask-prompt           Whether to mask the prompt during training"
      echo "  --prompt TEXT           Prompt for text generation"
      echo ""
      echo "Actions:"
      echo "  --prepare-data          Prepare the dataset for fine-tuning"
      echo "  --train                 Run the fine-tuning process"
      echo "  --evaluate              Evaluate the fine-tuned model"
      echo "  --fuse                  Fuse the adapter weights with the model"
      echo "  --generate              Generate text with the fine-tuned model"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Activate virtual environment if it exists
if [ -d "venv" ]; then
  source venv/bin/activate
fi

# Build command arguments
CMD_ARGS="--model $MODEL --data_path $DATA_PATH --adapter_path $ADAPTER_PATH --output_dir $OUTPUT_DIR"
CMD_ARGS="$CMD_ARGS --num_examples $NUM_EXAMPLES --batch_size $BATCH_SIZE --learning_rate $LEARNING_RATE"
CMD_ARGS="$CMD_ARGS --iters $ITERS --lora_layers $LORA_LAYERS"

if [ "$MASK_PROMPT" = true ]; then
  CMD_ARGS="$CMD_ARGS --mask_prompt"
fi

if [ ! -z "$DATASET_NAME" ]; then
  CMD_ARGS="$CMD_ARGS --dataset_name $DATASET_NAME"
fi

# Add action flags
if [ "$PREPARE_DATA" = true ]; then
  CMD_ARGS="$CMD_ARGS --prepare_data"
fi

if [ "$TRAIN" = true ]; then
  CMD_ARGS="$CMD_ARGS --train"
fi

if [ "$EVALUATE" = true ]; then
  CMD_ARGS="$CMD_ARGS --evaluate"
fi

if [ "$FUSE" = true ]; then
  CMD_ARGS="$CMD_ARGS --fuse"
fi

if [ "$GENERATE" = true ]; then
  CMD_ARGS="$CMD_ARGS --generate"
fi

# Run the Python script
if [ ! -z "$PROMPT" ]; then
  # Handle prompt separately to avoid quoting issues
  python scripts/finetune_mlx.py $CMD_ARGS --prompt "$PROMPT"
else
  python scripts/finetune_mlx.py $CMD_ARGS
fi

echo "MLX-LM fine-tuning workflow completed!"

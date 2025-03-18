#!/bin/bash
# DeepHermes MLX LangChain Integration CLI
# This script provides a simple interface for using DeepHermes MLX models with LangChain

# Default parameters
MODEL_PATH=""
HOST="127.0.0.1"
PORT="8080"
START_SERVER=false
ADAPTER_PATH=""
MAX_TOKENS=512
TEMPERATURE=0.7
TOP_P=0.9
STREAM=false
COMMAND="chat"
VERBOSE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      MODEL_PATH="$2"
      shift 2
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --start-server)
      START_SERVER=true
      shift
      ;;
    --adapter)
      ADAPTER_PATH="$2"
      shift 2
      ;;
    --max-tokens)
      MAX_TOKENS="$2"
      shift 2
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    --top-p)
      TOP_P="$2"
      shift 2
      ;;
    --stream)
      STREAM=true
      shift
      ;;
    --verbose)
      VERBOSE=true
      shift
      ;;
    generate|chain|chat)
      COMMAND="$1"
      shift
      ;;
    --prompt)
      PROMPT="$2"
      shift 2
      ;;
    --prompt-file)
      PROMPT_FILE="$2"
      shift 2
      ;;
    --topic)
      TOPIC="$2"
      shift 2
      ;;
    --concept)
      CONCEPT="$2"
      shift 2
      ;;
    --system)
      SYSTEM="$2"
      shift 2
      ;;
    --help)
      echo "Usage: ./integrate.sh [options] [command]"
      echo ""
      echo "Commands:"
      echo "  generate    Generate text using DeepHermesLLM"
      echo "  chain       Run a simple LangChain chain"
      echo "  chat        Start an interactive chat session (default)"
      echo ""
      echo "Options:"
      echo "  --model PATH           Path to the model directory"
      echo "  --host HOST            Server host (default: 127.0.0.1)"
      echo "  --port PORT            Server port (default: 8080)"
      echo "  --start-server         Start a server if not running"
      echo "  --adapter PATH         Path to adapter weights"
      echo "  --max-tokens N         Maximum tokens to generate (default: 512)"
      echo "  --temperature T        Sampling temperature (default: 0.7)"
      echo "  --top-p P              Top-p sampling parameter (default: 0.9)"
      echo "  --stream               Stream the output"
      echo "  --verbose              Enable verbose logging"
      echo ""
      echo "Command-specific options:"
      echo "  generate:"
      echo "    --prompt TEXT        Prompt for generation"
      echo "    --prompt-file PATH   File containing the prompt"
      echo ""
      echo "  chain:"
      echo "    --topic TEXT         Topic for the chain (default: artificial intelligence)"
      echo "    --concept TEXT       Concept to explain (default: neural networks)"
      echo ""
      echo "  chat:"
      echo "    --system TEXT        System message (default: You are a helpful assistant.)"
      echo ""
      echo "Examples:"
      echo "  ./integrate.sh --model ./models/deephermes-7b chat"
      echo "  ./integrate.sh --model ./models/deephermes-7b --start-server generate --prompt \"Explain quantum computing\""
      echo "  ./integrate.sh --host 127.0.0.1 --port 8080 chain --topic \"physics\" --concept \"relativity\""
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Activate virtual environment if it exists
if [ -d "venv" ]; then
  source venv/bin/activate
fi

# Build the command
CMD="python -m deephermes.cli.integrate_cli"

# Add server configuration
if [ -n "$MODEL_PATH" ]; then
  CMD="$CMD --model-path \"$MODEL_PATH\""
fi

CMD="$CMD --host \"$HOST\" --port \"$PORT\""

if [ "$START_SERVER" = true ]; then
  CMD="$CMD --start-server"
fi

if [ -n "$ADAPTER_PATH" ]; then
  CMD="$CMD --adapter-path \"$ADAPTER_PATH\""
fi

# Add generation parameters
CMD="$CMD --max-tokens \"$MAX_TOKENS\" --temperature \"$TEMPERATURE\" --top-p \"$TOP_P\""

if [ "$STREAM" = true ]; then
  CMD="$CMD --stream"
fi

if [ "$VERBOSE" = true ]; then
  CMD="$CMD --verbose"
fi

# Add command and its specific options
CMD="$CMD $COMMAND"

case $COMMAND in
  generate)
    if [ -n "$PROMPT" ]; then
      CMD="$CMD --prompt \"$PROMPT\""
    fi
    if [ -n "$PROMPT_FILE" ]; then
      CMD="$CMD --prompt-file \"$PROMPT_FILE\""
    fi
    ;;
  chain)
    if [ -n "$TOPIC" ]; then
      CMD="$CMD --topic \"$TOPIC\""
    fi
    if [ -n "$CONCEPT" ]; then
      CMD="$CMD --concept \"$CONCEPT\""
    fi
    ;;
  chat)
    if [ -n "$SYSTEM" ]; then
      CMD="$CMD --system \"$SYSTEM\""
    fi
    ;;
esac

# Execute the command
eval $CMD

#!/bin/bash
# Serve DeepHermes MLX models using the MLX-LM server

# Default parameters
MODEL=""
HOST="127.0.0.1"
PORT="8080"
ADAPTER_PATH=""
CACHE_LIMIT_GB=""
LOG_LEVEL="INFO"
DETACH=false
COMMAND="start"
PROMPT=""
MAX_TOKENS=512
TEMPERATURE=0.7
TOP_P=0.9
STREAM=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      MODEL="$2"
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
    --adapter-path)
      ADAPTER_PATH="$2"
      shift 2
      ;;
    --cache-limit-gb)
      CACHE_LIMIT_GB="$2"
      shift 2
      ;;
    --log-level)
      LOG_LEVEL="$2"
      shift 2
      ;;
    --detach)
      DETACH=true
      shift
      ;;
    --prompt)
      PROMPT="$2"
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
    start|stop|status|test)
      COMMAND="$1"
      shift
      ;;
    --help)
      echo "Usage: ./serve_model.sh [command] [options]"
      echo ""
      echo "Commands:"
      echo "  start   Start the MLX-LM server (default)"
      echo "  stop    Stop the MLX-LM server"
      echo "  status  Check the status of the MLX-LM server"
      echo "  test    Test the server by generating text"
      echo ""
      echo "Options:"
      echo "  --model PATH           Path to the exported model directory (required for start)"
      echo "  --host HOST            Host to bind the server to (default: 127.0.0.1)"
      echo "  --port PORT            Port to bind the server to (default: 8080)"
      echo "  --adapter-path PATH    Optional path to adapter weights"
      echo "  --cache-limit-gb GB    Memory cache limit in GB"
      echo "  --log-level LEVEL      Logging level (default: INFO)"
      echo "  --detach               Run the server in the background"
      echo ""
      echo "Options for 'test' command:"
      echo "  --prompt TEXT          Text prompt (required for test)"
      echo "  --max-tokens N         Maximum number of tokens to generate (default: 512)"
      echo "  --temperature T        Sampling temperature (default: 0.7)"
      echo "  --top-p P              Top-p sampling parameter (default: 0.9)"
      echo "  --stream               Stream the response"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Run './serve_model.sh --help' for usage information"
      exit 1
      ;;
  esac
done

# Activate virtual environment if it exists
if [ -d "venv" ]; then
  source venv/bin/activate
fi

# Build command based on the selected operation
CMD="python -m deephermes.cli.serve_cli --log-level $LOG_LEVEL"

case $COMMAND in
  start)
    # Check if model is provided
    if [ -z "$MODEL" ]; then
      echo "Error: Model path is required for start command. Use --model to specify."
      exit 1
    fi
    
    CMD="$CMD start --model $MODEL --host $HOST --port $PORT"
    
    # Add optional arguments
    if [ -n "$ADAPTER_PATH" ]; then
      CMD="$CMD --adapter-path $ADAPTER_PATH"
    fi
    
    if [ -n "$CACHE_LIMIT_GB" ]; then
      CMD="$CMD --cache-limit-gb $CACHE_LIMIT_GB"
    fi
    
    if [ "$DETACH" = true ]; then
      CMD="$CMD --detach"
    fi
    ;;
    
  stop)
    CMD="$CMD stop"
    if [ -n "$PORT" ]; then
      CMD="$CMD --port $PORT"
    fi
    ;;
    
  status)
    CMD="$CMD status --host $HOST --port $PORT"
    ;;
    
  test)
    # Check if prompt is provided
    if [ -z "$PROMPT" ]; then
      echo "Error: Prompt is required for test command. Use --prompt to specify."
      exit 1
    fi
    
    CMD="$CMD test --prompt \"$PROMPT\" --host $HOST --port $PORT --max-tokens $MAX_TOKENS --temperature $TEMPERATURE --top-p $TOP_P"
    
    if [ "$STREAM" = true ]; then
      CMD="$CMD --stream"
    fi
    ;;
    
  *)
    echo "Unknown command: $COMMAND"
    echo "Run './serve_model.sh --help' for usage information"
    exit 1
    ;;
esac

# Execute the command
eval $CMD

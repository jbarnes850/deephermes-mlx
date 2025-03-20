#!/bin/bash
# Quickstart script for DeepHermes MLX

# Parse command line arguments
TEST_MODE=false
TEST_OPTION=""
TEST_DATASET=""
TEST_NUM_EXAMPLES=""
TEST_PROMPT=""
SKIP_MODEL_SELECTION=false
FORCE_MODEL_SIZE=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --test)
      TEST_MODE=true
      TEST_OPTION="$2"
      if [[ "$TEST_OPTION" == "2" && $# -gt 2 ]]; then
        TEST_DATASET="$3"
        TEST_NUM_EXAMPLES="$4"
        TEST_PROMPT="$5"
        shift 5
      else
        shift 2
      fi
      ;;
    --skip-model-selection)
      SKIP_MODEL_SELECTION=true
      shift
      ;;
    --force-model)
      FORCE_MODEL_SIZE="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not found. Please install Python 3 and try again."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Use model selector to recommend the best model based on hardware
if [ "$SKIP_MODEL_SELECTION" = false ]; then
    echo ""
    echo "🔍 Analyzing your hardware to recommend the optimal DeepHermes model..."
    
    # Run the model selector CLI to get a recommendation
    if [ -n "$FORCE_MODEL_SIZE" ]; then
        MODEL_CONFIG=$(python -m deephermes.model_selector.cli --force-model-size "$FORCE_MODEL_SIZE" --json)
    else
        MODEL_CONFIG=$(python -m deephermes.model_selector.cli --json)
    fi
    
    # Parse the model ID from the JSON output
    MODEL_ID=$(echo $MODEL_CONFIG | python -c "import sys, json; print(json.load(sys.stdin)['model_id'])")
    QUANTIZATION=$(echo $MODEL_CONFIG | python -c "import sys, json; data=json.load(sys.stdin); print(data.get('quantization', 'none'))")
    REASON=$(echo $MODEL_CONFIG | python -c "import sys, json; print(json.load(sys.stdin)['reason'])")
    
    echo "✅ Recommendation: $MODEL_ID"
    echo "📝 $REASON"
    
    if [ "$QUANTIZATION" != "none" ]; then
        echo "🔧 Using $QUANTIZATION quantization for optimal performance on your hardware."
    fi
else
    # Default model if model selection is skipped
    MODEL_ID="mlx-community/DeepHermes-3-Llama-3-8B-Preview-bf16"
    echo ""
    echo "Using default model: $MODEL_ID"
fi

# Download the recommended model
echo ""
echo "Downloading the model (this may take a few minutes)..."
python -c "from transformers import AutoTokenizer; from mlx_lm import load; model, tokenizer = load('$MODEL_ID'); print('Model downloaded successfully!')"

# Print success message
echo ""
echo "🎉 Setup complete! $MODEL_ID is ready to use."
echo ""

# Present options to the user
echo "What would you like to do?"
echo "1. Launch chat interface with reasoning"
echo "2. Fine-tune a reasoning model"
echo "3. Use LangChain integration"
echo "4. Run adaptive workflow"
echo ""
if [ "$TEST_MODE" = false ]; then
  read -p "Enter your choice (1-4): " choice
else
  choice=$TEST_OPTION
  echo "Selected option: $choice"
fi

case $choice in
    1)
        echo ""
        echo "Starting interactive chat with reasoning mode..."
        echo "Type 'exit' to quit or 'help' to see available commands."
        echo ""
        # Launch the chat interface with reasoning mode
        python chat.py --reasoning --model "$MODEL_ID"
        ;;
    2)
        echo ""
        echo "Starting fine-tuning workflow..."
        echo ""
        
        # Ask for dataset
        echo "Which dataset would you like to use for fine-tuning?"
        echo "1. Alpaca (instruction following)"
        echo "2. Custom dataset (provide path)"
        if [ "$TEST_MODE" = false ]; then
          read -p "Enter your choice (1 or 2): " dataset_choice
        else
          dataset_choice=$TEST_DATASET
          echo "Selected dataset: $dataset_choice"
        fi
        
        if [ "$dataset_choice" = "1" ]; then
            dataset="tatsu-lab/alpaca"
            echo "Using the Alpaca dataset for instruction tuning."
        else
            if [ "$TEST_MODE" = false ]; then
              read -p "Enter the path to your dataset: " dataset
            else
              dataset=$TEST_DATASET
              echo "Using dataset: $dataset"
            fi
        fi
        
        # Ask for number of examples
        if [ "$TEST_MODE" = false ]; then
          read -p "How many examples to use (default: 100): " num_examples
          num_examples=${num_examples:-100}
        else
          num_examples=$TEST_NUM_EXAMPLES
          echo "Using $num_examples examples"
        fi
        
        # Confirm and run
        echo ""
        echo "Ready to fine-tune $MODEL_ID with the following settings:"
        echo "- Dataset: $dataset"
        echo "- Number of examples: $num_examples"
        echo ""
        
        if [ "$TEST_MODE" = false ]; then
          read -p "Proceed with fine-tuning? (y/n): " confirm
        else
          confirm="y"
          echo "Proceeding with fine-tuning..."
        fi
        
        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            # Run fine-tuning
            ./scripts/finetune_mlx.sh --model "$MODEL_ID" \
                --dataset-name "$dataset" \
                --num-examples "$num_examples" \
                --prepare-data --train
            
            # Ask if user wants to generate with the fine-tuned model
            if [ "$TEST_MODE" = false ]; then
                read -p "Generate text with your fine-tuned model? (y/n): " generate
            else
                generate="y"
                echo "Generating text with fine-tuned model..."
            fi
            
            if [ "$generate" = "y" ] || [ "$generate" = "Y" ]; then
                if [ "$TEST_MODE" = false ]; then
                  read -p "Enter your prompt: " prompt
                else
                  prompt="$TEST_PROMPT"
                  echo "Using prompt: $prompt"
                fi
                ./scripts/finetune_mlx.sh --model "$MODEL_ID" \
                    --prompt "$prompt" \
                    --generate
            fi
        else
            echo "Fine-tuning cancelled."
        fi
        ;;
    3)
        echo ""
        echo "Starting LangChain integration..."
        echo ""
        
        # Check if model path is provided
        if [ "$TEST_MODE" = false ]; then
            read -p "Do you want to start a new server with the recommended model? (y/n): " start_server
        else
            start_server="n"
            echo "Using existing server..."
        fi
        
        if [ "$start_server" = "y" ] || [ "$start_server" = "Y" ]; then
            server_args="--model $MODEL_ID --start-server"
        else
            if [ "$TEST_MODE" = false ]; then
                read -p "Enter the host (default: 127.0.0.1): " host
                host=${host:-127.0.0.1}
                
                read -p "Enter the port (default: 8080): " port
                port=${port:-8080}
            else
                host="127.0.0.1"
                port="8080"
                echo "Using host: $host, port: $port"
            fi
            
            server_args="--host $host --port $port"
        fi
        
        # Ask for the LangChain mode
        if [ "$TEST_MODE" = false ]; then
            echo ""
            echo "Select LangChain mode:"
            echo "1. Chat interface"
            echo "2. Text generation"
            echo "3. Chain example"
            read -p "Enter your choice (1-3): " lc_mode
        else
            lc_mode="1"
            echo "Using chat interface mode"
        fi
        
        case $lc_mode in
            1)
                echo ""
                echo "Starting chat interface..."
                ./scripts/integrate.sh $server_args chat
                ;;
            2)
                if [ "$TEST_MODE" = false ]; then
                    read -p "Enter your prompt: " prompt
                else
                    prompt="Explain quantum computing in simple terms"
                    echo "Using prompt: $prompt"
                fi
                
                echo ""
                echo "Generating text..."
                ./scripts/integrate.sh $server_args generate --prompt "$prompt"
                ;;
            3)
                if [ "$TEST_MODE" = false ]; then
                    read -p "Enter a topic (default: artificial intelligence): " topic
                    topic=${topic:-"artificial intelligence"}
                    
                    read -p "Enter a concept to explain (default: neural networks): " concept
                    concept=${concept:-"neural networks"}
                else
                    topic="artificial intelligence"
                    concept="neural networks"
                    echo "Using topic: $topic, concept: $concept"
                fi
                
                echo ""
                echo "Running chain example..."
                ./scripts/integrate.sh $server_args chain --topic "$topic" --concept "$concept"
                ;;
            *)
                echo "Invalid choice. Using chat interface."
                ./scripts/integrate.sh $server_args chat
                ;;
        esac
        ;;
    4)
        echo ""
        echo "Starting adaptive workflow..."
        echo ""
        
        # Ask for workflow template
        echo "Select a workflow template:"
        echo "1. General (balanced configuration for general-purpose use)"
        echo "2. Content Creation (optimized for generating creative content)"
        echo "3. Coding (specialized for code generation and completion)"
        echo "4. Research (focused on high-quality, in-depth responses)"
        if [ "$TEST_MODE" = false ]; then
          read -p "Enter your choice (1-4): " workflow_choice
        else
          workflow_choice="1"
          echo "Selected workflow: General"
        fi
        
        # Ask for performance priority
        echo ""
        echo "Select performance priority:"
        echo "1. Balanced (default)"
        echo "2. Speed (faster responses, potentially lower quality)"
        echo "3. Quality (higher quality responses, potentially slower)"
        if [ "$TEST_MODE" = false ]; then
          read -p "Enter your choice (1-3): " priority_choice
        else
          priority_choice="1"
          echo "Selected priority: Balanced"
        fi
        
        # Set workflow and priority options
        case $workflow_choice in
            1)
                workflow="general"
                ;;
            2)
                workflow="content_creation"
                ;;
            3)
                workflow="coding"
                ;;
            4)
                workflow="research"
                ;;
            *)
                echo "Invalid choice. Using general workflow."
                workflow="general"
                ;;
        esac
        
        case $priority_choice in
            2)
                priority_option="--prioritize-speed"
                ;;
            3)
                priority_option="--prioritize-quality"
                ;;
            *)
                priority_option=""
                ;;
        esac
        
        # Ask if user wants to use LangChain integration
        echo ""
        if [ "$TEST_MODE" = false ]; then
          read -p "Use LangChain integration? (y/n): " use_langchain
        else
          use_langchain="n"
          echo "Not using LangChain integration"
        fi
        
        if [ "$use_langchain" = "y" ] || [ "$use_langchain" = "Y" ]; then
            langchain_option="--langchain"
        else
            langchain_option=""
        fi
        
        echo ""
        echo "Running $workflow workflow..."
        ./scripts/adaptive_workflow.sh run --workflow "$workflow" $priority_option $langchain_option
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

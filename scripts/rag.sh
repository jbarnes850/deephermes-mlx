#!/bin/bash
# DeepHermes MLX RAG (Retrieval-Augmented Generation) script

# Default parameters
VECTOR_STORE_PATH="$HOME/.deephermes/vector_store"
EMBEDDING_MODEL="e5-mistral-7b"
COLLECTION="default"
VERBOSE=false

# Function to display help
show_help() {
    echo "DeepHermes MLX RAG (Retrieval-Augmented Generation) Script"
    echo ""
    echo "Usage: ./rag.sh [command] [options]"
    echo ""
    echo "Commands:"
    echo "  add <paths...>      Add documents or directories to the RAG system"
    echo "  query [query]       Query the RAG system (omit query for interactive mode)"
    echo "  list                List collections in the vector store"
    echo "  delete <collection> Delete a collection from the vector store"
    echo ""
    echo "Options:"
    echo "  --vector-store-path PATH  Path to the vector store directory (default: ~/.deephermes/vector_store)"
    echo "  --embedding-model MODEL   Name of the embedding model to use (default: e5-mistral-7b)"
    echo "  --collection NAME         Name of the collection to use (default: default)"
    echo "  --verbose                 Enable verbose output"
    echo ""
    echo "Add command options:"
    echo "  --chunk-size SIZE         Size of chunks to split documents into (default: 1000)"
    echo "  --chunk-overlap SIZE      Overlap between chunks (default: 200)"
    echo "  --recursive               Recursively process subdirectories"
    echo "  --extensions EXT1,EXT2    Comma-separated list of file extensions to process"
    echo ""
    echo "Delete command options:"
    echo "  --force                   Force deletion without confirmation"
    echo ""
    echo "Examples:"
    echo "  ./rag.sh add ~/Documents/research --collection research"
    echo "  ./rag.sh query \"What is the main topic of my research?\""
    echo "  ./rag.sh query --collection research"
    echo "  ./rag.sh list"
    echo "  ./rag.sh delete old_collection --force"
}

# Parse command
if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

COMMAND=$1
shift

# Parse options
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RECURSIVE=false
EXTENSIONS=""
FORCE=false
PATHS=()
QUERY=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --vector-store-path)
            VECTOR_STORE_PATH="$2"
            shift 2
            ;;
        --embedding-model)
            EMBEDDING_MODEL="$2"
            shift 2
            ;;
        --collection)
            COLLECTION="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --chunk-size)
            CHUNK_SIZE="$2"
            shift 2
            ;;
        --chunk-overlap)
            CHUNK_OVERLAP="$2"
            shift 2
            ;;
        --recursive)
            RECURSIVE=true
            shift
            ;;
        --extensions)
            EXTENSIONS="$2"
            shift 2
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        -*)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
        *)
            if [ "$COMMAND" = "query" ] && [ -z "$QUERY" ]; then
                QUERY="$1"
            elif [ "$COMMAND" = "delete" ] && [ -z "$COLLECTION" ]; then
                COLLECTION="$1"
            else
                PATHS+=("$1")
            fi
            shift
            ;;
    esac
done

# Build common options
COMMON_OPTS="--vector-store-path \"$VECTOR_STORE_PATH\" --embedding-model \"$EMBEDDING_MODEL\""
if [ "$VERBOSE" = true ]; then
    COMMON_OPTS="$COMMON_OPTS --verbose"
fi

# Execute command
case $COMMAND in
    add)
        if [ ${#PATHS[@]} -eq 0 ]; then
            echo "Error: No paths specified for add command"
            exit 1
        fi
        
        # Build add options
        ADD_OPTS="--collection \"$COLLECTION\" --chunk-size $CHUNK_SIZE --chunk-overlap $CHUNK_OVERLAP"
        if [ "$RECURSIVE" = true ]; then
            ADD_OPTS="$ADD_OPTS --recursive"
        fi
        if [ -n "$EXTENSIONS" ]; then
            ADD_OPTS="$ADD_OPTS --extensions \"$EXTENSIONS\""
        fi
        
        # Build paths string
        PATHS_STR=""
        for path in "${PATHS[@]}"; do
            PATHS_STR="$PATHS_STR \"$path\""
        done
        
        # Run command
        eval "python -m deephermes.cli.rag_cli $COMMON_OPTS add $ADD_OPTS $PATHS_STR"
        ;;
    query)
        # Build query options
        QUERY_OPTS="--collection \"$COLLECTION\""
        
        # Run command
        if [ -z "$QUERY" ]; then
            # Interactive mode
            eval "python -m deephermes.cli.rag_cli $COMMON_OPTS query $QUERY_OPTS --interactive"
        else
            # Single query
            eval "python -m deephermes.cli.rag_cli $COMMON_OPTS query $QUERY_OPTS \"$QUERY\""
        fi
        ;;
    list)
        # Run command
        eval "python -m deephermes.cli.rag_cli $COMMON_OPTS list"
        ;;
    delete)
        if [ -z "$COLLECTION" ]; then
            echo "Error: No collection specified for delete command"
            exit 1
        fi
        
        # Build delete options
        DELETE_OPTS=""
        if [ "$FORCE" = true ]; then
            DELETE_OPTS="--force"
        fi
        
        # Run command
        eval "python -m deephermes.cli.rag_cli $COMMON_OPTS delete $DELETE_OPTS \"$COLLECTION\""
        ;;
    *)
        echo "Unknown command: $COMMAND"
        show_help
        exit 1
        ;;
esac

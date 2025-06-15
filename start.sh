#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}Starting Gemma3 RKLLM Server...${NC}"

if [ ! -d "venv" ]; then
    echo -e "${RED}Virtual environment not found. Please run setup.sh first.${NC}"
    exit 1
fi

source venv/bin/activate

if [ ! -f "lib/librkllmrt.so" ]; then
    echo -e "${RED}RKLLM library not found. Please install librkllmrt.so in the lib/ directory.${NC}"
    exit 1
fi

python server.py "$@"

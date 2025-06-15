#!/bin/bash
echo "Stopping Gemma3 RKLLM Server..."
pkill -f "python.*server.py" || true
echo "Server stopped."

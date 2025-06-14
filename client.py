#!/usr/bin/env python3
"""
Simple client for Gemma3 RKLLM server
"""

import requests
import json
import sys
import argparse
import base64
from pathlib import Path

def send_text_request(url, prompt, model="gemma3", stream=False):
    """Send text-only request"""
    data = {
        "model": model,
        "prompt": prompt,
        "stream": stream
    }
    
    try:
        response = requests.post(f"{url}/api/generate", json=data, stream=stream)
        response.raise_for_status()
        
        if stream:
            for line in response.iter_lines():
                if line:
                    try:
                        result = json.loads(line)
                        if not result.get("done", False):
                            print(result.get("response", ""), end="", flush=True)
                    except json.JSONDecodeError:
                        continue
            print()  # New line at end
        else:
            result = response.json()
            print(result.get("response", ""))
            
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        sys.exit(1)

def send_multimodal_request(url, prompt, image_path, model="gemma3"):
    """Send multimodal request with image"""
    try:
        # Read and encode image
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()
        
        data = {
            "model": model,
            "prompt": prompt,
            "images": [f"data:image/jpeg;base64,{image_data}"]
        }
        
        response = requests.post(f"{url}/generate", json=data)
        response.raise_for_status()
        result = response.json()
        print(result.get("text", ""))
        
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Gemma3 RKLLM Client")
    parser.add_argument("--url", default="http://localhost:8080", help="Server URL")
    parser.add_argument("--model", default="gemma3", help="Model name")
    parser.add_argument("--prompt", required=True, help="Text prompt")
    parser.add_argument("--image", help="Path to image file")
    parser.add_argument("--stream", action="store_true", help="Enable streaming")
    
    args = parser.parse_args()
    
    try:
        if args.image:
            if not Path(args.image).exists():
                print(f"Error: Image file not found: {args.image}")
                sys.exit(1)
            send_multimodal_request(args.url, args.prompt, args.image, args.model)
        else:
            send_text_request(args.url, args.prompt, args.model, args.stream)
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to server at {args.url}")
        print("Make sure the server is running with: ./start.sh")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()


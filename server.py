#!/usr/bin/env python3
"""
Gemma3 RKLLM Server
A Flask-based server for running Gemma3 multimodal models on Rockchip NPU

Author: AI Assistant
License: MIT
"""

import os
import sys
import json
import base64
import logging
import configparser
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

import click
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from PIL import Image
import io

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.gemma3_model import Gemma3Model
from src.image_processor import ImageProcessor
from src.config_manager import ConfigManager
from src.logger import setup_logger
from src.api_handlers import APIHandlers
from src.utils import validate_request, handle_errors

# Initialize configuration
config_manager = ConfigManager()
config = config_manager.load_config()

# Setup logging
logger = setup_logger(config)

# Initialize Flask app
app = Flask(__name__)

# Enable CORS
if config.getboolean('server', 'cors_enabled', fallback=True):
    CORS(app, origins=config.get('security', 'allowed_origins', fallback='*'))

# Global variables
current_model: Optional[Gemma3Model] = None
image_processor = ImageProcessor(config)
api_handlers = APIHandlers(config)

@app.errorhandler(Exception)
def handle_exception(e):
    """Global error handler"""
    logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
    return jsonify({
        "error": "Internal server error",
        "message": str(e) if config.getboolean('server', 'debug', fallback=False) else "An error occurred"
    }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": current_model is not None,
        "version": "1.0.0"
    })

@app.route('/models', methods=['GET'])
def list_models():
    """List available models"""
    try:
        models_dir = Path(config.get('model', 'models_dir', fallback='./models'))
        models = []
        
        if models_dir.exists():
            for model_dir in models_dir.iterdir():
                if model_dir.is_dir():
                    modelfile_path = model_dir / "Modelfile"
                    if modelfile_path.exists():
                        models.append({
                            "name": model_dir.name,
                            "path": str(model_dir),
                            "multimodal": True,  # Gemma3 is multimodal
                            "size": "unknown"  # Could be calculated
                        })
        
        return jsonify({"models": models})
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/load_model', methods=['POST'])
def load_model():
    """Load a model into NPU"""
    global current_model
    
    try:
        data = request.get_json()
        if not data or 'model' not in data:
            return jsonify({"error": "Model name required"}), 400
        
        model_name = data['model']
        
        # Unload current model if exists
        if current_model:
            current_model.release()
            current_model = None
        
        # Load new model
        current_model = Gemma3Model(model_name, config)
        
        logger.info(f"Model {model_name} loaded successfully")
        return jsonify({
            "message": f"Model {model_name} loaded successfully",
            "model": model_name,
            "multimodal": True
        })
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/unload_model', methods=['POST'])
def unload_model():
    """Unload current model"""
    global current_model
    
    try:
        if current_model:
            current_model.release()
            current_model = None
            logger.info("Model unloaded successfully")
            return jsonify({"message": "Model unloaded successfully"})
        else:
            return jsonify({"message": "No model loaded"})
            
    except Exception as e:
        logger.error(f"Error unloading model: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/generate', methods=['POST'])
def generate():
    """Generate text with optional image input"""
    if not current_model:
        return jsonify({"error": "No model loaded"}), 400
    
    try:
        data = request.get_json()
        
        # Validate request
        if not validate_request(data, ['prompt']):
            return jsonify({"error": "Invalid request format"}), 400
        
        prompt = data['prompt']
        images = data.get('images', [])
        stream = data.get('stream', False)
        
        # Process images if provided
        processed_images = []
        if images:
            for img_data in images:
                if isinstance(img_data, str):
                    # Base64 encoded image
                    processed_img = image_processor.process_base64_image(img_data)
                    processed_images.append(processed_img)
        
        # Generate response
        if stream:
            return Response(
                stream_with_context(current_model.generate_stream(prompt, processed_images)),
                content_type='application/x-ndjson'
            )
        else:
            response = current_model.generate(prompt, processed_images)
            return jsonify(response)
            
    except Exception as e:
        logger.error(f"Error in generation: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def ollama_chat():
    """Ollama-compatible chat endpoint with multimodal support"""
    if not current_model:
        return jsonify({"error": "No model loaded"}), 400
    
    try:
        data = request.get_json()
        return api_handlers.handle_chat_request(data, current_model)
        
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/generate', methods=['POST'])
def ollama_generate():
    """Ollama-compatible generate endpoint"""
    if not current_model:
        return jsonify({"error": "No model loaded"}), 400
    
    try:
        data = request.get_json()
        return api_handlers.handle_generate_request(data, current_model)
        
    except Exception as e:
        logger.error(f"Error in generate: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/upload_image', methods=['POST'])
def upload_image():
    """Upload and process image for multimodal input"""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Process uploaded image
        image_data = file.read()
        processed_image = image_processor.process_image_bytes(image_data)
        
        # Convert to base64 for response
        img_base64 = base64.b64encode(image_data).decode('utf-8')
        
        return jsonify({
            "message": "Image processed successfully",
            "image_id": processed_image.get('id'),
            "size": processed_image.get('size'),
            "format": processed_image.get('format')
        })
        
    except Exception as e:
        logger.error(f"Error uploading image: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/config', methods=['GET'])
def get_config():
    """Get current configuration"""
    try:
        # Return safe configuration (no sensitive data)
        safe_config = {
            "server": {
                "host": config.get('server', 'host'),
                "port": config.getint('server', 'port'),
                "debug": config.getboolean('server', 'debug')
            },
            "model": {
                "max_context_length": config.getint('model', 'max_context_length'),
                "default_temperature": config.getfloat('model', 'default_temperature')
            },
            "multimodal": {
                "max_image_size": config.getint('multimodal', 'max_image_size'),
                "supported_formats": config.get('multimodal', 'supported_formats').split(',')
            }
        }
        return jsonify(safe_config)
        
    except Exception as e:
        logger.error(f"Error getting config: {str(e)}")
        return jsonify({"error": str(e)}), 500

@click.command()
@click.option('--host', default=None, help='Host to bind to')
@click.option('--port', default=None, type=int, help='Port to bind to')
@click.option('--debug', is_flag=True, help='Enable debug mode')
@click.option('--config', default='config/default.ini', help='Configuration file path')
def main(host, port, debug, config_file):
    """Start the Gemma3 RKLLM server"""
    
    # Load configuration
    global config
    config = config_manager.load_config(config_file)
    
    # Override with command line arguments
    if host:
        config.set('server', 'host', host)
    if port:
        config.set('server', 'port', str(port))
    if debug:
        config.set('server', 'debug', 'true')
    
    # Get final configuration
    final_host = config.get('server', 'host', fallback='0.0.0.0')
    final_port = config.getint('server', 'port', fallback=8080)
    final_debug = config.getboolean('server', 'debug', fallback=False)
    
    logger.info(f"Starting Gemma3 RKLLM server on {final_host}:{final_port}")
    logger.info(f"Debug mode: {final_debug}")
    logger.info(f"Multimodal support: Enabled")
    
    # Start Flask app
    app.run(
        host=final_host,
        port=final_port,
        debug=final_debug,
        threaded=True
    )

if __name__ == '__main__':
    main()


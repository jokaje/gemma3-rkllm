import os
import sys
import logging
import configparser
from pathlib import Path
from datetime import datetime

import click
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import the new generic model class
from src.rkllm_model import RKLLMModel
from src.image_processor import ImageProcessor
from src.config_manager import ConfigManager
from src.logger import setup_logger
from src.api_handlers import APIHandlers
from src.utils import validate_request

# --- Global Initialization ---
config_manager = ConfigManager()
config = config_manager.load_config()
logger = setup_logger(config)

# Use the new generic RKLLMModel class
current_model: RKLLMModel | None = None
image_processor = ImageProcessor(config)
api_handlers = APIHandlers(config)


# Initialize Flask app
app = Flask(__name__)

# Enable CORS
if config.getboolean('server', 'cors_enabled', fallback=True):
    CORS(app, origins=config.get('security', 'allowed_origins', fallback='*'))


# --- API Routes ---

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
    model_name = current_model.model_name if current_model else None
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": current_model is not None,
        "loaded_model_name": model_name,
        "version": "1.1.0-refactored"
    })

@app.route('/models', methods=['GET'])
def list_models():
    """List available models by scanning the models directory."""
    try:
        models_dir = Path(config.get('model', 'models_dir', fallback='./models'))
        models = []
        
        if models_dir.exists():
            for model_dir in models_dir.iterdir():
                if model_dir.is_dir():
                    # Look for Modelfile or Modelfile.ini
                    modelfile_path = model_dir / "Modelfile"
                    if not modelfile_path.exists():
                        modelfile_path = model_dir / "Modelfile.ini"
                    
                    if modelfile_path.exists():
                        models.append({
                            "name": model_dir.name,
                            "path": str(model_dir)
                        })
        
        return jsonify({"models": models})
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/load_model', methods=['POST'])
def load_model():
    """Load a model into the NPU."""
    global current_model
    
    try:
        data = request.get_json()
        if not data or 'model' not in data:
            return jsonify({"error": "Model name required"}), 400
        
        model_name = data['model']
        
        # Unload current model if it exists
        if current_model:
            logger.info(f"Unloading current model: {current_model.model_name}")
            current_model.release()
            current_model = None
        
        # Load the new model using the generic class
        logger.info(f"Attempting to load new model: {model_name}")
        current_model = RKLLMModel(model_name, config)
        
        logger.info(f"Model {model_name} loaded successfully")
        return jsonify({
            "message": f"Model {model_name} loaded successfully",
            "model": model_name
        })
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/unload_model', methods=['POST'])
def unload_model():
    """Unload the current model."""
    global current_model
    
    try:
        if current_model:
            model_name = current_model.model_name
            current_model.release()
            current_model = None
            logger.info(f"Model {model_name} unloaded successfully")
            return jsonify({"message": f"Model {model_name} unloaded successfully"})
        else:
            return jsonify({"message": "No model was loaded"})
            
    except Exception as e:
        logger.error(f"Error unloading model: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Ollama-compatible endpoints now correctly delegate to the API handler
@app.route('/api/chat', methods=['POST'])
def ollama_chat():
    """Ollama-compatible chat endpoint."""
    if not current_model:
        return jsonify({"error": "No model loaded. Please use /load_model first."}), 400
    
    try:
        data = request.get_json()
        return api_handlers.handle_chat_request(data, current_model)
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/generate', methods=['POST'])
def ollama_generate():
    """Ollama-compatible generate endpoint."""
    if not current_model:
        return jsonify({"error": "No model loaded. Please use /load_model first."}), 400
    
    try:
        data = request.get_json()
        return api_handlers.handle_generate_request(data, current_model)
        
    except Exception as e:
        logger.error(f"Error in generate endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# --- Main function to start the server ---

@click.command()
@click.option('--host', default=None, help='Host to bind to.')
@click.option('--port', default=None, type=int, help='Port to bind to.')
@click.option('--debug', is_flag=True, help='Enable debug mode.')
@click.option('--config', 'config_path', default='config/default.ini', help='Configuration file path.')
def main(host, port, debug, config_path):
    """Start the Model-Agnostic RKLLM server."""
    global config, logger, image_processor, api_handlers
    
    # Load configuration from the specified file and update the global object
    config = config_manager.load_config(config_path)

    # Re-initialize logger and other components with the potentially new config
    logger = setup_logger(config)
    image_processor = ImageProcessor(config)
    api_handlers = APIHandlers(config)
    
    # Override configuration with command-line arguments if provided
    if host:
        config.set('server', 'host', host)
    if port:
        config.set('server', 'port', str(port))

    # Command-line debug flag takes precedence over the config file
    final_debug = debug or config.getboolean('server', 'debug', fallback=False)

    # Read final configuration values for server startup
    final_host = config.get('server', 'host', fallback='0.0.0.0')
    final_port = config.getint('server', 'port', fallback=8080)
    
    logger.info(f"Starting Model-Agnostic RKLLM Server on {final_host}:{final_port}")
    logger.info(f"Debug mode: {final_debug}")
    logger.info("Ready to load any compatible model from the 'models' directory.")
    
    # Start the Flask application
    app.run(
        host=final_host,
        port=final_port,
        debug=final_debug,
        threaded=True
    )

if __name__ == '__main__':
    main()

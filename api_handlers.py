"""
API Handlers for Gemma3 RKLLM
Handles different API formats including Ollama compatibility.
This version delegates prompt formatting and history management to the model class.
"""

import json
import time
import logging
from datetime import datetime  # <-- FIX IS HERE: This import was missing
from typing import Dict, List, Any, Optional, Generator
from flask import Response, stream_with_context, jsonify
import configparser

from .utils import validate_request, create_error_response

class APIHandlers:
    """Handles different API request formats by passing data to the model class."""
    
    def __init__(self, config: configparser.ConfigParser):
        self.config = config
        self.logger = logging.getLogger("gemma3-rkllm.api")
    
    def handle_chat_request(self, data: Dict[str, Any], model) -> Response:
        """
        Handle Ollama-compatible chat request.
        Passes the full message list to the model for template processing.
        """
        try:
            if not validate_request(data, ['model', 'messages']):
                return jsonify(create_error_response(
                    "invalid_request", 
                    "Missing required fields: model, messages"
                )), 400
            
            messages = data.get('messages', [])
            stream = data.get('stream', False)
            
            # --- Image extraction logic (can be expanded later) ---
            images = []
            if messages:
                last_message = messages[-1]
                if isinstance(last_message.get('content'), list):
                    for item in last_message['content']:
                        if item.get('type') == 'image_url':
                            images.append(item.get('image_url', {}).get('url', ''))
            
            self.logger.info(f"Handling chat request for model '{data.get('model')}'. Stream: {stream}. Messages: {len(messages)}. Images: {len(images)}")
            
            if stream:
                return self._stream_chat_response(messages, images, model, data)
            else:
                return self._generate_chat_response(messages, images, model, data)
                
        except Exception as e:
            self.logger.error(f"Error in chat request: {str(e)}", exc_info=True)
            return jsonify(create_error_response("generation_error", str(e))), 500
    
    def handle_generate_request(self, data: Dict[str, Any], model) -> Response:
        """
        Handle Ollama-compatible generate request.
        Wraps the single prompt into a 'messages' list to be processed by the model's template engine.
        """
        try:
            if not validate_request(data, ['model', 'prompt']):
                return jsonify(create_error_response(
                    "invalid_request", 
                    "Missing required fields: model, prompt"
                )), 400
            
            prompt = data['prompt']
            images_b64 = data.get('images', []) # Expects a list of base64 strings
            stream = data.get('stream', False)
            
            # Convert the single prompt into the standard messages format
            messages = [{"role": "user", "content": prompt}]
            
            # Convert raw base64 to data URLs for the model's image processor
            images_urls = [f"data:image/jpeg;base64,{b64}" for b64 in images_b64]

            if stream:
                generator = model.generate_stream(messages, images_urls, **self._extract_generation_params(data))
                return self._stream_generate_response(generator, data, model)
            else:
                result = model.generate(messages, images_urls, **self._extract_generation_params(data))
                return self._format_generate_response(result, data, model)
                
        except Exception as e:
            self.logger.error(f"Error in generate request: {str(e)}", exc_info=True)
            return jsonify(create_error_response("generation_error", str(e))), 500

    def _generate_chat_response(self, messages: List[Dict], images: List[str], model, data: Dict) -> Response:
        """Generate non-streaming chat response."""
        start_time = time.time()
        
        result = model.generate(messages, images, **self._extract_generation_params(data))
        
        generation_time = result.get('generation_time', time.time() - start_time)
        
        response = {
            "model": model.model_name,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "message": {
                "role": "assistant",
                "content": result.get('text', '')
            },
            "done": True,
            "total_duration": int(generation_time * 1e9),
            "eval_duration": int(generation_time * 1e9)
        }
        return jsonify(response)
    
    def _format_generate_response(self, result: Dict, data: Dict, model) -> Response:
        """Formats the result from a non-streaming /api/generate call."""
        generation_time = result.get('generation_time', 0)
        response = {
            "model": model.model_name,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "response": result.get('text', ''),
            "done": True,
            "total_duration": int(generation_time * 1e9),
        }
        return jsonify(response)

    def _stream_chat_response(self, messages: List[Dict], images: List[str], model, data: Dict) -> Response:
        """Generate streaming chat response."""
        def generate_stream_content():
            try:
                # The model's stream generator now handles templating and history
                for chunk in model.generate_stream(messages, images, **self._extract_generation_params(data)):
                    if chunk.get('done'):
                        break
                    
                    response = {
                        "model": model.model_name,
                        "created_at": datetime.utcnow().isoformat() + "Z",
                        "message": {
                            "role": "assistant",
                            "content": chunk.get('text', '')
                        },
                        "done": False
                    }
                    yield f"data: {json.dumps(response)}\n\n"
                
                # Final "done" message
                done_response = {
                    "model": model.model_name,
                    "created_at": datetime.utcnow().isoformat() + "Z",
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                }
                yield f"data: {json.dumps(done_response)}\n\n"
                
            except Exception as e:
                self.logger.error(f"Error during stream generation: {e}", exc_info=True)
                error_response = {"error": str(e)}
                yield f"data: {json.dumps(create_error_response('stream_error', str(e)))}\n\n"
        
        return Response(stream_with_context(generate_stream_content()), mimetype='application/x-ndjson')
    
    def _stream_generate_response(self, generator: Generator, data: Dict, model) -> Response:
        """Generate streaming response for /api/generate."""
        def generate_stream_content():
            try:
                for chunk in generator:
                    if chunk.get('done'):
                        break
                    
                    response = {
                        "model": model.model_name,
                        "created_at": datetime.utcnow().isoformat() + "Z",
                        "response": chunk.get('text', ''),
                        "done": False
                    }
                    yield f"{json.dumps(response)}\n"

                # Final "done" message
                done_response = {
                    "model": model.model_name,
                    "created_at": datetime.utcnow().isoformat() + "Z",
                    "response": "",
                    "done": True,
                }
                yield f"{json.dumps(done_response)}\n"

            except Exception as e:
                self.logger.error(f"Error during generate stream: {e}", exc_info=True)
                error_response = {"error": str(e)}
                yield f"{json.dumps(error_response)}\n"
        
        return Response(stream_with_context(generate_stream_content()), mimetype='application/x-ndjson')

    def _extract_generation_params(self, data: Dict) -> Dict[str, Any]:
        """Extract generation parameters from request 'options' dictionary."""
        options = data.get('options', {})
        return {
            'temperature': options.get('temperature'),
            'top_p': options.get('top_p'),
            'top_k': options.get('top_k'),
            'num_predict': options.get('num_predict'),
            'stop': options.get('stop'),
            'repeat_penalty': options.get('repeat_penalty'),
        }

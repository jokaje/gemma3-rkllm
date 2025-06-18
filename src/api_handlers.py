"""
API Handlers for Gemma3 RKLLM
Handles different API formats including Ollama compatibility.
This version delegates prompt formatting to the model class.
"""

import json
import time
import logging
from typing import Dict, List, Any, Optional, Generator
from flask import Response, stream_with_context, jsonify
import configparser

from .utils import validate_request, create_error_response, create_success_response

class APIHandlers:
    """Handles different API request formats"""
    
    def __init__(self, config: configparser.ConfigParser):
        self.config = config
        self.logger = logging.getLogger("gemma3-rkllm.api")
        self.ollama_enabled = config.getboolean('api', 'enable_ollama_compatibility', fallback=True)
        self.openai_enabled = config.getboolean('api', 'enable_openai_compatibility', fallback=False)
    
    def handle_chat_request(self, data: Dict[str, Any], model) -> Response:
        """
        Handle Ollama-compatible chat request.
        Extracts messages and images, then passes them to the model class for processing.
        """
        try:
            # Validate request
            if not validate_request(data, ['model', 'messages']):
                return jsonify(create_error_response(
                    "invalid_request", 
                    "Missing required fields: model, messages"
                )), 400
            
            messages = data.get('messages', [])
            stream = data.get('stream', False)
            
            # --- Simplified Logic: Extract messages and images ---
            images = []
            if messages:
                # Look for images in the last user message, as is common
                last_message_content = messages[-1].get('content', '')
                if isinstance(last_message_content, list):
                    for item in last_message_content:
                        if item.get('type') == 'image_url':
                            img_data_url = item.get('image_url', {}).get('url', '')
                            if img_data_url:
                                images.append(img_data_url)
            # --- End Simplified Logic ---
            
            self.logger.info(f"Handling chat request for model '{data.get('model')}'. Stream: {stream}. Images: {len(images)}")
            
            # Generate response
            if stream:
                return self._stream_chat_response(messages, images, model, data)
            else:
                return self._generate_chat_response(messages, images, model, data)
                
        except Exception as e:
            self.logger.error(f"Error in chat request: {str(e)}", exc_info=True)
            return jsonify(create_error_response(
                "generation_error", 
                str(e)
            )), 500
    
    def handle_generate_request(self, data: Dict[str, Any], model) -> Response:
        """
        Handle Ollama-compatible generate request.
        Wraps the single prompt into a 'messages' list for compatibility with the model class.
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
            
            # Convert the single prompt into the messages format
            messages = [{"role": "user", "content": prompt}]
            
            # For this endpoint, images are passed as raw base64. Convert to data URLs.
            images_urls = [f"data:image/jpeg;base64,{b64}" for b64 in images_b64]

            if stream:
                # The generate_stream method now takes the messages list
                generator = model.generate_stream(messages, images_urls, **self._extract_generation_params(data))
                return self._stream_generate_response(generator, data)
            else:
                result = model.generate(messages, images_urls, **self._extract_generation_params(data))
                return self._generate_response(result, data)
                
        except Exception as e:
            self.logger.error(f"Error in generate request: {str(e)}", exc_info=True)
            return jsonify(create_error_response("generation_error", str(e))), 500

    def _generate_chat_response(self, messages: List[Dict], images: List[str], model, data: Dict) -> Response:
        """Generate non-streaming chat response"""
        try:
            start_time = time.time()
            
            # The model's generate method now handles the conversation history
            result = model.generate(messages, images, **self._extract_generation_params(data))
            
            generation_time = time.time() - start_time
            
            response = {
                "model": data.get('model', 'gemma3'),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "message": {
                    "role": "assistant",
                    "content": result.get('text', '')
                },
                "done": True,
                "total_duration": int(generation_time * 1e9),
                "load_duration": 0,
                "prompt_eval_count": result.get('prompt_tokens', 0),
                "eval_count": result.get('completion_tokens', 0),
                "eval_duration": int(result.get('generation_time', 0) * 1e9)
            }
            return jsonify(response)
            
        except Exception as e:
            self.logger.error(f"Error generating chat response: {e}", exc_info=True)
            return jsonify(create_error_response("generation_error", str(e))), 500
    
    def _generate_response(self, result: Dict, data: Dict) -> Response:
        """Formats the result from a non-streaming /api/generate call"""
        response = {
            "model": data.get('model', 'gemma3'),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "response": result.get('text', ''),
            "done": True,
            "context": [],
            "total_duration": int(result.get('generation_time', 0) * 1e9),
            "load_duration": 0,
            "prompt_eval_count": result.get('prompt_tokens', 0),
            "eval_count": result.get('completion_tokens', 0),
            "eval_duration": int(result.get('generation_time', 0) * 1e9)
        }
        return jsonify(response)

    def _stream_chat_response(self, messages: List[Dict], images: List[str], model, data: Dict) -> Response:
        """Generate streaming chat response"""
        def generate_stream_content():
            try:
                final_chunk = {}
                # The model now handles history and streaming
                for chunk in model.generate_stream(messages, images, **self._extract_generation_params(data)):
                    if chunk.get('done'):
                        final_chunk = chunk
                        break
                    
                    response = {
                        "model": data.get('model', 'gemma3'),
                        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                        "message": {
                            "role": "assistant",
                            "content": chunk.get('text', '')
                        },
                        "done": False
                    }
                    yield f"data: {json.dumps(response)}\n\n"
                
                # Final done message
                done_response = {
                    "model": data.get('model', 'gemma3'),
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                    "done": True,
                    "total_duration": int(final_chunk.get('generation_time', 0) * 1e9),
                    "prompt_eval_count": final_chunk.get('prompt_tokens', 0),
                    "eval_count": final_chunk.get('completion_tokens', 0),
                }
                yield f"data: {json.dumps(done_response)}\n\n"
                
            except Exception as e:
                self.logger.error(f"Error during stream generation: {e}", exc_info=True)
                error_response = {"error": str(e)}
                yield f"data: {json.dumps(error_response)}\n\n"
        
        return Response(stream_with_context(generate_stream_content()), mimetype='application/x-ndjson')
    
    def _stream_generate_response(self, generator: Generator, data: Dict) -> Response:
        """Generate streaming response for /api/generate"""
        def generate_stream_content():
            try:
                final_chunk = {}
                for chunk in generator:
                    if chunk.get('done'):
                        final_chunk = chunk
                        break
                    
                    response = {
                        "model": data.get('model', 'gemma3'),
                        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                        "response": chunk.get('text', ''),
                        "done": False
                    }
                    yield f"{json.dumps(response)}\n"

                # Final done message
                done_response = {
                    "model": data.get('model', 'gemma3'),
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                    "response": "",
                    "done": True,
                    "context": [],
                    "total_duration": int(final_chunk.get('generation_time', 0) * 1e9),
                    "prompt_eval_count": final_chunk.get('prompt_tokens', 0),
                    "eval_count": final_chunk.get('completion_tokens', 0),
                }
                yield f"{json.dumps(done_response)}\n"

            except Exception as e:
                self.logger.error(f"Error during stream generation: {e}", exc_info=True)
                error_response = {"error": str(e)}
                yield f"{json.dumps(error_response)}\n"
        
        return Response(stream_with_context(generate_stream_content()), mimetype='application/x-ndjson')

    def _extract_generation_params(self, data: Dict) -> Dict[str, Any]:
        """Extract generation parameters from request data"""
        options = data.get('options', {})
        params = {}
        
        # Use values from 'options' if they exist, otherwise use model defaults
        if 'temperature' in options:
            params['temperature'] = options['temperature']
        if 'top_p' in options:
            params['top_p'] = options['top_p']
        if 'top_k' in options:
            params['top_k'] = options['top_k']
        if 'num_predict' in options:
            params['max_tokens'] = options['num_predict']
        if 'stop' in options:
            params['stop'] = options['stop']
        if 'repeat_penalty' in options:
            params['repeat_penalty'] = options['repeat_penalty']
        
        return params

"""
API Handlers for Gemma3 RKLLM
Handles different API formats including Ollama compatibility
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
        Handle Ollama-compatible chat request with multimodal support.
        This now correctly extracts the prompt and images.
        """
        try:
            # Validate request
            if not validate_request(data, ['model', 'messages']):
                return jsonify(create_error_response(
                    "invalid_request", 
                    "Missing required fields: model, messages"
                )), 400
            
            messages = data['messages']
            stream = data.get('stream', False)
            
            prompt = ""
            images = [] # This will hold base64 strings

            # --- KORRIGIERTE LOGIK ZUR PROMPT-EXTRAKTION ---
            # We only care about the content of the last message for a simple chat logic.
            # The rkllm library handles conversation history if configured.
            if messages:
                last_message = messages[-1]
                content = last_message.get('content', '')
                
                # Handle both string content and the multimodal list format
                if isinstance(content, str):
                    prompt = content
                elif isinstance(content, list):
                    for item in content:
                        if item.get('type') == 'text':
                            prompt = item.get('text', '')
                        elif item.get('type') == 'image_url':
                            # Extract the base64 data URL
                            img_data_url = item.get('image_url', {}).get('url', '')
                            if img_data_url:
                                images.append(img_data_url)
            # --- ENDE KORRIGIERTE LOGIK ---

            # Generate response
            if stream:
                # The model's generate_stream method will handle image processing
                return self._stream_chat_response(prompt, images, model, data)
            else:
                # The model's generate method will handle image processing
                return self._generate_chat_response(prompt, images, model, data)
                
        except Exception as e:
            self.logger.error(f"Error in chat request: {str(e)}", exc_info=True)
            return jsonify(create_error_response(
                "generation_error", 
                str(e)
            )), 500
    
    def handle_generate_request(self, data: Dict[str, Any], model) -> Response:
        """
        Handle Ollama-compatible generate request
        """
        try:
            if not validate_request(data, ['model', 'prompt']):
                return jsonify(create_error_response(
                    "invalid_request", 
                    "Missing required fields: model, prompt"
                )), 400
            
            prompt = data['prompt']
            images = data.get('images', []) # Expects a list of base64 strings
            stream = data.get('stream', False)
            
            # Generate response
            if stream:
                return self._stream_generate_response(prompt, images, model, data)
            else:
                return self._generate_response(prompt, images, model, data)
                
        except Exception as e:
            self.logger.error(f"Error in generate request: {str(e)}", exc_info=True)
            return jsonify(create_error_response(
                "generation_error", 
                str(e)
            )), 500
    
    def _generate_chat_response(self, prompt: str, images: List[str], model, data: Dict) -> Response:
        """Generate non-streaming chat response"""
        try:
            start_time = time.time()
            
            # The `generate` method in the model class now handles image processing
            result = model.generate(prompt, images, **self._extract_generation_params(data))
            
            generation_time = time.time() - start_time
            
            # Format Ollama-compatible response
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
            self.logger.error(f"Error generating chat response: {str(e)}", exc_info=True)
            return jsonify(create_error_response("generation_error", str(e))), 500
    
    def _generate_response(self, prompt: str, images: List[str], model, data: Dict) -> Response:
        """Generate non-streaming response"""
        try:
            start_time = time.time()
            
            result = model.generate(prompt, images, **self._extract_generation_params(data))
            
            generation_time = time.time() - start_time
            
            response = {
                "model": data.get('model', 'gemma3'),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "response": result.get('text', ''),
                "done": True,
                "context": [], # Context handling can be added later
                "total_duration": int(generation_time * 1e9),
                "load_duration": 0,
                "prompt_eval_count": result.get('prompt_tokens', 0),
                "eval_count": result.get('completion_tokens', 0),
                "eval_duration": int(result.get('generation_time', 0) * 1e9)
            }
            
            return jsonify(response)
            
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}", exc_info=True)
            return jsonify(create_error_response("generation_error", str(e))), 500
    
    def _stream_chat_response(self, prompt: str, images: List[str], model, data: Dict) -> Response:
        """Generate streaming chat response"""
        def generate_stream_content():
            try:
                final_chunk = {}
                for chunk in model.generate_stream(prompt, images, **self._extract_generation_params(data)):
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
        
        return Response(
            stream_with_context(generate_stream_content()),
            mimetype='application/x-ndjson'
        )
    
    def _stream_generate_response(self, prompt: str, images: List[str], model, data: Dict) -> Response:
        """Generate streaming response for /api/generate"""
        def generate_stream_content():
            try:
                final_chunk = {}
                for chunk in model.generate_stream(prompt, images, **self._extract_generation_params(data)):
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
        
        return Response(
            stream_with_context(generate_stream_content()),
            mimetype='application/x-ndjson'
        )

    def _extract_generation_params(self, data: Dict) -> Dict[str, Any]:
        """Extract generation parameters from request data"""
        options = data.get('options', {})
        params = {}
        
        params['temperature'] = options.get('temperature', self.config.getfloat('model', 'default_temperature', fallback=0.7))
        params['top_p'] = options.get('top_p', 0.9)
        params['top_k'] = options.get('top_k', 40)
        params['max_tokens'] = options.get('num_predict', self.config.getint('model', 'max_new_tokens', fallback=2048))
        params['stop'] = options.get('stop', [])
        params['repeat_penalty'] = options.get('repeat_penalty', 1.1)
        
        return params


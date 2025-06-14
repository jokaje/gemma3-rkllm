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
        Handle Ollama-compatible chat request with multimodal support
        
        Args:
            data: Request data
            model: Loaded model instance
            
        Returns:
            Flask Response
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
            
            # Extract text and images from messages
            prompt_parts = []
            images = []
            
            for message in messages:
                role = message.get('role', 'user')
                content = message.get('content', '')
                
                if isinstance(content, str):
                    prompt_parts.append(f"{role}: {content}")
                elif isinstance(content, list):
                    # Handle multimodal content
                    for item in content:
                        if item.get('type') == 'text':
                            prompt_parts.append(f"{role}: {item.get('text', '')}")
                        elif item.get('type') == 'image_url':
                            image_url = item.get('image_url', {}).get('url', '')
                            if image_url.startswith('data:'):
                                images.append(image_url)
            
            prompt = '\n'.join(prompt_parts)
            
            # Generate response
            if stream:
                return self._stream_chat_response(prompt, images, model, data)
            else:
                return self._generate_chat_response(prompt, images, model, data)
                
        except Exception as e:
            self.logger.error(f"Error in chat request: {str(e)}")
            return jsonify(create_error_response(
                "generation_error", 
                str(e)
            )), 500
    
    def handle_generate_request(self, data: Dict[str, Any], model) -> Response:
        """
        Handle Ollama-compatible generate request
        
        Args:
            data: Request data
            model: Loaded model instance
            
        Returns:
            Flask Response
        """
        try:
            # Validate request
            if not validate_request(data, ['model', 'prompt']):
                return jsonify(create_error_response(
                    "invalid_request", 
                    "Missing required fields: model, prompt"
                )), 400
            
            prompt = data['prompt']
            images = data.get('images', [])
            stream = data.get('stream', False)
            
            # Generate response
            if stream:
                return self._stream_generate_response(prompt, images, model, data)
            else:
                return self._generate_response(prompt, images, model, data)
                
        except Exception as e:
            self.logger.error(f"Error in generate request: {str(e)}")
            return jsonify(create_error_response(
                "generation_error", 
                str(e)
            )), 500
    
    def _generate_chat_response(self, prompt: str, images: List[str], model, data: Dict) -> Response:
        """Generate non-streaming chat response"""
        try:
            start_time = time.time()
            
            # Generate response
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
                "total_duration": int(generation_time * 1e9),  # nanoseconds
                "load_duration": 0,
                "prompt_eval_count": result.get('prompt_tokens', 0),
                "prompt_eval_duration": int(result.get('prompt_time', 0) * 1e9),
                "eval_count": result.get('completion_tokens', 0),
                "eval_duration": int(result.get('completion_time', 0) * 1e9)
            }
            
            return jsonify(response)
            
        except Exception as e:
            self.logger.error(f"Error generating chat response: {str(e)}")
            return jsonify(create_error_response("generation_error", str(e))), 500
    
    def _generate_response(self, prompt: str, images: List[str], model, data: Dict) -> Response:
        """Generate non-streaming response"""
        try:
            start_time = time.time()
            
            # Generate response
            result = model.generate(prompt, images, **self._extract_generation_params(data))
            
            generation_time = time.time() - start_time
            
            # Format Ollama-compatible response
            response = {
                "model": data.get('model', 'gemma3'),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "response": result.get('text', ''),
                "done": True,
                "context": result.get('context', []),
                "total_duration": int(generation_time * 1e9),
                "load_duration": 0,
                "prompt_eval_count": result.get('prompt_tokens', 0),
                "prompt_eval_duration": int(result.get('prompt_time', 0) * 1e9),
                "eval_count": result.get('completion_tokens', 0),
                "eval_duration": int(result.get('completion_time', 0) * 1e9)
            }
            
            return jsonify(response)
            
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return jsonify(create_error_response("generation_error", str(e))), 500
    
    def _stream_chat_response(self, prompt: str, images: List[str], model, data: Dict) -> Response:
        """Generate streaming chat response"""
        def generate():
            try:
                start_time = time.time()
                
                for chunk in model.generate_stream(prompt, images, **self._extract_generation_params(data)):
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
                
                # Final message
                final_response = {
                    "model": data.get('model', 'gemma3'),
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                    "message": {
                        "role": "assistant",
                        "content": ""
                    },
                    "done": True,
                    "total_duration": int((time.time() - start_time) * 1e9)
                }
                
                yield f"data: {json.dumps(final_response)}\n\n"
                
            except Exception as e:
                error_response = {
                    "error": str(e)
                }
                yield f"data: {json.dumps(error_response)}\n\n"
        
        return Response(
            stream_with_context(generate()),
            content_type='text/plain',
            headers={'Cache-Control': 'no-cache'}
        )
    
    def _stream_generate_response(self, prompt: str, images: List[str], model, data: Dict) -> Response:
        """Generate streaming response"""
        def generate():
            try:
                start_time = time.time()
                
                for chunk in model.generate_stream(prompt, images, **self._extract_generation_params(data)):
                    response = {
                        "model": data.get('model', 'gemma3'),
                        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                        "response": chunk.get('text', ''),
                        "done": False
                    }
                    
                    yield f"{json.dumps(response)}\n"
                
                # Final message
                final_response = {
                    "model": data.get('model', 'gemma3'),
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                    "response": "",
                    "done": True,
                    "total_duration": int((time.time() - start_time) * 1e9)
                }
                
                yield f"{json.dumps(final_response)}\n"
                
            except Exception as e:
                error_response = {
                    "error": str(e)
                }
                yield f"{json.dumps(error_response)}\n"
        
        return Response(
            stream_with_context(generate()),
            content_type='application/x-ndjson',
            headers={'Cache-Control': 'no-cache'}
        )
    
    def _extract_generation_params(self, data: Dict) -> Dict[str, Any]:
        """Extract generation parameters from request data"""
        params = {}
        
        # Temperature
        if 'temperature' in data:
            params['temperature'] = float(data['temperature'])
        
        # Top-p
        if 'top_p' in data:
            params['top_p'] = float(data['top_p'])
        
        # Top-k
        if 'top_k' in data:
            params['top_k'] = int(data['top_k'])
        
        # Max tokens
        if 'max_tokens' in data:
            params['max_tokens'] = int(data['max_tokens'])
        elif 'num_predict' in data:  # Ollama parameter
            params['max_tokens'] = int(data['num_predict'])
        
        # Stop sequences
        if 'stop' in data:
            params['stop'] = data['stop']
        
        # Repeat penalty
        if 'repeat_penalty' in data:
            params['repeat_penalty'] = float(data['repeat_penalty'])
        
        # Frequency penalty
        if 'frequency_penalty' in data:
            params['frequency_penalty'] = float(data['frequency_penalty'])
        
        # Presence penalty
        if 'presence_penalty' in data:
            params['presence_penalty'] = float(data['presence_penalty'])
        
        return params
    
    def handle_openai_chat_request(self, data: Dict[str, Any], model) -> Response:
        """
        Handle OpenAI-compatible chat request (if enabled)
        
        Args:
            data: Request data
            model: Loaded model instance
            
        Returns:
            Flask Response
        """
        if not self.openai_enabled:
            return jsonify(create_error_response(
                "not_supported", 
                "OpenAI API compatibility is not enabled"
            )), 501
        
        # Convert OpenAI format to internal format
        # This would need to be implemented based on OpenAI API spec
        # For now, return not implemented
        return jsonify(create_error_response(
            "not_implemented", 
            "OpenAI API compatibility not yet implemented"
        )), 501
    
    def get_api_info(self) -> Dict[str, Any]:
        """Get API information and capabilities"""
        return {
            "name": "Gemma3 RKLLM API",
            "version": "1.0.0",
            "capabilities": {
                "multimodal": True,
                "streaming": True,
                "ollama_compatible": self.ollama_enabled,
                "openai_compatible": self.openai_enabled
            },
            "supported_formats": {
                "images": ["jpeg", "png", "webp", "bmp"],
                "text": ["plain", "markdown"]
            },
            "limits": {
                "max_context_length": self.config.getint('model', 'max_context_length', fallback=128000),
                "max_image_size": self.config.getint('multimodal', 'max_image_size', fallback=2048),
                "max_request_size": self.config.get('security', 'max_request_size', fallback='50MB')
            }
        }


"""
Gemma3 Model Integration
Main model class that combines RKLLM runtime, NPU optimization, and multimodal processing
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Generator
import configparser
import json

from .rkllm_runtime import RKLLMRuntime
from .npu_optimizer import NPUOptimizer
from .image_processor import ImageProcessor
from .utils import generate_unique_id, format_duration, measure_execution_time
from .logger import PerformanceLogger, LogExecutionTime


class Gemma3Model:
    """Main Gemma3 model class with multimodal support"""
    
    def __init__(self, model_name: str, config: configparser.ConfigParser):
        self.model_name = model_name
        self.config = config
        self.logger = logging.getLogger("gemma3-rkllm.model")
        self.perf_logger = PerformanceLogger(self.logger)
        
        # Model configuration
        self.models_dir = Path(config.get('model', 'models_dir', fallback='./models'))
        self.max_context_length = config.getint('model', 'max_context_length', fallback=128000)
        self.default_temperature = config.getfloat('model', 'default_temperature', fallback=0.7)
        self.max_new_tokens = config.getint('model', 'max_new_tokens', fallback=2048)
        
        # Initialize components
        self.rkllm_runtime = None
        self.npu_optimizer = NPUOptimizer(config)
        self.image_processor = ImageProcessor(config)
        
        # Model state
        self.is_loaded = False
        self.model_path = None
        self.model_config = {}
        self.tokenizer = None
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the Gemma3 model"""
        try:
            with LogExecutionTime(self.logger, f"Loading model {self.model_name}"):
                # Find model directory
                model_dir = self.models_dir / self.model_name
                if not model_dir.exists():
                    raise FileNotFoundError(f"Model directory not found: {model_dir}")
                
                # Load model configuration
                self._load_model_config(model_dir)
                
                # Find .rkllm file
                rkllm_files = list(model_dir.glob("*.rkllm"))
                if not rkllm_files:
                    raise FileNotFoundError(f"No .rkllm file found in {model_dir}")
                
                self.model_path = str(rkllm_files[0])
                self.logger.info(f"Found model file: {self.model_path}")
                
                # Apply NPU optimizations
                if not self.npu_optimizer.apply_optimization():
                    self.logger.warning("NPU optimization failed, continuing without optimization")
                
                # Initialize RKLLM runtime
                self._initialize_rkllm()
                
                # Load tokenizer if available
                self._load_tokenizer(model_dir)
                
                self.is_loaded = True
                self.logger.info(f"Model {self.model_name} loaded successfully")
                
        except Exception as e:
            self.logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    def _load_model_config(self, model_dir: Path):
        """Load model configuration from Modelfile"""
        try:
            modelfile_path = model_dir / "Modelfile"
            if modelfile_path.exists():
                with open(modelfile_path, 'r') as f:
                    content = f.read()
                
                # Parse Modelfile format
                config = {}
                for line in content.split('\n'):
                    line = line.strip()
                    if '=' in line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"')
                        config[key] = value
                
                self.model_config = config
                self.logger.debug(f"Loaded model config: {config}")
            else:
                self.logger.warning(f"No Modelfile found in {model_dir}")
                self.model_config = {}
                
        except Exception as e:
            self.logger.error(f"Error loading model config: {e}")
            self.model_config = {}
    
    def _initialize_rkllm(self):
        """Initialize RKLLM runtime"""
        try:
            # Create RKLLM runtime
            lib_path = self.model_config.get('RKLLM_LIB_PATH', './lib/librkllmrt.so')
            self.rkllm_runtime = RKLLMRuntime(lib_path)
            
            # Prepare RKLLM configuration
            rkllm_config = {
                'max_context_len': self.max_context_length,
                'max_new_tokens': self.max_new_tokens,
                'temperature': self.default_temperature,
                'top_k': 1,
                'top_p': 0.9,
                'repeat_penalty': 1.1,
                'frequency_penalty': 0.0,
                'presence_penalty': 0.0,
                'skip_special_token': True,
                'is_async': False,
                # Multimodal tokens for Gemma3
                'img_start': '<start_of_image>',
                'img_end': '<end_of_image>',
                'img_content': '',
                # NPU optimization parameters
                'base_domain_id': 0,
                'embed_flash': 1,
                'enabled_cpus_num': 8,  # RK3588 has 8 cores
                'enabled_cpus_mask': 0xFF
            }
            
            # Override with model-specific config
            if 'TEMPERATURE' in self.model_config:
                rkllm_config['temperature'] = float(self.model_config['TEMPERATURE'])
            
            # Initialize RKLLM
            if not self.rkllm_runtime.initialize(self.model_path, rkllm_config):
                raise RuntimeError("Failed to initialize RKLLM runtime")
                
        except Exception as e:
            self.logger.error(f"Error initializing RKLLM: {e}")
            raise
    
    def _load_tokenizer(self, model_dir: Path):
        """Load tokenizer if available"""
        try:
            # Look for tokenizer files
            tokenizer_files = [
                'tokenizer.json',
                'tokenizer_config.json',
                'vocab.txt'
            ]
            
            tokenizer_found = False
            for filename in tokenizer_files:
                if (model_dir / filename).exists():
                    tokenizer_found = True
                    break
            
            if tokenizer_found:
                try:
                    from transformers import AutoTokenizer
                    self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
                    self.logger.info("Tokenizer loaded successfully")
                except Exception as e:
                    self.logger.warning(f"Could not load tokenizer: {e}")
                    self.tokenizer = None
            else:
                self.logger.info("No tokenizer files found, using RKLLM internal tokenization")
                self.tokenizer = None
                
        except Exception as e:
            self.logger.warning(f"Error loading tokenizer: {e}")
            self.tokenizer = None
    
    @measure_execution_time
    def generate(self, prompt: str, images: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Dict[str, Any]:
        """
        Generate text response with optional image input
        
        Args:
            prompt: Text prompt
            images: Optional list of processed image dictionaries
            **kwargs: Generation parameters
            
        Returns:
            Generation result dictionary
        """
        try:
            if not self.is_loaded:
                raise RuntimeError("Model not loaded")
            
            start_time = time.time()
            
            # Process multimodal input
            if images:
                prompt = self._create_multimodal_prompt(prompt, images)
            
            # Extract generation parameters
            generation_params = self._extract_generation_params(kwargs)
            
            # Tokenize if tokenizer is available
            if self.tokenizer:
                tokens = self.tokenizer.encode(prompt)
                input_mode = "token"
                input_data = tokens
            else:
                input_mode = "prompt"
                input_data = prompt
            
            # Run inference
            if not self.rkllm_runtime.run_inference(input_data, input_mode):
                raise RuntimeError("Inference failed")
            
            # Get response
            response_text = self.rkllm_runtime.get_response(timeout=generation_params.get('timeout', 30.0))
            
            generation_time = time.time() - start_time
            
            # Calculate metrics
            prompt_tokens = len(tokens) if self.tokenizer else len(prompt.split())
            completion_tokens = len(response_text.split())
            
            # Log performance
            self.perf_logger.log_inference_time(
                self.model_name, 
                prompt_tokens, 
                generation_time, 
                completion_tokens
            )
            
            result = {
                'text': response_text,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': prompt_tokens + completion_tokens,
                'generation_time': generation_time,
                'tokens_per_second': completion_tokens / generation_time if generation_time > 0 else 0,
                'model': self.model_name,
                'multimodal': bool(images),
                'finish_reason': 'stop'
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in generation: {e}")
            raise
    
    def generate_stream(self, prompt: str, images: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Generator[Dict[str, Any], None, None]:
        """
        Generate streaming text response
        
        Args:
            prompt: Text prompt
            images: Optional list of processed image dictionaries
            **kwargs: Generation parameters
            
        Yields:
            Generation result chunks
        """
        try:
            if not self.is_loaded:
                raise RuntimeError("Model not loaded")
            
            start_time = time.time()
            
            # Process multimodal input
            if images:
                prompt = self._create_multimodal_prompt(prompt, images)
            
            # Extract generation parameters
            generation_params = self._extract_generation_params(kwargs)
            
            # Tokenize if tokenizer is available
            if self.tokenizer:
                tokens = self.tokenizer.encode(prompt)
                input_mode = "token"
                input_data = tokens
            else:
                input_mode = "prompt"
                input_data = prompt
            
            # Run inference
            if not self.rkllm_runtime.run_inference(input_data, input_mode):
                raise RuntimeError("Inference failed")
            
            # Stream response
            full_response = ""
            for chunk in self.rkllm_runtime.get_response_stream(timeout=generation_params.get('timeout', 30.0)):
                full_response += chunk
                
                yield {
                    'text': chunk,
                    'full_text': full_response,
                    'model': self.model_name,
                    'multimodal': bool(images),
                    'done': False
                }
            
            # Final chunk
            generation_time = time.time() - start_time
            prompt_tokens = len(tokens) if self.tokenizer else len(prompt.split())
            completion_tokens = len(full_response.split())
            
            yield {
                'text': '',
                'full_text': full_response,
                'model': self.model_name,
                'multimodal': bool(images),
                'done': True,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'generation_time': generation_time,
                'tokens_per_second': completion_tokens / generation_time if generation_time > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error in streaming generation: {e}")
            yield {
                'error': str(e),
                'model': self.model_name,
                'done': True
            }
    
    def _create_multimodal_prompt(self, text_prompt: str, images: List[Dict[str, Any]]) -> str:
        """
        Create multimodal prompt combining text and images
        
        Args:
            text_prompt: Text part of the prompt
            images: List of processed image dictionaries
            
        Returns:
            Combined multimodal prompt
        """
        try:
            # Gemma3 multimodal format
            prompt_parts = []
            
            for image in images:
                image_prompt = self.image_processor.create_image_prompt("", image)
                prompt_parts.append(image_prompt)
            
            # Add text prompt
            prompt_parts.append(text_prompt)
            
            combined_prompt = "\n".join(prompt_parts)
            
            self.logger.debug(f"Created multimodal prompt with {len(images)} images")
            return combined_prompt
            
        except Exception as e:
            self.logger.error(f"Error creating multimodal prompt: {e}")
            return text_prompt
    
    def _extract_generation_params(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and validate generation parameters"""
        params = {
            'temperature': kwargs.get('temperature', self.default_temperature),
            'top_p': kwargs.get('top_p', 0.9),
            'top_k': kwargs.get('top_k', 1),
            'max_tokens': kwargs.get('max_tokens', self.max_new_tokens),
            'stop': kwargs.get('stop', []),
            'repeat_penalty': kwargs.get('repeat_penalty', 1.1),
            'frequency_penalty': kwargs.get('frequency_penalty', 0.0),
            'presence_penalty': kwargs.get('presence_penalty', 0.0),
            'timeout': kwargs.get('timeout', 30.0)
        }
        
        # Validate ranges
        params['temperature'] = max(0.0, min(2.0, params['temperature']))
        params['top_p'] = max(0.0, min(1.0, params['top_p']))
        params['top_k'] = max(1, params['top_k'])
        params['max_tokens'] = max(1, min(self.max_new_tokens, params['max_tokens']))
        
        return params
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information
        
        Returns:
            Dictionary with model information
        """
        return {
            'name': self.model_name,
            'path': self.model_path,
            'loaded': self.is_loaded,
            'multimodal': True,
            'max_context_length': self.max_context_length,
            'max_new_tokens': self.max_new_tokens,
            'default_temperature': self.default_temperature,
            'config': self.model_config,
            'tokenizer_available': self.tokenizer is not None,
            'npu_optimized': self.npu_optimizer.optimization_applied,
            'npu_status': self.npu_optimizer.get_optimization_status()
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics
        
        Returns:
            Dictionary with performance statistics
        """
        return {
            'model_name': self.model_name,
            'npu_optimization': self.npu_optimizer.get_optimization_status(),
            'image_processing': self.image_processor.get_processing_stats(),
            'memory_usage': self._get_memory_usage()
        }
    
    def _get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'rss_mb': memory_info.rss / (1024 * 1024),
                'vms_mb': memory_info.vms / (1024 * 1024),
                'percent': process.memory_percent()
            }
        except Exception:
            return {}
    
    def release(self):
        """Release model resources"""
        try:
            if self.rkllm_runtime:
                self.rkllm_runtime.release()
                self.rkllm_runtime = None
            
            if self.npu_optimizer:
                self.npu_optimizer.restore_original_settings()
            
            if self.image_processor:
                self.image_processor.clear_cache()
            
            self.is_loaded = False
            self.logger.info(f"Model {self.model_name} released successfully")
            
        except Exception as e:
            self.logger.error(f"Error releasing model: {e}")
    
    def __del__(self):
        """Destructor to ensure resources are released"""
        self.release()


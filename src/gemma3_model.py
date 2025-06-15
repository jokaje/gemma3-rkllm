"""
Gemma3 Model Integration
Main model class that combines RKLLM runtime, NPU optimization, and multimodal processing
"""

import os
import time
import logging
import threading
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
        
        self.models_dir = Path(config.get('model', 'models_dir', fallback='./models'))
        self.max_context_length = config.getint('model', 'max_context_length', fallback=16384)
        self.default_temperature = config.getfloat('model', 'default_temperature', fallback=0.7)
        self.max_new_tokens = config.getint('model', 'max_new_tokens', fallback=2048)
        
        self.rkllm_runtime, self.npu_optimizer, self.image_processor = None, NPUOptimizer(config), ImageProcessor(config)
        self.is_loaded, self.model_path, self.model_config, self.tokenizer = False, None, {}, None
        
        self._load_model()
    
    def _load_model(self):
        """Load the Gemma3 model"""
        try:
            with LogExecutionTime(self.logger, f"Loading model {self.model_name}"):
                model_dir = self.models_dir / self.model_name
                if not model_dir.exists(): raise FileNotFoundError(f"Model directory not found: {model_dir}")
                
                self._load_model_config(model_dir)
                
                rkllm_files = list(model_dir.glob("*.rkllm"))
                if not rkllm_files: raise FileNotFoundError(f"No .rkllm file found in {model_dir}")
                
                self.model_path = str(rkllm_files[0])
                self.logger.info(f"Found model file: {self.model_path}")
                
                if not self.npu_optimizer.apply_optimization():
                    self.logger.warning("NPU optimization failed")
                
                self._initialize_rkllm()
                self._load_tokenizer(model_dir)
                
                self.is_loaded = True
                self.logger.info(f"Model {self.model_name} loaded successfully")
                
        except Exception as e:
            self.logger.error(f"Failed to load model {self.model_name}: {e}")
            self.release()
            raise
    
    def _load_model_config(self, model_dir: Path):
        """Load model configuration from Modelfile"""
        try:
            modelfile_path = model_dir / "Modelfile"
            if modelfile_path.exists():
                with open(modelfile_path, 'r') as f:
                    for line in f:
                        if '=' in line and not line.strip().startswith('#'):
                            key, value = line.split('=', 1)
                            self.model_config[key.strip().upper()] = value.strip().strip('"')
                self.logger.debug(f"Loaded model config: {self.model_config}")
            else:
                self.logger.warning(f"No Modelfile found in {model_dir}")
                
        except Exception as e:
            self.logger.error(f"Error loading model config: {e}")

    def _initialize_rkllm(self):
        """Initialize RKLLM runtime and set the chat template."""
        try:
            lib_path = self.model_config.get('RKLLM_LIB_PATH', './lib/librkllmrt.so')
            self.rkllm_runtime = RKLLMRuntime(lib_path)
            
            rkllm_config = {
                'max_context_len': self.max_context_length,
                'max_new_tokens': self.max_new_tokens,
                'temperature': self.model_config.get('TEMPERATURE', self.default_temperature)
            }
            
            if not self.rkllm_runtime.initialize(self.model_path, rkllm_config):
                raise RuntimeError("Failed to initialize RKLLM runtime")

            # --- KORREKTE TEMPLATE-INITIALISIERUNG ---
            # Set the chat template using the official library function
            system_prompt = self.model_config.get('SYSTEM', '') # System prompt can be empty
            user_prefix = self.model_config.get('PROMPT_PREFIX', '<start_of_turn>user\n')
            user_postfix = self.model_config.get('PROMPT_POSTFIX', '<end_of_turn>\n<start_of_turn>model\n')
            
            self.rkllm_runtime.set_chat_template(system_prompt, user_prefix, user_postfix)
            # --- ENDE KORREKTE TEMPLATE-INITIALISIERUNG ---

        except Exception as e:
            self.logger.error(f"Error during RKLLM initialization: {e}", exc_info=True)
            raise

    def _load_tokenizer(self, model_dir: Path):
        """Load tokenizer if available"""
        try:
            tokenizer_path = self.model_config.get('TOKENIZER', str(model_dir))
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            self.logger.info(f"Tokenizer loaded successfully from {tokenizer_path}")
        except Exception as e:
            self.logger.warning(f"Could not load HuggingFace tokenizer. Error: {e}")
            self.tokenizer = None

    def _execute_generation(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Helper to run threaded generation and process results."""
        start_time = time.time()
        generation_params = self._extract_generation_params(kwargs)
        
        # We pass the raw user prompt. The library now handles all templating.
        self.logger.info(f"Passing raw prompt to RKLLM runtime: {prompt[:200]}...")
        model_thread = threading.Thread(target=self.rkllm_runtime.run_inference, args=(prompt,))
        model_thread.start()
        
        response_text = self.rkllm_runtime.get_response(timeout=generation_params.get('timeout', 60.0))
        model_thread.join()
        
        generation_time = time.time() - start_time
        prompt_tokens = len(self.tokenizer.encode(prompt)) if self.tokenizer else len(prompt.split())
        completion_tokens = len(self.tokenizer.encode(response_text)) if self.tokenizer else len(response_text.split())
        
        self.perf_logger.log_inference_time(self.model_name, prompt_tokens, generation_time, completion_tokens)
        
        return {
            'text': response_text, 'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens,
            'total_tokens': prompt_tokens + completion_tokens, 'generation_time': generation_time,
            'tokens_per_second': completion_tokens / generation_time if generation_time > 0 else 0
        }

    @measure_execution_time
    def generate(self, prompt: str, images: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Dict[str, Any]:
        """Generate text response with optional image input"""
        if not self.is_loaded: raise RuntimeError("Model not loaded")
        if images: prompt = self._create_multimodal_prompt(prompt, images)
        
        result = self._execute_generation(prompt, **kwargs)
        result.update({'model': self.model_name, 'multimodal': bool(images), 'finish_reason': 'stop'})
        return result
            
    def generate_stream(self, prompt: str, images: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Generator[Dict[str, Any], None, None]:
        """Generate streaming text response"""
        if not self.is_loaded: raise RuntimeError("Model not loaded")
        if images: prompt = self._create_multimodal_prompt(prompt, images)
        
        generation_params = self._extract_generation_params(kwargs)
        
        self.logger.info(f"Passing raw prompt to RKLLM for streaming: {prompt[:200]}...")
        model_thread = threading.Thread(target=self.rkllm_runtime.run_inference, args=(prompt,))
        model_thread.start()
        
        full_response = ""
        for chunk in self.rkllm_runtime.get_response_stream(timeout=generation_params.get('timeout', 60.0)):
            full_response += chunk
            yield {'text': chunk, 'full_text': full_response, 'model': self.model_name, 'multimodal': bool(images), 'done': False}
        
        model_thread.join()
        
        prompt_tokens = len(self.tokenizer.encode(prompt)) if self.tokenizer else len(prompt.split())
        completion_tokens = len(self.tokenizer.encode(full_response)) if self.tokenizer else len(full_response.split())
        
        yield {
            'text': '', 'full_text': full_response, 'model': self.model_name,
            'multimodal': bool(images), 'done': True, 'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens
        }
            
    def _create_multimodal_prompt(self, text_prompt: str, images: List[Dict[str, Any]]) -> str:
        """Create multimodal prompt combining text and images"""
        try:
            prompt_parts = [self.image_processor.create_image_prompt("", image) for image in images]
            prompt_parts.append(text_prompt)
            return "\n".join(prompt_parts)
        except Exception as e:
            self.logger.error(f"Error creating multimodal prompt: {e}")
            return text_prompt
    
    def _extract_generation_params(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and validate generation parameters"""
        return {
            'temperature': kwargs.get('temperature', self.default_temperature),
            'timeout': kwargs.get('timeout', 60.0)
        }
    
    def release(self):
        """Release model resources"""
        try:
            if self.rkllm_runtime: self.rkllm_runtime.release()
            if self.npu_optimizer: self.npu_optimizer.restore_original_settings()
            if self.image_processor: self.image_processor.clear_cache()
            self.is_loaded = False
            self.logger.info(f"Model {self.model_name} released successfully")
        except Exception as e:
            self.logger.error(f"Error releasing model: {e}")
    
    def __del__(self):
        self.release()


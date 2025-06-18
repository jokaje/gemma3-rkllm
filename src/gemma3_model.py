"""
Gemma3 Model Integration
This version is adapted to use the RKLLMRuntime class directly.
It includes a robust Modelfile parser and handles chat history formatting.
"""
import os
import time
import logging
import threading
import re
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Generator
import configparser

# Import the RKLLMRuntime class directly
from .rkllm_runtime import RKLLMRuntime
from .npu_optimizer import NPUOptimizer
from .image_processor import ImageProcessor
from .utils import measure_execution_time
from .logger import PerformanceLogger, LogExecutionTime

class Gemma3Model:
    """Main Gemma3 model class using the RKLLMRuntime directly"""
    
    def __init__(self, model_name: str, config: configparser.ConfigParser):
        self.model_name = model_name
        self.config = config
        self.logger = logging.getLogger("gemma3-rkllm.model")
        self.perf_logger = PerformanceLogger(self.logger)
        
        self.models_dir = Path(config.get('model', 'models_dir', fallback='./models'))
        self.max_context_length = config.getint('model', 'max_context_length', fallback=16384)
        
        self.npu_optimizer = NPUOptimizer(config)
        self.image_processor = ImageProcessor(config)
        
        self.rkllm_runtime = None
        self.is_loaded = False
        self.model_path = None
        self.model_config = {}
        
        # Chat template components will be loaded from Modelfile
        self.system_prompt = ""
        self.user_prefix = "<start_of_turn>user\n"
        self.assistant_prefix = "<start_of_turn>model\n"
        self.user_postfix = "<end_of_turn>\n"
        self.assistant_postfix = "<end_of_turn>\n"
        
        self._load_model()
    
    def _load_model(self):
        """Load the Gemma3 model"""
        try:
            with LogExecutionTime(self.logger, f"Loading model {self.model_name}"):
                model_dir = self.models_dir / self.model_name
                if not model_dir.exists():
                    raise FileNotFoundError(f"Model directory not found: {model_dir}")
                
                # This now uses the robust parser
                self._load_model_config(model_dir)
                
                rkllm_files = list(model_dir.glob("*.rkllm"))
                if not rkllm_files:
                    # Try to find the model name specified in FROM
                    from_model = self.model_config.get("FROM")
                    if from_model:
                        rkllm_files = list(model_dir.glob(from_model))

                if not rkllm_files:
                     raise FileNotFoundError(f"No .rkllm file found in {model_dir} matching the Modelfile's FROM directive.")

                self.model_path = str(rkllm_files[0])
                
                if not self.npu_optimizer.apply_optimization():
                    self.logger.warning("NPU optimization failed")
                
                self._initialize_rkllm()
                
                self.is_loaded = True
                self.logger.info(f"Model {self.model_name} loaded successfully")
                
        except Exception as e:
            self.logger.error(f"Failed to load model {self.model_name}: {e}", exc_info=True)
            self.release()
            raise
    
    def _load_model_config(self, model_dir: Path):
        """
        Load model configuration from a structured Modelfile that can handle
        multiline strings and PARAMETER/TEMPLATE directives.
        """
        modelfile_path = model_dir / "Modelfile"
        self.model_config = {}  # Reset config for clean loading

        if not modelfile_path.exists():
            self.logger.warning(f"Modelfile not found in {model_dir}")
            return

        self.logger.info(f"Parsing Modelfile: {modelfile_path}")
        with open(modelfile_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 1. Parse multiline SYSTEM prompt using """
        system_match = re.search(r'SYSTEM\s*"""(.*?)"""', content, re.DOTALL)
        if system_match:
            self.model_config['SYSTEM'] = system_match.group(1).strip()

        # 2. Parse all other simple KEY=VALUE assignments
        simple_params = re.findall(r'^\b([A-Z_]+)\b\s*=\s*(.+)$', content, re.MULTILINE)
        for key, value in simple_params:
            self.model_config[key.strip().upper()] = value.strip().strip('"')

        # 3. Parse PARAMETER directives (e.g., PARAMETER stop "...")
        parameters = re.findall(r'^PARAMETER\s+([^\s]+)\s+(.*)$', content, re.MULTILINE)
        if parameters:
            self.model_config['PARAMETERS'] = {}
            for key, value in parameters:
                clean_value = value.strip().strip('"')
                # Store multiple stop tokens in a list
                if key in self.model_config['PARAMETERS']:
                    if not isinstance(self.model_config['PARAMETERS'][key], list):
                        self.model_config['PARAMETERS'][key] = [self.model_config['PARAMETERS'][key]]
                    self.model_config['PARAMETERS'][key].append(clean_value)
                else:
                    self.model_config['PARAMETERS'][key] = clean_value

        # 4. Parse TEMPLATE block
        template_match = re.search(r'TEMPLATE\s*"""(.*?)"""', content, re.DOTALL)
        if template_match:
            self.model_config['TEMPLATE'] = template_match.group(1).strip()

        self.logger.info("Modelfile parsed successfully with structured parser.")
        self.logger.debug(f"Loaded model config: {json.dumps(self.model_config, indent=2)}")

    def _initialize_rkllm(self):
        """Initialize RKLLM runtime and set model parameters from parsed config."""
        
        # Set parameters from the parsed Modelfile, with fallbacks to the global config
        rkllm_config = {
            'max_context_len': self.max_context_length,
            'max_new_tokens': int(self.model_config.get('MAX_NEW_TOKENS', self.config.getint('model', 'max_new_tokens', fallback=2048))),
            'temperature': float(self.model_config.get('TEMPERATURE', self.config.getfloat('model', 'default_temperature', fallback=0.7))),
            'top_k': int(self.model_config.get('TOP_K', 1)),
            'top_p': float(self.model_config.get('TOP_P', 0.9)),
            'repeat_penalty': float(self.model_config.get('REPEAT_PENALTY', 1.1)),
            'skip_special_token': True,
        }
        
        # Initialize RKLLM runtime
        self.rkllm_runtime = RKLLMRuntime()
        success = self.rkllm_runtime.initialize(self.model_path, rkllm_config)
        
        if not success:
            raise RuntimeError("Failed to initialize RKLLM runtime")
        
        # Set chat template components from the parsed config
        self.system_prompt = self.model_config.get('SYSTEM', '')

        # You can add logic here to parse the TEMPLATE block if needed,
        # but for now, the defaults are robust.
        # e.g., self.user_prefix = ...
        
        self.logger.info("RKLLM runtime initialized with parameters from Modelfile.")

    def _format_chat_history(self, messages: List[Dict[str, Any]], images: List[str]) -> str:
        """Build a single prompt string from a list of messages, incorporating the system prompt."""
        
        full_prompt_parts = []
        image_placeholder = "\n[Image Content]\n" # Placeholder for where image context is inserted

        # 1. Add System Prompt
        if self.system_prompt:
            full_prompt_parts.append(self.system_prompt)
        
        # 2. Build conversation history
        for message in messages:
            role = message.get("role")
            content = message.get("content", "")

            if role == "user":
                full_prompt_parts.append(self.user_prefix)
                # Handle multimodal content
                if isinstance(content, list):
                    text_parts = []
                    for item in content:
                        if item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                    full_prompt_parts.append(" ".join(text_parts))
                    # Add a placeholder for each image provided with this user message
                    if images:
                        full_prompt_parts.append(image_placeholder * len(images))
                elif isinstance(content, str):
                    full_prompt_parts.append(content)
                full_prompt_parts.append(self.user_postfix)

            elif role == "assistant":
                full_prompt_parts.append(self.assistant_prefix)
                if isinstance(content, str):
                    full_prompt_parts.append(content)
                full_prompt_parts.append(self.assistant_postfix)
        
        # 3. Add the model's turn prefix to signal it to start generating
        full_prompt_parts.append(self.assistant_prefix)

        final_prompt = "".join(full_prompt_parts)
        self.logger.debug(f"Formatted final prompt for RKLLM: {final_prompt[:500]}...") # Log first 500 chars
        return final_prompt

    def generate(self, messages: List[Dict[str, Any]], images: List[str] = None, **kwargs) -> Dict[str, Any]:
        """Generate text response from a list of messages, with optional image support."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        start_time = time.time()
        
        # Format the full prompt using the conversation history
        formatted_prompt = self._format_chat_history(messages, images or [])
        
        # TODO: Add actual multimodal input to rkllm_run when the library supports it.
        # For now, images are used to create a placeholder in the prompt.

        # Run inference
        success = self.rkllm_runtime.run_inference(formatted_prompt)
        if not success:
            raise RuntimeError("Inference failed")
        
        # Get response
        response_text = self.rkllm_runtime.get_response(timeout=kwargs.get('timeout', 120.0))
        generation_time = time.time() - start_time
        
        return {
            'text': response_text,
            'generation_time': generation_time,
            'model': self.model_name,
            'images_processed': len(images) if images else 0
        }

    def generate_stream(self, messages: List[Dict[str, Any]], images: List[str] = None, **kwargs) -> Generator[Dict[str, Any], None, None]:
        """Generate streaming text response from a list of messages."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        formatted_prompt = self._format_chat_history(messages, images or [])

        # Run inference
        success = self.rkllm_runtime.run_inference(formatted_prompt)
        if not success:
            raise RuntimeError("Inference failed")
        
        # Stream response
        for chunk in self.rkllm_runtime.get_response_stream(timeout=kwargs.get('timeout', 120.0)):
            yield {
                'text': chunk, 
                'done': False, 
                'model': self.model_name,
                'images_processed': len(images) if images else 0
            }
        
        yield {
            'text': '', 
            'done': True, 
            'model': self.model_name,
            'images_processed': len(images) if images else 0
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'name': self.model_name,
            'path': self.model_path,
            'is_loaded': self.is_loaded,
            'config': self.model_config,
            'max_context_length': self.max_context_length,
        }

    def release(self):
        """Release model resources"""
        try:
            if hasattr(self, 'rkllm_runtime') and self.rkllm_runtime:
                self.rkllm_runtime.release()
                self.rkllm_runtime = None
            
            if hasattr(self, 'npu_optimizer') and self.npu_optimizer:
                self.npu_optimizer.restore_original_settings()
            
            self.is_loaded = False
            self.logger.info(f"Model {self.model_name} released successfully")
            
        except Exception as e:
            self.logger.error(f"Error releasing model: {e}")

    def __del__(self):
        """Destructor to ensure resources are released"""
        self.release()

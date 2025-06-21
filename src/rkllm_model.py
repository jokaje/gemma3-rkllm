"""
RKLLM Model Integration
A generic model handler that loads its configuration and chat template 
from a Modelfile, making it compatible with various model architectures.
"""
import os
import time
import logging
import re
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Generator
import configparser

from .rkllm_runtime import RKLLMRuntime
from .npu_optimizer import NPUOptimizer
from .image_processor import ImageProcessor
from .logger import PerformanceLogger, LogExecutionTime

class RKLLMModel:
    """Generic model class using RKLLMRuntime, configured by a Modelfile."""
    
    def __init__(self, model_name: str, config: configparser.ConfigParser):
        self.model_name = model_name
        self.global_config = config
        self.logger = logging.getLogger("gemma3-rkllm.model")
        self.perf_logger = PerformanceLogger(self.logger)
        
        self.models_dir = Path(config.get('model', 'models_dir', fallback='./models'))
        
        self.npu_optimizer = NPUOptimizer(config)
        self.image_processor = ImageProcessor(config)
        
        self.rkllm_runtime: Optional[RKLLMRuntime] = None
        self.is_loaded = False
        self.model_path: Optional[str] = None
        self.model_config: Dict[str, Any] = {} # Parsed Modelfile
        
        self._load_model()
    
    def _load_model(self):
        """Load the model and its configuration."""
        try:
            with LogExecutionTime(self.logger, f"Loading model '{self.model_name}'"):
                model_dir = self.models_dir / self.model_name
                if not model_dir.exists():
                    raise FileNotFoundError(f"Model directory not found: {model_dir}")
                
                self._parse_modelfile(model_dir)
                
                model_filename = self.model_config.get("FROM")
                if not model_filename:
                    raise ValueError("Modelfile must contain a 'FROM' directive specifying the .rkllm file.")

                self.model_path = str(model_dir / model_filename)
                if not os.path.exists(self.model_path):
                    raise FileNotFoundError(f"RKLLM file not found: {self.model_path}")

                self.npu_optimizer.apply_optimization()
                self._initialize_rkllm()
                
                self.is_loaded = True
                self.logger.info(f"Model '{self.model_name}' loaded successfully.")
                
        except Exception as e:
            self.logger.error(f"Failed to load model '{self.model_name}': {e}", exc_info=True)
            self.release()
            raise

    def _parse_modelfile(self, model_dir: Path):
        """
        Parse the Modelfile using a robust line-by-line method for parameters
        and regex for multiline blocks (SYSTEM, TEMPLATE).
        """
        modelfile_path = model_dir / "Modelfile"
        if not modelfile_path.exists():
            modelfile_path = model_dir / "Modelfile.ini" # Fallback for old name
            if not modelfile_path.exists():
                 raise FileNotFoundError(f"Modelfile or Modelfile.ini not found in {model_dir}")

        self.logger.info(f"Parsing Modelfile: {modelfile_path}")
        with open(modelfile_path, 'r', encoding='utf-8') as f:
            content = f.read()

        self.model_config['PARAMETERS'] = {}
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                continue
            
            command = parts[0].upper()
            value = parts[1]

            if command == 'FROM':
                self.model_config['FROM'] = value.strip('"')
            elif command == 'PARAMETER':
                param_parts = value.split(maxsplit=1)
                if len(param_parts) == 2:
                    key, val = param_parts
                    key = key.lower()
                    val = val.strip('"')
                    if key == 'stop':
                        if 'stop' not in self.model_config['PARAMETERS']:
                            self.model_config['PARAMETERS']['stop'] = []
                        self.model_config['PARAMETERS']['stop'].append(val)
                    else:
                        self.model_config['PARAMETERS'][key] = val

        system_match = re.search(r'^SYSTEM\s+"""(.*?)"""', content, re.DOTALL | re.MULTILINE)
        self.model_config['SYSTEM'] = system_match.group(1).strip() if system_match else ""
        
        self.logger.info("Successfully parsed Modelfile.")
        self.logger.debug(f"Parsed config: {json.dumps(self.model_config, indent=2)}")

    def _initialize_rkllm(self):
        """Initialize RKLLM runtime with parameters from the parsed Modelfile."""
        params = self.model_config.get('PARAMETERS', {})
        self.logger.debug(f"Initializing RKLLM with parameters: {params}")
        
        max_context_len_from_model = params.get('max_context_len')
        if max_context_len_from_model:
            max_context_len = int(max_context_len_from_model)
            self.logger.info(f"Using max_context_len from Modelfile: {max_context_len}")
        else:
            max_context_len = self.global_config.getint('model', 'max_context_length')
            self.logger.info(f"Warning: No max_context_len in Modelfile. Falling back to global config value: {max_context_len}")

        rkllm_config = {
            'max_context_len': max_context_len,
            'max_new_tokens': int(params.get('num_predict', self.global_config.getint('model', 'max_new_tokens'))),
            'temperature': float(params.get('temperature', self.global_config.getfloat('model', 'default_temperature'))),
            'top_k': int(params.get('top_k', 40)),
            'top_p': float(params.get('top_p', 0.9)),
            'repeat_penalty': float(params.get('repeat_penalty', 1.1)),
            'skip_special_token': True,
        }
        
        self.rkllm_runtime = RKLLMRuntime()
        if not self.rkllm_runtime.initialize(self.model_path, rkllm_config):
            raise RuntimeError("Failed to initialize RKLLM runtime")
        
        self.logger.info("RKLLM runtime initialized successfully.")

    def _apply_chat_template(self, messages: List[Dict[str, Any]]) -> str:
        """
        Build a single prompt string from messages using the model's specific chat template.
        This is a robust implementation that handles different template styles correctly.
        """
        system_prompt = self.model_config.get('SYSTEM', '')
        model_family = "unknown"

        # Automatically detect model family from its name
        if "gemma" in self.model_name.lower():
            model_family = "gemma"
        elif "llama-3.1" in self.model_name.lower() or "llama3.1" in self.model_name.lower():
            model_family = "llama3.1"
        elif "llama" in self.model_name.lower():
            model_family = "llama" # Fallback for other Llama versions

        self.logger.info(f"Applying chat template for detected model family: '{model_family}'")
        final_prompt_parts = []
        
        # --- Llama 3.1 Template ---
        if model_family == "llama3.1":
            final_prompt_parts.append("<|begin_of_text|>")
            if system_prompt:
                final_prompt_parts.append(f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>")
            
            for msg in messages:
                role, content = msg.get("role"), msg.get("content", "")
                if role == "user":
                    final_prompt_parts.append(f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>")
                elif role == "assistant":
                    final_prompt_parts.append(f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>")
            
            # This is crucial: it signals the model to start its response.
            final_prompt_parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")

        # --- Gemma Template ---
        elif model_family == "gemma":
            # Gemma has a simpler turn-based structure
            for i, msg in enumerate(messages):
                role, content = msg.get("role"), msg.get("content", "")
                if role == "user":
                    prompt_text = content
                    # Gemma includes the system prompt within the first user turn
                    if i == 0 and system_prompt:
                        prompt_text = f"{system_prompt}\n{prompt_text}"
                    final_prompt_parts.append(f"<start_of_turn>user\n{prompt_text}<end_of_turn>\n")
                elif role == "assistant":
                    final_prompt_parts.append(f"<start_of_turn>model\n{content}<end_of_turn>\n")
            
            # Signal the model to start its response
            final_prompt_parts.append("<start_of_turn>model\n")
            
        # --- Fallback for unknown models ---
        else:
            self.logger.warning("Unknown model family, using basic role: content formatting.")
            if system_prompt:
                final_prompt_parts.append(system_prompt + "\n\n")
            for msg in messages:
                final_prompt_parts.append(f"{msg.get('role')}: {msg.get('content')}\n")
            final_prompt_parts.append("assistant:")

        final_prompt = "".join(final_prompt_parts)
        self.logger.debug(f"Final formatted prompt (first 500 chars):\n---\n{final_prompt[:500]}\n---")
        return final_prompt
        
    def generate(self, messages: List[Dict[str, Any]], images: List[str] = None, **kwargs) -> Dict[str, Any]:
        """Generate a response. Now uses the templating system."""
        if not self.is_loaded or not self.rkllm_runtime:
            raise RuntimeError("Model not loaded")
        
        start_time = time.time()
        
        if images:
            self.logger.info(f"Received {len(images)} images (multimodal templating not yet implemented).")

        formatted_prompt = self._apply_chat_template(messages)
        
        if not self.rkllm_runtime.run_inference(formatted_prompt):
            raise RuntimeError("RKLLM inference failed to start")
        
        response_text = self.rkllm_runtime.get_response(timeout=120.0)
        
        stop_tokens = self.model_config.get('PARAMETERS', {}).get('stop', [])
        if stop_tokens:
            for token in stop_tokens:
                if response_text.endswith(token):
                    response_text = response_text[:-len(token)]
        
        generation_time = time.time() - start_time

        return {
            'text': response_text.strip(),
            'model': self.model_name,
            'generation_time': generation_time
        }

    def generate_stream(self, messages: List[Dict[str, Any]], images: List[str] = None, **kwargs) -> Generator[Dict[str, Any], None, None]:
        """Generate a streaming response using the templating system."""
        if not self.is_loaded or not self.rkllm_runtime:
            raise RuntimeError("Model not loaded")

        formatted_prompt = self._apply_chat_template(messages)

        if not self.rkllm_runtime.run_inference(formatted_prompt):
            raise RuntimeError("RKLLM inference failed to start")

        full_response = ""
        stop_tokens = self.model_config.get('PARAMETERS', {}).get('stop', [])

        for chunk in self.rkllm_runtime.get_response_stream(timeout=120.0):
            full_response += chunk
            
            stopped = False
            if stop_tokens:
                for token in stop_tokens:
                    if full_response.endswith(token):
                        # Yield the part before the stop token and then terminate
                        yield {'text': chunk.replace(token, ''), 'done': False, 'model': self.model_name}
                        stopped = True
                        break
            if stopped:
                break
            
            yield {'text': chunk, 'done': False, 'model': self.model_name}
        
        yield {'text': '', 'done': True, 'model': self.model_name}

    def release(self):
        """Release all model resources."""
        if self.rkllm_runtime:
            self.rkllm_runtime.release()
            self.rkllm_runtime = None
        if self.npu_optimizer:
            self.npu_optimizer.restore_original_settings()
        self.is_loaded = False
        self.logger.info(f"Model '{self.model_name}' released.")

    def __del__(self):
        self.release()

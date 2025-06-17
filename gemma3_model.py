"""
Gemma3 Model Integration
This version is adapted to use the RKLLMRuntime class directly.
"""
import os
import time
import logging
import threading
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
        self.default_temperature = config.getfloat('model', 'default_temperature', fallback=0.7)
        
        self.npu_optimizer = NPUOptimizer(config)
        self.image_processor = ImageProcessor(config)
        
        self.rkllm_runtime = None
        self.is_loaded = False
        self.model_path = None
        self.model_config = {}
        
        # Chat template components
        self.system_prompt = ""
        self.user_prefix = ""
        self.user_postfix = ""
        
        self._load_model()
    
    def _load_model(self):
        """Load the Gemma3 model"""
        try:
            with LogExecutionTime(self.logger, f"Loading model {self.model_name}"):
                model_dir = self.models_dir / self.model_name
                if not model_dir.exists():
                    raise FileNotFoundError(f"Model directory not found: {model_dir}")
                
                self._load_model_config(model_dir)
                
                rkllm_files = list(model_dir.glob("*.rkllm"))
                if not rkllm_files:
                    raise FileNotFoundError(f"No .rkllm file found in {model_dir}")
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
        """Load model configuration from Modelfile"""
        modelfile_path = model_dir / "Modelfile"
        if modelfile_path.exists():
            with open(modelfile_path, 'r') as f:
                for line in f:
                    if '=' in line and not line.strip().startswith('#'):
                        key, value = line.split('=', 1)
                        self.model_config[key.strip().upper()] = value.strip().strip('"')
            self.logger.debug(f"Loaded model config: {self.model_config}")

    def _initialize_rkllm(self):
        """Initialize RKLLM runtime directly"""
        rkllm_config = {
            'max_context_len': self.max_context_length,
            'max_new_tokens': 2048,
            'temperature': float(self.model_config.get('TEMPERATURE', self.default_temperature)),
            'top_k': 1,
            'top_p': 0.9,
            'repeat_penalty': 1.1,
            'frequency_penalty': 0.0,
            'presence_penalty': 0.0,
            'skip_special_token': True,
            'img_start': "",
            'img_end': "",
            'img_content': ""
        }
        
        # Initialize RKLLM runtime
        self.rkllm_runtime = RKLLMRuntime()
        success = self.rkllm_runtime.initialize(self.model_path, rkllm_config)
        
        if not success:
            raise RuntimeError("Failed to initialize RKLLM runtime")
        
        # Set chat template components
        self.system_prompt = self.model_config.get('SYSTEM', '')
        self.user_prefix = '<start_of_turn>user\n'
        self.user_postfix = '<end_of_turn>\n<start_of_turn>model\n'
        
        self.logger.info("RKLLM runtime initialized successfully")

    def _format_prompt(self, prompt: str) -> str:
        """Format prompt with chat template"""
        return f"{self.system_prompt}{self.user_prefix}{prompt}{self.user_postfix}"

    def generate(self, prompt: str, images: List[str] = None, **kwargs) -> Dict[str, Any]:
        """Generate text response with optional image support"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        start_time = time.time()
        
        # Handle multimodal input if images are provided
        if images:
            # Process images and modify prompt accordingly
            processed_images = []
            for image_path in images:
                try:
                    processed_image = self.image_processor.process_image(image_path)
                    processed_images.append(processed_image)
                except Exception as e:
                    self.logger.warning(f"Failed to process image {image_path}: {e}")
            
            if processed_images:
                # Add image context to prompt
                image_context = f"\n[Context: {len(processed_images)} image(s) provided for analysis]"
                prompt = prompt + image_context
        
        # Format prompt with chat template
        formatted_prompt = self._format_prompt(prompt)
        
        # Run inference
        success = self.rkllm_runtime.run_inference(formatted_prompt)
        if not success:
            raise RuntimeError("Inference failed")
        
        # Get response
        response_text = self.rkllm_runtime.get_response(timeout=kwargs.get('timeout', 60.0))
        generation_time = time.time() - start_time
        
        return {
            'text': response_text,
            'generation_time': generation_time,
            'model': self.model_name,
            'images_processed': len(images) if images else 0
        }

    def generate_stream(self, prompt: str, images: List[str] = None, **kwargs) -> Generator[Dict[str, Any], None, None]:
        """Generate streaming text response with optional image support"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        # Handle multimodal input if images are provided
        if images:
            processed_images = []
            for image_path in images:
                try:
                    processed_image = self.image_processor.process_image(image_path)
                    processed_images.append(processed_image)
                except Exception as e:
                    self.logger.warning(f"Failed to process image {image_path}: {e}")
            
            if processed_images:
                image_context = f"\n[Context: {len(processed_images)} image(s) provided for analysis]"
                prompt = prompt + image_context
        
        # Format prompt with chat template
        formatted_prompt = self._format_prompt(prompt)
        
        # Run inference
        success = self.rkllm_runtime.run_inference(formatted_prompt)
        if not success:
            raise RuntimeError("Inference failed")
        
        # Stream response
        for chunk in self.rkllm_runtime.get_response_stream(timeout=kwargs.get('timeout', 60.0)):
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

    def generate_multimodal(self, prompt: str, images: List[str] = None, **kwargs) -> Dict[str, Any]:
        """Generate multimodal response (deprecated - use generate() instead)"""
        self.logger.warning("generate_multimodal() is deprecated, use generate() with images parameter")
        return self.generate(prompt, images, **kwargs)

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'name': self.model_name,
            'path': self.model_path,
            'is_loaded': self.is_loaded,
            'config': self.model_config,
            'max_context_length': self.max_context_length,
            'temperature': self.default_temperature
        }

    def update_config(self, **kwargs):
        """Update model configuration"""
        for key, value in kwargs.items():
            if key == 'temperature':
                self.default_temperature = float(value)
            elif key == 'max_context_length':
                self.max_context_length = int(value)
        
        self.logger.info(f"Model configuration updated: {kwargs}")

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


class ModelManager:
    """Manager for multiple Gemma3 models"""
    
    def __init__(self, config: configparser.ConfigParser):
        self.config = config
        self.logger = logging.getLogger("gemma3-rkllm.manager")
        self.models = {}
        self.current_model = None
    
    def load_model(self, model_name: str) -> Gemma3Model:
        """Load a specific model"""
        if model_name in self.models:
            self.logger.info(f"Model {model_name} already loaded")
            return self.models[model_name]
        
        try:
            model = Gemma3Model(model_name, self.config)
            self.models[model_name] = model
            self.current_model = model_name
            self.logger.info(f"Model {model_name} loaded and set as current")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def unload_model(self, model_name: str):
        """Unload a specific model"""
        if model_name in self.models:
            self.models[model_name].release()
            del self.models[model_name]
            
            if self.current_model == model_name:
                self.current_model = None
                
            self.logger.info(f"Model {model_name} unloaded")
    
    def get_current_model(self) -> Optional[Gemma3Model]:
        """Get the current active model"""
        if self.current_model and self.current_model in self.models:
            return self.models[self.current_model]
        return None
    
    def set_current_model(self, model_name: str):
        """Set the current active model"""
        if model_name in self.models:
            self.current_model = model_name
            self.logger.info(f"Current model set to {model_name}")
        else:
            raise ValueError(f"Model {model_name} not loaded")
    
    def list_models(self) -> List[str]:
        """List all loaded models"""
        return list(self.models.keys())
    
    def list_available_models(self) -> List[str]:
        """List all available models in the models directory"""
        models_dir = Path(self.config.get('model', 'models_dir', fallback='./models'))
        available_models = []
        
        if models_dir.exists():
            for model_dir in models_dir.iterdir():
                if model_dir.is_dir():
                    rkllm_files = list(model_dir.glob("*.rkllm"))
                    if rkllm_files:
                        available_models.append(model_dir.name)
        
        return available_models
    
    def release_all(self):
        """Release all loaded models"""
        for model_name in list(self.models.keys()):
            self.unload_model(model_name)
        self.current_model = None
        self.logger.info("All models released")
    
    def __del__(self):
        """Destructor to ensure all models are released"""
        self.release_all()

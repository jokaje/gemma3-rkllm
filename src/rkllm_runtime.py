"""
RKLLM Integration for Gemma3
Handles the C++ RKLLM library integration with NPU optimization
"""

import os
import sys
import ctypes
import logging
import multiprocessing
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Generator
import configparser
import time

from .utils import generate_unique_id, format_duration


# RKLLM C++ library structures and enums
class RKLLMInputMode(ctypes.c_int):
    RKLLM_INPUT_PROMPT = 0
    RKLLM_INPUT_TOKEN = 1
    RKLLM_INPUT_MULTIMODAL = 2


class RKLLMInferMode(ctypes.c_int):
    RKLLM_INFER_GENERATE = 0
    RKLLM_INFER_GET_LAST_HIDDEN_LAYER = 1


class RKLLMExtendParam(ctypes.Structure):
    _fields_ = [
        ("base_domain_id", ctypes.c_int),
        ("embed_flash", ctypes.c_int),
        ("enabled_cpus_num", ctypes.c_int),
        ("enabled_cpus_mask", ctypes.c_uint64)
    ]


class RKLLMParam(ctypes.Structure):
    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("max_context_len", ctypes.c_int),
        ("max_new_tokens", ctypes.c_int),
        ("skip_special_token", ctypes.c_bool),
        ("top_k", ctypes.c_int),
        ("top_p", ctypes.c_float),
        ("temperature", ctypes.c_float),
        ("repeat_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("mirostat", ctypes.c_int),
        ("mirostat_tau", ctypes.c_float),
        ("mirostat_eta", ctypes.c_float),
        ("is_async", ctypes.c_bool),
        ("img_start", ctypes.c_char_p),
        ("img_end", ctypes.c_char_p),
        ("img_content", ctypes.c_char_p),
        ("extend_param", RKLLMExtendParam)
    ]


class RKLLMTokenInput(ctypes.Structure):
    _fields_ = [
        ("input_ids", ctypes.POINTER(ctypes.c_int)),
        ("n_tokens", ctypes.c_ulong)
    ]


class RKLLMPromptInput(ctypes.Structure):
    _fields_ = [
        ("prompt", ctypes.c_char_p)
    ]


class RKLLMMultimodalInput(ctypes.Structure):
    _fields_ = [
        ("prompt", ctypes.c_char_p),
        ("image_embed", ctypes.POINTER(ctypes.c_float)),
        ("image_embed_num", ctypes.c_int)
    ]


class RKLLMInputUnion(ctypes.Union):
    _fields_ = [
        ("prompt_input", RKLLMPromptInput),
        ("token_input", RKLLMTokenInput),
        ("multimodal_input", RKLLMMultimodalInput)
    ]


class RKLLMInput(ctypes.Structure):
    _fields_ = [
        ("input_mode", ctypes.c_int),
        ("input_data", RKLLMInputUnion)
    ]


class RKLLMLoraAdapter(ctypes.Structure):
    _fields_ = [
        ("lora_adapter_path", ctypes.c_char_p),
        ("lora_adapter_name", ctypes.c_char_p),
        ("scale", ctypes.c_float)
    ]


class RKLLMLoraParam(ctypes.Structure):
    _fields_ = [
        ("lora_adapter_name", ctypes.c_char_p)
    ]


class RKLLMInferParam(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_int),
        ("lora_params", ctypes.POINTER(RKLLMLoraParam))
    ]


class RKLLMResult(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("size", ctypes.c_int),
        ("flag", ctypes.c_int)
    ]


# Handle type
RKLLM_Handle_t = ctypes.c_void_p


class RKLLMRuntime:
    """RKLLM Runtime wrapper for C++ library"""
    
    def __init__(self, lib_path: Optional[str] = None):
        self.logger = logging.getLogger("gemma3-rkllm.rkllm")
        
        # Load RKLLM library
        if lib_path is None:
            lib_path = self._find_rkllm_library()
        
        try:
            self.lib = ctypes.CDLL(lib_path)
            self.logger.info(f"RKLLM library loaded from: {lib_path}")
        except Exception as e:
            self.logger.error(f"Failed to load RKLLM library: {e}")
            raise RuntimeError(f"Could not load RKLLM library: {e}")
        
        self._setup_function_signatures()
        self.callback_type = ctypes.CFUNCTYPE(None, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int)
        self.callback = self.callback_type(self._callback_impl)
        
        # Runtime state
        self.handle = None
        self.is_initialized = False
        self.current_response = []
        self.generation_complete = False
    
    def _find_rkllm_library(self) -> str:
        """Find RKLLM library in common locations"""
        possible_paths = [
            "./lib/librkllmrt.so",
            "/usr/local/lib/librkllmrt.so",
            "/usr/lib/librkllmrt.so",
            "./librkllmrt.so"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        raise FileNotFoundError("RKLLM library not found. Please ensure librkllmrt.so is available.")
    
    def _setup_function_signatures(self):
        """Setup C function signatures"""
        # rkllm_init
        self.lib.rkllm_init.argtypes = [
            ctypes.POINTER(RKLLM_Handle_t), 
            ctypes.POINTER(RKLLMParam), 
            self.callback_type
        ]
        self.lib.rkllm_init.restype = ctypes.c_int
        
        # rkllm_run
        self.lib.rkllm_run.argtypes = [
            RKLLM_Handle_t, 
            ctypes.POINTER(RKLLMInput), 
            ctypes.POINTER(RKLLMInferParam), 
            ctypes.c_void_p
        ]
        self.lib.rkllm_run.restype = ctypes.c_int
        
        # rkllm_destroy
        self.lib.rkllm_destroy.argtypes = [RKLLM_Handle_t]
        self.lib.rkllm_destroy.restype = ctypes.c_int
        
        # rkllm_load_lora (optional)
        if hasattr(self.lib, 'rkllm_load_lora'):
            self.lib.rkllm_load_lora.argtypes = [
                RKLLM_Handle_t, 
                ctypes.POINTER(RKLLMLoraAdapter)
            ]
            self.lib.rkllm_load_lora.restype = ctypes.c_int
        
        # rkllm_load_prompt_cache (optional)
        if hasattr(self.lib, 'rkllm_load_prompt_cache'):
            self.lib.rkllm_load_prompt_cache.argtypes = [
                RKLLM_Handle_t, 
                ctypes.c_char_p
            ]
            self.lib.rkllm_load_prompt_cache.restype = ctypes.c_int
    
    def _callback_impl(self, result: ctypes.POINTER(RKLLMResult), userdata: ctypes.c_void_p, state: ctypes.c_int):
        """Callback implementation for RKLLM responses"""
        try:
            if result and result.contents:
                text = result.contents.text.decode('utf-8') if result.contents.text else ""
                flag = result.contents.flag
                
                if text:
                    self.current_response.append(text)
                
                # Check if generation is complete
                if flag == 1:  # Assuming flag 1 means completion
                    self.generation_complete = True
                    
        except Exception as e:
            self.logger.error(f"Error in callback: {e}")
    
    def initialize(self, model_path: str, config: Dict[str, Any]) -> bool:
        """
        Initialize RKLLM model
        
        Args:
            model_path: Path to .rkllm model file
            config: Configuration dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.is_initialized:
                self.logger.warning("RKLLM already initialized")
                return True
            
            # Setup parameters
            rkllm_param = RKLLMParam()
            rkllm_param.model_path = model_path.encode('utf-8')
            rkllm_param.max_context_len = config.get('max_context_len', 128000)
            rkllm_param.max_new_tokens = config.get('max_new_tokens', 2048)
            rkllm_param.skip_special_token = config.get('skip_special_token', True)
            rkllm_param.top_k = config.get('top_k', 1)
            rkllm_param.top_p = config.get('top_p', 0.9)
            rkllm_param.temperature = config.get('temperature', 0.7)
            rkllm_param.repeat_penalty = config.get('repeat_penalty', 1.1)
            rkllm_param.frequency_penalty = config.get('frequency_penalty', 0.0)
            rkllm_param.presence_penalty = config.get('presence_penalty', 0.0)
            rkllm_param.mirostat = config.get('mirostat', 0)
            rkllm_param.mirostat_tau = config.get('mirostat_tau', 5.0)
            rkllm_param.mirostat_eta = config.get('mirostat_eta', 0.1)
            rkllm_param.is_async = config.get('is_async', False)
            
            # Multimodal parameters for Gemma3
            rkllm_param.img_start = config.get('img_start', "<start_of_image>").encode('utf-8')
            rkllm_param.img_end = config.get('img_end', "<end_of_image>").encode('utf-8')
            rkllm_param.img_content = config.get('img_content', "").encode('utf-8')
            
            # Extended parameters for NPU optimization
            rkllm_param.extend_param.base_domain_id = config.get('base_domain_id', 0)
            rkllm_param.extend_param.embed_flash = config.get('embed_flash', 1)
            rkllm_param.extend_param.enabled_cpus_num = config.get('enabled_cpus_num', multiprocessing.cpu_count())
            rkllm_param.extend_param.enabled_cpus_mask = config.get('enabled_cpus_mask', (1 << (rkllm_param.extend_param.enabled_cpus_num + 1)) - 1)
            
            # Initialize
            self.handle = RKLLM_Handle_t()
            ret = self.lib.rkllm_init(ctypes.byref(self.handle), ctypes.byref(rkllm_param), self.callback)
            
            if ret != 0:
                raise RuntimeError(f"RKLLM initialization failed with code: {ret}")
            
            self.is_initialized = True
            self.logger.info("RKLLM initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize RKLLM: {e}")
            return False
    
    def run_inference(self, input_data: Union[str, List[int]], input_mode: str = "prompt") -> bool:
        """
        Run inference with RKLLM
        
        Args:
            input_data: Input prompt string or token list
            input_mode: Input mode ("prompt", "token", or "multimodal")
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.is_initialized:
                raise RuntimeError("RKLLM not initialized")
            
            # Reset response state
            self.current_response = []
            self.generation_complete = False
            
            # Setup input
            rkllm_input = RKLLMInput()
            
            if input_mode == "prompt":
                rkllm_input.input_mode = RKLLMInputMode.RKLLM_INPUT_PROMPT
                rkllm_input.input_data.prompt_input.prompt = input_data.encode('utf-8')
                
            elif input_mode == "token":
                rkllm_input.input_mode = RKLLMInputMode.RKLLM_INPUT_TOKEN
                if not isinstance(input_data, list):
                    raise ValueError("Token input must be a list of integers")
                
                # Ensure EOS token
                if input_data[-1] != 2:
                    input_data.append(2)
                
                token_array = (ctypes.c_int * len(input_data))(*input_data)
                rkllm_input.input_data.token_input.input_ids = token_array
                rkllm_input.input_data.token_input.n_tokens = ctypes.c_ulong(len(input_data))
                
            elif input_mode == "multimodal":
                rkllm_input.input_mode = RKLLMInputMode.RKLLM_INPUT_MULTIMODAL
                # Multimodal input handling would be implemented here
                # For now, fall back to prompt mode
                rkllm_input.input_mode = RKLLMInputMode.RKLLM_INPUT_PROMPT
                rkllm_input.input_data.prompt_input.prompt = str(input_data).encode('utf-8')
            
            # Setup inference parameters
            rkllm_infer_params = RKLLMInferParam()
            ctypes.memset(ctypes.byref(rkllm_infer_params), 0, ctypes.sizeof(RKLLMInferParam))
            rkllm_infer_params.mode = RKLLMInferMode.RKLLM_INFER_GENERATE
            rkllm_infer_params.lora_params = None  # No LoRA for now
            
            # Run inference
            ret = self.lib.rkllm_run(self.handle, ctypes.byref(rkllm_input), ctypes.byref(rkllm_infer_params), None)
            
            if ret != 0:
                raise RuntimeError(f"RKLLM inference failed with code: {ret}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to run inference: {e}")
            return False
    
    def get_response(self, timeout: float = 30.0) -> str:
        """
        Get complete response from inference
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Generated text response
        """
        start_time = time.time()
        
        while not self.generation_complete:
            if time.time() - start_time > timeout:
                self.logger.warning("Response generation timed out")
                break
            time.sleep(0.01)  # Small sleep to prevent busy waiting
        
        return ''.join(self.current_response)
    
    def get_response_stream(self, timeout: float = 30.0) -> Generator[str, None, None]:
        """
        Get streaming response from inference
        
        Args:
            timeout: Timeout in seconds
            
        Yields:
            Text chunks as they are generated
        """
        start_time = time.time()
        last_length = 0
        
        while not self.generation_complete:
            if time.time() - start_time > timeout:
                self.logger.warning("Response generation timed out")
                break
            
            current_length = len(self.current_response)
            if current_length > last_length:
                # Yield new chunks
                for i in range(last_length, current_length):
                    yield self.current_response[i]
                last_length = current_length
            
            time.sleep(0.01)
    
    def release(self):
        """Release RKLLM resources"""
        try:
            if self.is_initialized and self.handle:
                ret = self.lib.rkllm_destroy(self.handle)
                if ret != 0:
                    self.logger.warning(f"RKLLM destroy returned code: {ret}")
                else:
                    self.logger.info("RKLLM resources released successfully")
                
                self.handle = None
                self.is_initialized = False
                
        except Exception as e:
            self.logger.error(f"Error releasing RKLLM: {e}")
    
    def load_lora_adapter(self, adapter_path: str, adapter_name: str, scale: float = 1.0) -> bool:
        """
        Load LoRA adapter (if supported)
        
        Args:
            adapter_path: Path to LoRA adapter
            adapter_name: Name of the adapter
            scale: LoRA scale factor
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not hasattr(self.lib, 'rkllm_load_lora'):
                self.logger.warning("LoRA loading not supported in this RKLLM version")
                return False
            
            if not self.is_initialized:
                raise RuntimeError("RKLLM not initialized")
            
            lora_adapter = RKLLMLoraAdapter()
            ctypes.memset(ctypes.byref(lora_adapter), 0, ctypes.sizeof(RKLLMLoraAdapter))
            lora_adapter.lora_adapter_path = adapter_path.encode('utf-8')
            lora_adapter.lora_adapter_name = adapter_name.encode('utf-8')
            lora_adapter.scale = scale
            
            ret = self.lib.rkllm_load_lora(self.handle, ctypes.byref(lora_adapter))
            
            if ret != 0:
                raise RuntimeError(f"LoRA loading failed with code: {ret}")
            
            self.logger.info(f"LoRA adapter loaded: {adapter_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load LoRA adapter: {e}")
            return False
    
    def __del__(self):
        """Destructor to ensure resources are released"""
        self.release()


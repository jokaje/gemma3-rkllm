"""
RKLLM Integration for Gemma3
Handles the C++ RKLLM library integration with NPU optimization
"""

import os
import sys
import ctypes
import logging
import threading
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Generator
import configparser

from .utils import generate_unique_id, format_duration


# ==============================================================================
# KORRIGIERTE CTYPES-STRUKTUREN
# ==============================================================================

class RKLLMInputMode(ctypes.c_int):
    RKLLM_INPUT_PROMPT = 0
    RKLLM_INPUT_TOKEN = 1
    RKLLM_INPUT_MULTIMODAL = 2


class RKLLMInferMode(ctypes.c_int):
    RKLLM_INFER_GENERATE = 0
    RKLLM_INFER_GET_LAST_HIDDEN_LAYER = 1


class LLMCallState(ctypes.c_int):
    RKLLM_RUN_NORMAL = 0
    RKLLM_RUN_WAITING = 1
    RKLLM_RUN_FINISH = 2
    RKLLM_RUN_ERROR = 3


class RKLLMExtendParam(ctypes.Structure):
    _fields_ = [
        ("base_domain_id", ctypes.c_int32),
        ("embed_flash", ctypes.c_int8),
        ("enabled_cpus_num", ctypes.c_int8),
        ("enabled_cpus_mask", ctypes.c_uint32),
        ("reserved", ctypes.c_uint8 * 106)
    ]


class RKLLMParam(ctypes.Structure):
    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("max_context_len", ctypes.c_int32),
        ("max_new_tokens", ctypes.c_int32),
        ("top_k", ctypes.c_int32),
        ("n_keep", ctypes.c_int32),
        ("top_p", ctypes.c_float),
        ("temperature", ctypes.c_float),
        ("repeat_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("mirostat", ctypes.c_int32),
        ("mirostat_tau", ctypes.c_float),
        ("mirostat_eta", ctypes.c_float),
        ("skip_special_token", ctypes.c_bool),
        ("is_async", ctypes.c_bool),
        ("img_start", ctypes.c_char_p),
        ("img_end", ctypes.c_char_p),
        ("img_content", ctypes.c_char_p),
        ("extend_param", RKLLMExtendParam)
    ]


class RKLLMTokenInput(ctypes.Structure):
    _fields_ = [
        ("input_ids", ctypes.POINTER(ctypes.c_int32)),
        ("n_tokens", ctypes.c_size_t)
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
        ("lora_params", ctypes.POINTER(RKLLMLoraParam)),
        ("keep_history", ctypes.c_int)
    ]


class RKLLMResultLastHiddenLayer(ctypes.Structure):
    _fields_ = [
        ("hidden_states", ctypes.POINTER(ctypes.c_float)),
        ("embd_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int)
    ]


class RKLLMResultLogits(ctypes.Structure):
    _fields_ = [
        ("logits", ctypes.POINTER(ctypes.c_float)),
        ("vocab_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int)
    ]


class RKLLMResult(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("token_id", ctypes.c_int),
        ("last_hidden_layer", RKLLMResultLastHiddenLayer),
        ("logits", RKLLMResultLogits)
    ]


# Handle type
RKLLM_Handle_t = ctypes.c_void_p


class RKLLMRuntime:
    """RKLLM Runtime wrapper for C++ library with proper threading"""

    def __init__(self, lib_path: Optional[str] = None):
        self.logger = logging.getLogger("gemma3-rkllm.rkllm")
        
        if lib_path is None:
            lib_path = self._find_rkllm_library()
        
        try:
            self.lib = ctypes.CDLL(lib_path)
            self.logger.info(f"RKLLM library loaded from: {lib_path}")
        except Exception as e:
            self.logger.error(f"Failed to load RKLLM library: {e}")
            raise RuntimeError(f"Could not load RKLLM library: {e}")
        
        # Threading components
        self.inference_thread = None
        self.thread_lock = threading.Lock()
        self.response_event = threading.Event()
        
        # Response handling
        self.current_response = []
        self.generation_complete = False
        self.generation_error = False
        
        # Setup callback
        self.callback_type = ctypes.CFUNCTYPE(None, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int)
        self.callback = self.callback_type(self._callback_impl)

        self._setup_function_signatures()
        
        self.handle = None
        self.is_initialized = False

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
        self.lib.rkllm_init.argtypes = [
            ctypes.POINTER(RKLLM_Handle_t), 
            ctypes.POINTER(RKLLMParam), 
            self.callback_type
        ]
        self.lib.rkllm_init.restype = ctypes.c_int
        
        self.lib.rkllm_run.argtypes = [
            RKLLM_Handle_t, 
            ctypes.POINTER(RKLLMInput), 
            ctypes.POINTER(RKLLMInferParam), 
            ctypes.c_void_p
        ]
        self.lib.rkllm_run.restype = ctypes.c_int
        
        self.lib.rkllm_destroy.argtypes = [RKLLM_Handle_t]
        self.lib.rkllm_destroy.restype = ctypes.c_int
        
        # Optional functions
        try:
            self.lib.rkllm_set_chat_template.argtypes = [
                RKLLM_Handle_t, 
                ctypes.c_char_p, 
                ctypes.c_char_p, 
                ctypes.c_char_p
            ]
            self.lib.rkllm_set_chat_template.restype = ctypes.c_int
            self.has_chat_template = True
        except AttributeError:
            self.has_chat_template = False
            self.logger.warning("rkllm_set_chat_template not available in library")
        
    def _callback_impl(self, result: ctypes.POINTER(RKLLMResult), userdata: ctypes.c_void_p, state: ctypes.c_int):
        """Callback implementation for RKLLM responses with proper threading"""
        try:
            with self.thread_lock:
                if state == LLMCallState.RKLLM_RUN_NORMAL:
                    # Normal token generation
                    if result and result.contents and result.contents.text:
                        text = result.contents.text.decode('utf-8')
                        if text:
                            self.current_response.append(text)
                            
                elif state == LLMCallState.RKLLM_RUN_FINISH:
                    # Generation finished successfully
                    self.generation_complete = True
                    self.response_event.set()
                    
                elif state == LLMCallState.RKLLM_RUN_ERROR:
                    # Error occurred
                    self.logger.error("RKLLM callback reported an error state")
                    self.generation_error = True
                    self.generation_complete = True
                    self.response_event.set()
                    
        except Exception as e:
            self.logger.error(f"Error in callback: {e}", exc_info=True)
            with self.thread_lock:
                self.generation_error = True
                self.generation_complete = True
                self.response_event.set()

    def initialize(self, model_path: str, config: Dict[str, Any]) -> bool:
        """Initialize RKLLM model"""
        try:
            if self.is_initialized:
                self.logger.warning("RKLLM already initialized")
                return True
            
            rkllm_param = RKLLMParam()
            rkllm_param.model_path = model_path.encode('utf-8')
            rkllm_param.max_context_len = config.get('max_context_len', 16384)
            rkllm_param.max_new_tokens = config.get('max_new_tokens', 2048)
            rkllm_param.skip_special_token = config.get('skip_special_token', True)
            rkllm_param.top_k = config.get('top_k', 1)
            rkllm_param.top_p = float(config.get('top_p', 0.9))
            rkllm_param.temperature = float(config.get('temperature', 0.8))
            rkllm_param.repeat_penalty = float(config.get('repeat_penalty', 1.1))
            rkllm_param.frequency_penalty = float(config.get('frequency_penalty', 0.0))
            rkllm_param.presence_penalty = float(config.get('presence_penalty', 0.0))
            rkllm_param.n_keep = -1
            rkllm_param.mirostat = 0
            rkllm_param.mirostat_tau = 5.0
            rkllm_param.mirostat_eta = 0.1
            rkllm_param.is_async = False
            
            rkllm_param.img_start = config.get('img_start', "").encode('utf-8')
            rkllm_param.img_end = config.get('img_end', "").encode('utf-8')
            rkllm_param.img_content = config.get('img_content', "").encode('utf-8')
            
            # CPU configuration - assign high-performance cores
            self.logger.info("Assigning 4 high-performance CPU cores (4,5,6,7) to RKLLM")
            rkllm_param.extend_param.base_domain_id = 0
            rkllm_param.extend_param.embed_flash = 0
            rkllm_param.extend_param.enabled_cpus_num = 4
            rkllm_param.extend_param.enabled_cpus_mask = (1 << 4)|(1 << 5)|(1 << 6)|(1 << 7)
            
            self.handle = RKLLM_Handle_t()
            ret = self.lib.rkllm_init(ctypes.byref(self.handle), ctypes.byref(rkllm_param), self.callback)
            
            if ret != 0:
                raise RuntimeError(f"RKLLM initialization failed with code: {ret}")
            
            self.is_initialized = True
            self.logger.info("RKLLM initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize RKLLM: {e}")
            self.release()
            return False

    def set_chat_template(self, system_prompt: str, user_prefix: str, user_postfix: str) -> bool:
        """Set chat template if supported"""
        if not self.has_chat_template or not self.is_initialized:
            return False
            
        try:
            ret = self.lib.rkllm_set_chat_template(
                self.handle,
                system_prompt.encode('utf-8'),
                user_prefix.encode('utf-8'),
                user_postfix.encode('utf-8')
            )
            
            if ret == 0:
                self.logger.info("Chat template set successfully")
                return True
            else:
                self.logger.warning(f"Failed to set chat template, code: {ret}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error setting chat template: {e}")
            return False

    def _run_inference_thread(self, input_data: str):
        """Run inference in separate thread"""
        try:
            rkllm_input = RKLLMInput()
            rkllm_input.input_mode = RKLLMInputMode.RKLLM_INPUT_PROMPT
            rkllm_input.input_data.prompt_input.prompt = input_data.encode('utf-8')

            rkllm_infer_params = RKLLMInferParam()
            ctypes.memset(ctypes.byref(rkllm_infer_params), 0, ctypes.sizeof(RKLLMInferParam))
            rkllm_infer_params.mode = RKLLMInferMode.RKLLM_INFER_GENERATE
            rkllm_infer_params.lora_params = None
            rkllm_infer_params.keep_history = 0
            
            ret = self.lib.rkllm_run(self.handle, ctypes.byref(rkllm_input), ctypes.byref(rkllm_infer_params), None)
            
            if ret != 0:
                self.logger.error(f"RKLLM inference failed with code: {ret}")
                with self.thread_lock:
                    self.generation_error = True
                    self.generation_complete = True
                    self.response_event.set()
            
        except Exception as e:
            self.logger.error(f"Error in inference thread: {e}")
            with self.thread_lock:
                self.generation_error = True
                self.generation_complete = True
                self.response_event.set()

    def run_inference(self, input_data: str) -> bool:
        """Run inference with RKLLM in separate thread"""
        try:
            if not self.is_initialized:
                raise RuntimeError("RKLLM not initialized")
            
            # Reset state
            with self.thread_lock:
                self.current_response = []
                self.generation_complete = False
                self.generation_error = False
                self.response_event.clear()
            
            # Start inference in separate thread
            self.inference_thread = threading.Thread(
                target=self._run_inference_thread, 
                args=(input_data,),
                daemon=True
            )
            self.inference_thread.start()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to run inference: {e}")
            return False
    
    def get_response(self, timeout: float = 60.0) -> str:
        """Get complete response from inference"""
        try:
            # Wait for generation to complete
            if not self.response_event.wait(timeout=timeout):
                self.logger.warning("Response generation timed out")
                return ""
            
            # Check for errors
            with self.thread_lock:
                if self.generation_error:
                    self.logger.error("Generation completed with error")
                    return ""
                
                # Join response parts
                response = ''.join(self.current_response)
                self.logger.info(f"Generated response length: {len(response)} characters")
                return response
                
        except Exception as e:
            self.logger.error(f"Error getting response: {e}")
            return ""
    
    def get_response_stream(self, timeout: float = 60.0) -> Generator[str, None, None]:
        """Get streaming response from inference"""
        start_time = time.time()
        last_length = 0
        
        while not self.generation_complete:
            if time.time() - start_time > timeout:
                self.logger.warning("Response generation timed out")
                break
            
            with self.thread_lock:
                current_length = len(self.current_response)
                if current_length > last_length:
                    for i in range(last_length, current_length):
                        yield self.current_response[i]
                    last_length = current_length
                
                if self.generation_error:
                    self.logger.error("Generation error during streaming")
                    break
            
            time.sleep(0.01)
    
    def release(self):
        """Release RKLLM resources"""
        try:
            # Wait for inference thread to complete
            if self.inference_thread and self.inference_thread.is_alive():
                self.logger.info("Waiting for inference thread to complete...")
                self.inference_thread.join(timeout=5.0)
            
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
    
    def __del__(self):
        """Destructor to ensure resources are released"""
        self.release()

"""
RKLLM Integration for Gemma3
Handles the C++ RKLLM library integration with NPU optimization.
Final version with corrected ctypes structures based on user's working example
to resolve the segmentation fault.
"""

import os
import sys
import ctypes
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Generator

# ==============================================================================
# KORREKTE CTYPES-STRUKTUREN (EXAKTE KOPIE AUS DEM FUNKTIONIERENDEN BEISPIEL)
# ==============================================================================

class RKLLMInputMode(ctypes.c_int):
    RKLLM_INPUT_PROMPT = 0

class RKLLMInferMode(ctypes.c_int):
    RKLLM_INFER_GENERATE = 0

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
        ("extend_param", RKLLMExtendParam),
    ]

class RKLLMInputUnion(ctypes.Union):
    _fields_ = [("prompt_input", ctypes.c_char_p)]

class RKLLMInput(ctypes.Structure):
    _fields_ = [("input_mode", ctypes.c_int), ("input_data", RKLLMInputUnion)]

class RKLLMInferParam(ctypes.Structure):
    _fields_ = [("mode", ctypes.c_int), ("lora_params", ctypes.c_void_p), ("prompt_cache_params", ctypes.c_void_p), ("keep_history", ctypes.c_int)]

class RKLLMResultLastHiddenLayer(ctypes.Structure):
    _fields_ = [("hidden_states", ctypes.POINTER(ctypes.c_float)), ("embd_size", ctypes.c_int), ("num_tokens", ctypes.c_int)]

class RKLLMResultLogits(ctypes.Structure):
    _fields_ = [("logits", ctypes.POINTER(ctypes.c_float)), ("vocab_size", ctypes.c_int), ("num_tokens", ctypes.c_int)]

class RKLLMResult(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("token_id", ctypes.c_int),
        ("last_hidden_layer", RKLLMResultLastHiddenLayer),
        ("logits", RKLLMResultLogits)
    ]

# ==============================================================================
# ENDE KORRIGIERTE STRUKTUREN
# ==============================================================================

# Handle type
RKLLM_Handle_t = ctypes.c_void_p

class RKLLMRuntime:
    """RKLLM Runtime wrapper for C++ library"""

    def __init__(self, lib_path: Optional[str] = None):
        self.logger = logging.getLogger("gemma3-rkllm.rkllm")
        if lib_path is None: lib_path = self._find_rkllm_library()
        try:
            self.lib = ctypes.CDLL(lib_path)
            self.logger.info(f"RKLLM library loaded from: {lib_path}")
        except Exception as e:
            raise RuntimeError(f"Could not load RKLLM library: {e}")
        
        self.callback_type = ctypes.CFUNCTYPE(None, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int)
        self.callback = self.callback_type(self._callback_impl)
        self._setup_function_signatures()
        
        self.handle = None
        self.is_initialized = False
        self.current_response = []
        self.generation_complete = False

    def _find_rkllm_library(self) -> str:
        for path in ["./lib/librkllmrt.so", "/usr/local/lib/librkllmrt.so", "/usr/lib/librkllmrt.so"]:
            if os.path.exists(path): return path
        raise FileNotFoundError("RKLLM library (librkllmrt.so) not found.")

    def _setup_function_signatures(self):
        self.lib.rkllm_init.argtypes = [ctypes.POINTER(RKLLM_Handle_t), ctypes.POINTER(RKLLMParam), self.callback_type]
        self.lib.rkllm_init.restype = ctypes.c_int
        self.lib.rkllm_run.argtypes = [RKLLM_Handle_t, ctypes.POINTER(RKLLMInput), ctypes.POINTER(RKLLMInferParam), ctypes.c_void_p]
        self.lib.rkllm_run.restype = ctypes.c_int
        self.lib.rkllm_destroy.argtypes = [RKLLM_Handle_t]
        self.lib.rkllm_destroy.restype = ctypes.c_int
        if hasattr(self.lib, 'rkllm_set_chat_template'):
            self.lib.rkllm_set_chat_template.argtypes = [RKLLM_Handle_t, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
            self.lib.rkllm_set_chat_template.restype = ctypes.c_int
        
    def _callback_impl(self, result: ctypes.POINTER(RKLLMResult), userdata: ctypes.c_void_p, state: ctypes.c_int):
        try:
            if state == 0 and result and result.contents and result.contents.text:
                self.current_response.append(result.contents.text.decode('utf-8', errors='ignore'))
            elif state == 2:
                self.generation_complete = True
            elif state == 3:
                self.logger.error("RKLLM callback reported an error state (3).")
                self.generation_complete = True
        except Exception as e:
            self.logger.error(f"Error in callback: {e}", exc_info=True)
            self.generation_complete = True

    def initialize(self, model_path: str, config: Dict[str, Any]) -> bool:
        try:
            if self.is_initialized: return True
            p = RKLLMParam()
            p.model_path = model_path.encode('utf-8')
            p.max_context_len = config.get('max_context_len', 16384)
            p.max_new_tokens = -1
            p.skip_special_token = True
            p.top_k = 1
            p.top_p = float(config.get('top_p', 0.9))
            p.temperature = float(config.get('temperature', 0.8))
            p.repeat_penalty = float(config.get('repeat_penalty', 1.1))
            p.n_keep = -1
            p.extend_param.enabled_cpus_num = 4
            p.extend_param.enabled_cpus_mask = 0xF0

            self.handle = RKLLM_Handle_t()
            ret = self.lib.rkllm_init(ctypes.byref(self.handle), ctypes.byref(p), self.callback)
            if ret != 0: raise RuntimeError(f"RKLLM initialization failed with code: {ret}")
            self.is_initialized = True
            self.logger.info("RKLLM initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize RKLLM: {e}")
            self.release()
            return False

    def set_chat_template(self, system_prompt: str, user_prefix: str, user_postfix: str) -> bool:
        if not hasattr(self.lib, 'rkllm_set_chat_template'): return False
        if not self.is_initialized: return False
        try:
            ret = self.lib.rkllm_set_chat_template(self.handle, system_prompt.encode('utf-8'), user_prefix.encode('utf-8'), user_postfix.encode('utf-8'))
            if ret == 0:
                self.logger.info("Chat template set successfully in RKLLM library.")
                return True
            self.logger.error(f"rkllm_set_chat_template failed with code: {ret}")
            return False
        except Exception as e:
            self.logger.error(f"Exception while setting chat template: {e}")
            return False

    def run_inference(self, prompt: str) -> bool:
        try:
            if not self.is_initialized: raise RuntimeError("RKLLM not initialized")
            self.current_response, self.generation_complete = [], False
            
            rkllm_input = RKLLMInput()
            rkllm_input.input_mode = RKLLMInputMode.RKLLM_INPUT_PROMPT
            rkllm_input.input_data.prompt_input = prompt.encode('utf-8')

            rkllm_infer_params = RKLLMInferParam()
            rkllm_infer_params.mode = RKLLMInferMode.RKLLM_INFER_GENERATE
            
            ret = self.lib.rkllm_run(self.handle, ctypes.byref(rkllm_input), ctypes.byref(rkllm_infer_params), None)
            if ret != 0: raise RuntimeError(f"RKLLM inference failed with code: {ret}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to run inference: {e}")
            return False
    
    def get_response(self, timeout: float = 60.0) -> str:
        import time
        start_time = time.time()
        while not self.generation_complete and time.time() - start_time < timeout:
            time.sleep(0.01)
        if not self.generation_complete: self.logger.warning("Response generation timed out")
        return ''.join(self.current_response)
    
    def get_response_stream(self, timeout: float = 60.0) -> Generator[str, None, None]:
        import time
        start_time, last_length = time.time(), 0
        while not self.generation_complete and time.time() - start_time < timeout:
            time.sleep(0.01)
            current_length = len(self.current_response)
            if current_length > last_length:
                for i in range(last_length, current_length): yield self.current_response[i]
                last_length = current_length
        if not self.generation_complete: self.logger.warning("Response stream timed out")
    
    def release(self):
        try:
            if self.is_initialized and self.handle:
                self.lib.rkllm_destroy(self.handle)
                self.handle, self.is_initialized = None, False
                self.logger.info("RKLLM resources released successfully")
        except Exception as e:
            self.logger.error(f"Error releasing RKLLM: {e}")
    
    def __del__(self):
        self.release()


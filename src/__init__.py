"""
Gemma3 RKLLM Package
A complete solution for running Gemma3 multimodal models on Rockchip NPU
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"
__license__ = "MIT"

from .gemma3_model import Gemma3Model
from .rkllm_runtime import RKLLMRuntime
from .npu_optimizer import NPUOptimizer
from .image_processor import ImageProcessor
from .config_manager import ConfigManager
from .api_handlers import APIHandlers
from .logger import setup_logger, PerformanceLogger, DebugLogger
from .utils import (
    validate_request,
    handle_errors,
    validate_image_format,
    encode_image_to_base64,
    decode_base64_image,
    generate_unique_id,
    format_file_size,
    get_system_info,
    sanitize_filename,
    validate_model_name,
    create_error_response,
    create_success_response,
    RateLimiter
)

__all__ = [
    # Main classes
    'Gemma3Model',
    'RKLLMRuntime',
    'NPUOptimizer',
    'ImageProcessor',
    'ConfigManager',
    'APIHandlers',
    
    # Logging
    'setup_logger',
    'PerformanceLogger',
    'DebugLogger',
    
    # Utilities
    'validate_request',
    'handle_errors',
    'validate_image_format',
    'encode_image_to_base64',
    'decode_base64_image',
    'generate_unique_id',
    'format_file_size',
    'get_system_info',
    'sanitize_filename',
    'validate_model_name',
    'create_error_response',
    'create_success_response',
    'RateLimiter'
]


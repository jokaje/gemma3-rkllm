"""
Logging setup for Gemma3 RKLLM
Provides structured logging with file rotation and console output
"""

import os
import sys
import logging
import logging.handlers
from pathlib import Path
from typing import Optional
import configparser


def setup_logger(config: configparser.ConfigParser, name: str = "gemma3-rkllm") -> logging.Logger:
    """
    Setup logger with file and console handlers
    
    Args:
        config: Configuration parser instance
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    
    # Get logging configuration
    log_level = config.get('logging', 'level', fallback='INFO').upper()
    log_file = config.get('logging', 'file', fallback='./logs/gemma3-rkllm.log')
    max_size = config.get('logging', 'max_size', fallback='10MB')
    backup_count = config.getint('logging', 'backup_count', fallback=5)
    
    # Convert max_size to bytes
    max_bytes = _parse_size(max_size)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level, logging.INFO))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    try:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, log_level, logging.INFO))
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
        
    except Exception as e:
        logger.warning(f"Could not setup file logging: {e}")
    
    # Error handler for critical errors
    if log_level == 'DEBUG':
        error_handler = logging.StreamHandler(sys.stderr)
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        logger.addHandler(error_handler)
    
    return logger


def _parse_size(size_str: str) -> int:
    """
    Parse size string to bytes
    
    Args:
        size_str: Size string like '10MB', '1GB', etc.
        
    Returns:
        Size in bytes
    """
    size_str = size_str.upper().strip()
    
    if size_str.endswith('KB'):
        return int(float(size_str[:-2]) * 1024)
    elif size_str.endswith('MB'):
        return int(float(size_str[:-2]) * 1024 * 1024)
    elif size_str.endswith('GB'):
        return int(float(size_str[:-2]) * 1024 * 1024 * 1024)
    else:
        # Assume bytes
        return int(size_str)


class PerformanceLogger:
    """Logger for performance metrics"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def log_inference_time(self, model_name: str, prompt_length: int, 
                          generation_time: float, tokens_generated: int):
        """Log inference performance metrics"""
        tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
        
        self.logger.info(
            f"Inference completed - Model: {model_name}, "
            f"Prompt length: {prompt_length}, "
            f"Generation time: {generation_time:.2f}s, "
            f"Tokens generated: {tokens_generated}, "
            f"Tokens/sec: {tokens_per_second:.2f}"
        )
    
    def log_model_load_time(self, model_name: str, load_time: float):
        """Log model loading time"""
        self.logger.info(f"Model loaded - Name: {model_name}, Load time: {load_time:.2f}s")
    
    def log_memory_usage(self, memory_mb: float, gpu_memory_mb: Optional[float] = None):
        """Log memory usage"""
        msg = f"Memory usage - RAM: {memory_mb:.1f}MB"
        if gpu_memory_mb:
            msg += f", GPU: {gpu_memory_mb:.1f}MB"
        self.logger.info(msg)
    
    def log_request_metrics(self, endpoint: str, response_time: float, status_code: int):
        """Log API request metrics"""
        self.logger.info(
            f"API request - Endpoint: {endpoint}, "
            f"Response time: {response_time:.3f}s, "
            f"Status: {status_code}"
        )


class DebugLogger:
    """Logger for debug information"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def log_request_data(self, endpoint: str, data: dict):
        """Log request data for debugging"""
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Request to {endpoint}: {data}")
    
    def log_model_config(self, config: dict):
        """Log model configuration"""
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Model configuration: {config}")
    
    def log_image_processing(self, image_info: dict):
        """Log image processing information"""
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Image processing: {image_info}")
    
    def log_tokenization(self, text: str, tokens: list):
        """Log tokenization details"""
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Tokenization - Text length: {len(text)}, Tokens: {len(tokens)}")


def get_logger(name: str = "gemma3-rkllm") -> logging.Logger:
    """Get logger instance"""
    return logging.getLogger(name)


def log_system_info(logger: logging.Logger):
    """Log system information at startup"""
    import platform
    import psutil
    
    logger.info("=== System Information ===")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"CPU cores: {psutil.cpu_count()}")
    logger.info(f"Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    logger.info("=== Starting Gemma3 RKLLM Server ===")


def log_startup_banner(logger: logging.Logger):
    """Log startup banner"""
    banner = """
    ╔═══════════════════════════════════════╗
    ║           Gemma3 RKLLM Server         ║
    ║     Multimodal AI on Rockchip NPU     ║
    ║              Version 1.0              ║
    ╚═══════════════════════════════════════╝
    """
    for line in banner.split('\n'):
        if line.strip():
            logger.info(line)


# Context manager for performance logging
class LogExecutionTime:
    """Context manager to log execution time"""
    
    def __init__(self, logger: logging.Logger, operation: str):
        self.logger = logger
        self.operation = operation
        self.start_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        self.logger.debug(f"Starting {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        execution_time = time.time() - self.start_time
        if exc_type:
            self.logger.error(f"{self.operation} failed after {execution_time:.3f}s: {exc_val}")
        else:
            self.logger.info(f"{self.operation} completed in {execution_time:.3f}s")


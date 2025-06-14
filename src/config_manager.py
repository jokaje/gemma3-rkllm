"""
Configuration Manager for Gemma3 RKLLM
Handles loading and managing configuration from multiple sources
"""

import os
import configparser
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigManager:
    """Manages configuration for Gemma3 RKLLM server"""
    
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config_loaded = False
    
    def load_config(self, config_path: Optional[str] = None) -> configparser.ConfigParser:
        """
        Load configuration from file with fallback to defaults
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            ConfigParser instance
        """
        # Default configuration
        self._set_defaults()
        
        # Load from file if provided
        if config_path and os.path.exists(config_path):
            try:
                self.config.read(config_path)
                print(f"Configuration loaded from: {config_path}")
            except Exception as e:
                print(f"Warning: Could not load config from {config_path}: {e}")
        
        # Load from environment variables
        self._load_from_env()
        
        self.config_loaded = True
        return self.config
    
    def _set_defaults(self):
        """Set default configuration values"""
        
        # Server configuration
        self.config.add_section('server')
        self.config.set('server', 'host', '0.0.0.0')
        self.config.set('server', 'port', '8080')
        self.config.set('server', 'debug', 'false')
        self.config.set('server', 'cors_enabled', 'true')
        self.config.set('server', 'max_workers', '4')
        
        # Model configuration
        self.config.add_section('model')
        self.config.set('model', 'default_model', 'gemma3-4b')
        self.config.set('model', 'max_context_length', '128000')
        self.config.set('model', 'default_temperature', '0.7')
        self.config.set('model', 'max_new_tokens', '2048')
        self.config.set('model', 'models_dir', './models')
        
        # Multimodal configuration
        self.config.add_section('multimodal')
        self.config.set('multimodal', 'max_image_size', '2048')
        self.config.set('multimodal', 'supported_formats', 'jpg,jpeg,png,webp,bmp')
        self.config.set('multimodal', 'image_quality', '85')
        self.config.set('multimodal', 'enable_pan_and_scan', 'true')
        
        # NPU configuration
        self.config.add_section('npu')
        self.config.set('npu', 'platform', 'rk3588')
        self.config.set('npu', 'frequency_mode', 'performance')
        self.config.set('npu', 'enable_optimization', 'true')
        self.config.set('npu', 'memory_pool_size', '8192')
        
        # Logging configuration
        self.config.add_section('logging')
        self.config.set('logging', 'level', 'INFO')
        self.config.set('logging', 'file', './logs/gemma3-rkllm.log')
        self.config.set('logging', 'max_size', '10MB')
        self.config.set('logging', 'backup_count', '5')
        
        # API configuration
        self.config.add_section('api')
        self.config.set('api', 'enable_ollama_compatibility', 'true')
        self.config.set('api', 'enable_openai_compatibility', 'false')
        self.config.set('api', 'rate_limit', '100')
        self.config.set('api', 'timeout', '300')
        
        # Security configuration
        self.config.add_section('security')
        self.config.set('security', 'api_key_required', 'false')
        self.config.set('security', 'allowed_origins', '*')
        self.config.set('security', 'max_request_size', '50MB')
    
    def _load_from_env(self):
        """Load configuration from environment variables"""
        env_mappings = {
            'GEMMA3_HOST': ('server', 'host'),
            'GEMMA3_PORT': ('server', 'port'),
            'GEMMA3_DEBUG': ('server', 'debug'),
            'GEMMA3_MODELS_DIR': ('model', 'models_dir'),
            'GEMMA3_MAX_CONTEXT': ('model', 'max_context_length'),
            'GEMMA3_TEMPERATURE': ('model', 'default_temperature'),
            'GEMMA3_LOG_LEVEL': ('logging', 'level'),
            'GEMMA3_LOG_FILE': ('logging', 'file'),
            'GEMMA3_NPU_PLATFORM': ('npu', 'platform'),
            'GEMMA3_MAX_IMAGE_SIZE': ('multimodal', 'max_image_size'),
        }
        
        for env_var, (section, option) in env_mappings.items():
            value = os.getenv(env_var)
            if value:
                self.config.set(section, option, value)
    
    def get_models_dir(self) -> Path:
        """Get models directory as Path object"""
        models_dir = self.config.get('model', 'models_dir', fallback='./models')
        return Path(models_dir).resolve()
    
    def get_logs_dir(self) -> Path:
        """Get logs directory as Path object"""
        log_file = self.config.get('logging', 'file', fallback='./logs/gemma3-rkllm.log')
        return Path(log_file).parent.resolve()
    
    def is_debug_mode(self) -> bool:
        """Check if debug mode is enabled"""
        return self.config.getboolean('server', 'debug', fallback=False)
    
    def get_npu_config(self) -> Dict[str, Any]:
        """Get NPU configuration as dictionary"""
        return {
            'platform': self.config.get('npu', 'platform', fallback='rk3588'),
            'frequency_mode': self.config.get('npu', 'frequency_mode', fallback='performance'),
            'enable_optimization': self.config.getboolean('npu', 'enable_optimization', fallback=True),
            'memory_pool_size': self.config.getint('npu', 'memory_pool_size', fallback=8192)
        }
    
    def get_multimodal_config(self) -> Dict[str, Any]:
        """Get multimodal configuration as dictionary"""
        return {
            'max_image_size': self.config.getint('multimodal', 'max_image_size', fallback=2048),
            'supported_formats': self.config.get('multimodal', 'supported_formats', fallback='jpg,jpeg,png,webp,bmp').split(','),
            'image_quality': self.config.getint('multimodal', 'image_quality', fallback=85),
            'enable_pan_and_scan': self.config.getboolean('multimodal', 'enable_pan_and_scan', fallback=True)
        }
    
    def validate_config(self) -> bool:
        """Validate configuration values"""
        try:
            # Check required directories
            models_dir = self.get_models_dir()
            logs_dir = self.get_logs_dir()
            
            # Create directories if they don't exist
            models_dir.mkdir(parents=True, exist_ok=True)
            logs_dir.mkdir(parents=True, exist_ok=True)
            
            # Validate port range
            port = self.config.getint('server', 'port')
            if not (1 <= port <= 65535):
                raise ValueError(f"Invalid port number: {port}")
            
            # Validate temperature range
            temperature = self.config.getfloat('model', 'default_temperature')
            if not (0.0 <= temperature <= 2.0):
                raise ValueError(f"Invalid temperature: {temperature}")
            
            return True
            
        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return False
    
    def save_config(self, config_path: str):
        """Save current configuration to file"""
        try:
            with open(config_path, 'w') as f:
                self.config.write(f)
            print(f"Configuration saved to: {config_path}")
        except Exception as e:
            print(f"Error saving configuration: {e}")
    
    def update_config(self, section: str, option: str, value: str):
        """Update configuration value"""
        if not self.config.has_section(section):
            self.config.add_section(section)
        self.config.set(section, option, value)
    
    def get_config_dict(self) -> Dict[str, Dict[str, Any]]:
        """Get configuration as nested dictionary"""
        config_dict = {}
        for section in self.config.sections():
            config_dict[section] = dict(self.config.items(section))
        return config_dict


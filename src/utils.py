"""
Utility functions for Gemma3 RKLLM
Common helper functions and validators
"""

import os
import re
import json
import base64
import hashlib
import mimetypes
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import time
import psutil
from functools import wraps


def validate_request(data: Dict[str, Any], required_fields: List[str]) -> bool:
    """
    Validate request data contains required fields
    
    Args:
        data: Request data dictionary
        required_fields: List of required field names
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(data, dict):
        return False
    
    for field in required_fields:
        if field not in data:
            return False
    
    return True


def handle_errors(func):
    """
    Decorator for error handling in API endpoints
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            from flask import jsonify
            import logging
            
            logger = logging.getLogger("gemma3-rkllm")
            logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            
            return jsonify({
                "error": "Internal server error",
                "message": str(e)
            }), 500
    
    return wrapper


def validate_image_format(file_data: bytes) -> Tuple[bool, Optional[str]]:
    """
    Validate image format and return MIME type
    
    Args:
        file_data: Image file data
        
    Returns:
        Tuple of (is_valid, mime_type)
    """
    # Check file signatures
    signatures = {
        b'\xff\xd8\xff': 'image/jpeg',
        b'\x89PNG\r\n\x1a\n': 'image/png',
        b'GIF87a': 'image/gif',
        b'GIF89a': 'image/gif',
        b'RIFF': 'image/webp',  # WebP starts with RIFF
        b'BM': 'image/bmp'
    }
    
    for signature, mime_type in signatures.items():
        if file_data.startswith(signature):
            return True, mime_type
    
    return False, None


def encode_image_to_base64(image_path: Union[str, Path]) -> str:
    """
    Encode image file to base64 string
    
    Args:
        image_path: Path to image file
        
    Returns:
        Base64 encoded string
    """
    with open(image_path, 'rb') as f:
        image_data = f.read()
    
    return base64.b64encode(image_data).decode('utf-8')


def decode_base64_image(base64_string: str) -> bytes:
    """
    Decode base64 string to image bytes
    
    Args:
        base64_string: Base64 encoded image string
        
    Returns:
        Image bytes
    """
    # Remove data URL prefix if present
    if base64_string.startswith('data:'):
        base64_string = base64_string.split(',', 1)[1]
    
    return base64.b64decode(base64_string)


def generate_unique_id(prefix: str = "") -> str:
    """
    Generate unique ID with optional prefix
    
    Args:
        prefix: Optional prefix for the ID
        
    Returns:
        Unique ID string
    """
    timestamp = str(int(time.time() * 1000))
    random_part = hashlib.md5(os.urandom(16)).hexdigest()[:8]
    
    if prefix:
        return f"{prefix}_{timestamp}_{random_part}"
    else:
        return f"{timestamp}_{random_part}"


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human readable format
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def get_system_info() -> Dict[str, Any]:
    """
    Get system information
    
    Returns:
        Dictionary with system information
    """
    try:
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        return {
            "cpu_count": psutil.cpu_count(),
            "cpu_percent": cpu_percent,
            "memory_total": memory.total,
            "memory_available": memory.available,
            "memory_percent": memory.percent,
            "disk_usage": dict(psutil.disk_usage('/')._asdict())
        }
    except Exception:
        return {}


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe file operations
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove or replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove leading/trailing spaces and dots
    filename = filename.strip(' .')
    
    # Ensure filename is not empty
    if not filename:
        filename = "unnamed_file"
    
    return filename


def validate_model_name(model_name: str) -> bool:
    """
    Validate model name format
    
    Args:
        model_name: Model name to validate
        
    Returns:
        True if valid, False otherwise
    """
    # Allow alphanumeric, hyphens, underscores, and dots
    pattern = r'^[a-zA-Z0-9._-]+$'
    return bool(re.match(pattern, model_name)) and len(model_name) <= 100


def parse_content_type(content_type: str) -> Tuple[str, Dict[str, str]]:
    """
    Parse content type header
    
    Args:
        content_type: Content-Type header value
        
    Returns:
        Tuple of (main_type, parameters)
    """
    parts = content_type.split(';')
    main_type = parts[0].strip()
    
    params = {}
    for part in parts[1:]:
        if '=' in part:
            key, value = part.split('=', 1)
            params[key.strip()] = value.strip().strip('"')
    
    return main_type, params


def create_error_response(error_code: str, message: str, details: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Create standardized error response
    
    Args:
        error_code: Error code identifier
        message: Human readable error message
        details: Optional additional details
        
    Returns:
        Error response dictionary
    """
    response = {
        "error": {
            "code": error_code,
            "message": message,
            "timestamp": time.time()
        }
    }
    
    if details:
        response["error"]["details"] = details
    
    return response


def create_success_response(data: Any, message: Optional[str] = None) -> Dict[str, Any]:
    """
    Create standardized success response
    
    Args:
        data: Response data
        message: Optional success message
        
    Returns:
        Success response dictionary
    """
    response = {
        "success": True,
        "data": data,
        "timestamp": time.time()
    }
    
    if message:
        response["message"] = message
    
    return response


def measure_execution_time(func):
    """
    Decorator to measure function execution time
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        # Add execution time to result if it's a dictionary
        if isinstance(result, dict):
            result['execution_time'] = execution_time
        
        return result
    
    return wrapper


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split list into chunks of specified size
    
    Args:
        lst: List to chunk
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def safe_json_loads(json_string: str, default: Any = None) -> Any:
    """
    Safely parse JSON string with fallback
    
    Args:
        json_string: JSON string to parse
        default: Default value if parsing fails
        
    Returns:
        Parsed JSON or default value
    """
    try:
        return json.loads(json_string)
    except (json.JSONDecodeError, TypeError):
        return default


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if necessary
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def get_file_hash(file_path: Union[str, Path], algorithm: str = 'md5') -> str:
    """
    Calculate file hash
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm (md5, sha1, sha256)
        
    Returns:
        Hex digest of file hash
    """
    hash_obj = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)
    
    return hash_obj.hexdigest()


def is_port_available(port: int, host: str = 'localhost') -> bool:
    """
    Check if port is available
    
    Args:
        port: Port number to check
        host: Host to check on
        
    Returns:
        True if port is available, False otherwise
    """
    import socket
    
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            result = sock.connect_ex((host, port))
            return result != 0
    except Exception:
        return False


def format_duration(seconds: float) -> str:
    """
    Format duration in human readable format
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


class RateLimiter:
    """Simple rate limiter implementation"""
    
    def __init__(self, max_requests: int, time_window: int):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = {}
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed for identifier"""
        current_time = time.time()
        
        # Clean old entries
        self.requests = {
            k: v for k, v in self.requests.items()
            if current_time - v['first_request'] < self.time_window
        }
        
        if identifier not in self.requests:
            self.requests[identifier] = {
                'count': 1,
                'first_request': current_time
            }
            return True
        
        request_info = self.requests[identifier]
        if request_info['count'] < self.max_requests:
            request_info['count'] += 1
            return True
        
        return False


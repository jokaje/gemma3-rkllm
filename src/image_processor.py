"""
Image Processor for Gemma3 Multimodal
Handles image preprocessing for Gemma3 vision capabilities
"""

import os
import io
import base64
import logging
import hashlib
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import configparser

import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import cv2

from .utils import validate_image_format, generate_unique_id, format_file_size


class ImageProcessor:
    """Image processor for Gemma3 multimodal input"""
    
    def __init__(self, config: configparser.ConfigParser):
        self.config = config
        self.logger = logging.getLogger("gemma3-rkllm.image")
        
        # Configuration
        self.max_image_size = config.getint('multimodal', 'max_image_size', fallback=2048)
        self.supported_formats = config.get('multimodal', 'supported_formats', fallback='jpg,jpeg,png,webp,bmp').split(',')
        self.image_quality = config.getint('multimodal', 'image_quality', fallback=85)
        self.enable_pan_and_scan = config.getboolean('multimodal', 'enable_pan_and_scan', fallback=True)
        
        # Gemma3 vision model parameters (based on SigLIP)
        self.target_size = (384, 384)  # SigLIP input size
        self.mean = [0.5, 0.5, 0.5]    # SigLIP normalization
        self.std = [0.5, 0.5, 0.5]
        
        # Cache for processed images
        self.image_cache = {}
        self.cache_max_size = 100
    
    def process_image_bytes(self, image_data: bytes, image_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process image from bytes
        
        Args:
            image_data: Raw image bytes
            image_id: Optional image identifier
            
        Returns:
            Dictionary with processed image information
        """
        try:
            # Validate image format
            is_valid, mime_type = validate_image_format(image_data)
            if not is_valid:
                raise ValueError("Unsupported image format")
            
            # Generate ID if not provided
            if image_id is None:
                image_id = generate_unique_id("img")
            
            # Check cache
            image_hash = hashlib.md5(image_data).hexdigest()
            if image_hash in self.image_cache:
                self.logger.debug(f"Using cached image: {image_id}")
                cached_result = self.image_cache[image_hash].copy()
                cached_result['id'] = image_id
                return cached_result
            
            # Load image
            image = Image.open(io.BytesIO(image_data))
            
            # Process image
            processed_result = self._process_pil_image(image, image_id, mime_type)
            processed_result['original_size'] = len(image_data)
            processed_result['hash'] = image_hash
            
            # Cache result
            self._cache_image(image_hash, processed_result)
            
            return processed_result
            
        except Exception as e:
            self.logger.error(f"Error processing image bytes: {e}")
            raise
    
    def process_base64_image(self, base64_string: str, image_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process image from base64 string
        
        Args:
            base64_string: Base64 encoded image
            image_id: Optional image identifier
            
        Returns:
            Dictionary with processed image information
        """
        try:
            # Remove data URL prefix if present
            if base64_string.startswith('data:'):
                base64_string = base64_string.split(',', 1)[1]
            
            # Decode base64
            image_data = base64.b64decode(base64_string)
            
            return self.process_image_bytes(image_data, image_id)
            
        except Exception as e:
            self.logger.error(f"Error processing base64 image: {e}")
            raise
    
    def process_image_file(self, image_path: Union[str, Path], image_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process image from file path
        
        Args:
            image_path: Path to image file
            image_id: Optional image identifier
            
        Returns:
            Dictionary with processed image information
        """
        try:
            image_path = Path(image_path)
            
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Read image data
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            # Use filename as ID if not provided
            if image_id is None:
                image_id = image_path.stem
            
            result = self.process_image_bytes(image_data, image_id)
            result['source_path'] = str(image_path)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing image file {image_path}: {e}")
            raise
    
    def _process_pil_image(self, image: Image.Image, image_id: str, mime_type: str) -> Dict[str, Any]:
        """
        Process PIL Image object
        
        Args:
            image: PIL Image object
            image_id: Image identifier
            mime_type: MIME type of original image
            
        Returns:
            Dictionary with processed image information
        """
        try:
            original_size = image.size
            original_mode = image.mode
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply preprocessing
            processed_image = self._preprocess_for_gemma3(image)
            
            # Generate embeddings/features (placeholder for actual vision model)
            features = self._extract_image_features(processed_image)
            
            # Convert to base64 for storage/transmission
            processed_base64 = self._image_to_base64(processed_image)
            
            result = {
                'id': image_id,
                'original_size': original_size,
                'processed_size': processed_image.size,
                'original_mode': original_mode,
                'mime_type': mime_type,
                'format': image.format or 'unknown',
                'features': features,
                'processed_base64': processed_base64,
                'preprocessing_applied': True
            }
            
            self.logger.debug(f"Processed image {image_id}: {original_size} -> {processed_image.size}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing PIL image: {e}")
            raise
    
    def _preprocess_for_gemma3(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for Gemma3 vision model (SigLIP-based)
        
        Args:
            image: PIL Image object
            
        Returns:
            Preprocessed PIL Image
        """
        try:
            # Resize while maintaining aspect ratio
            if self.enable_pan_and_scan:
                processed_image = self._smart_resize(image)
            else:
                processed_image = self._simple_resize(image)
            
            # Apply image enhancements if needed
            processed_image = self._apply_enhancements(processed_image)
            
            return processed_image
            
        except Exception as e:
            self.logger.error(f"Error in Gemma3 preprocessing: {e}")
            raise
    
    def _smart_resize(self, image: Image.Image) -> Image.Image:
        """
        Smart resize with aspect ratio preservation and cropping
        
        Args:
            image: PIL Image object
            
        Returns:
            Resized PIL Image
        """
        try:
            target_w, target_h = self.target_size
            original_w, original_h = image.size
            
            # Calculate scaling factor
            scale_w = target_w / original_w
            scale_h = target_h / original_h
            scale = max(scale_w, scale_h)  # Scale to fill target size
            
            # Resize image
            new_w = int(original_w * scale)
            new_h = int(original_h * scale)
            resized_image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # Center crop to target size
            left = (new_w - target_w) // 2
            top = (new_h - target_h) // 2
            right = left + target_w
            bottom = top + target_h
            
            cropped_image = resized_image.crop((left, top, right, bottom))
            
            return cropped_image
            
        except Exception as e:
            self.logger.error(f"Error in smart resize: {e}")
            return self._simple_resize(image)
    
    def _simple_resize(self, image: Image.Image) -> Image.Image:
        """
        Simple resize to target size
        
        Args:
            image: PIL Image object
            
        Returns:
            Resized PIL Image
        """
        return image.resize(self.target_size, Image.Resampling.LANCZOS)
    
    def _apply_enhancements(self, image: Image.Image) -> Image.Image:
        """
        Apply image enhancements for better model performance
        
        Args:
            image: PIL Image object
            
        Returns:
            Enhanced PIL Image
        """
        try:
            # Auto-contrast for better feature extraction
            enhanced_image = ImageOps.autocontrast(image, cutoff=1)
            
            # Slight sharpening
            enhancer = ImageEnhance.Sharpness(enhanced_image)
            enhanced_image = enhancer.enhance(1.1)
            
            return enhanced_image
            
        except Exception as e:
            self.logger.warning(f"Error applying enhancements: {e}")
            return image
    
    def _extract_image_features(self, image: Image.Image) -> Dict[str, Any]:
        """
        Extract image features for Gemma3 processing
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary with image features
        """
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Normalize to [0, 1]
            img_array = img_array.astype(np.float32) / 255.0
            
            # Apply SigLIP normalization
            for i in range(3):
                img_array[:, :, i] = (img_array[:, :, i] - self.mean[i]) / self.std[i]
            
            # Flatten for transmission (in practice, this would be processed by vision model)
            flattened = img_array.flatten()
            
            # Basic image statistics
            stats = {
                'mean_rgb': [float(np.mean(img_array[:, :, i])) for i in range(3)],
                'std_rgb': [float(np.std(img_array[:, :, i])) for i in range(3)],
                'shape': img_array.shape,
                'dtype': str(img_array.dtype)
            }
            
            return {
                'normalized_array': flattened.tolist(),  # For JSON serialization
                'statistics': stats,
                'preprocessing': 'siglip_compatible'
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting image features: {e}")
            return {}
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """
        Convert PIL Image to base64 string
        
        Args:
            image: PIL Image object
            
        Returns:
            Base64 encoded string
        """
        try:
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=self.image_quality)
            img_bytes = buffer.getvalue()
            return base64.b64encode(img_bytes).decode('utf-8')
            
        except Exception as e:
            self.logger.error(f"Error converting image to base64: {e}")
            return ""
    
    def _cache_image(self, image_hash: str, result: Dict[str, Any]):
        """
        Cache processed image result
        
        Args:
            image_hash: Hash of original image
            result: Processing result to cache
        """
        try:
            # Remove oldest entries if cache is full
            if len(self.image_cache) >= self.cache_max_size:
                oldest_key = next(iter(self.image_cache))
                del self.image_cache[oldest_key]
            
            # Cache result (without ID to allow reuse)
            cached_result = result.copy()
            if 'id' in cached_result:
                del cached_result['id']
            
            self.image_cache[image_hash] = cached_result
            
        except Exception as e:
            self.logger.warning(f"Error caching image: {e}")
    
    def create_image_prompt(self, text_prompt: str, image_info: Dict[str, Any]) -> str:
        """
        Create multimodal prompt combining text and image information
        
        Args:
            text_prompt: Text part of the prompt
            image_info: Processed image information
            
        Returns:
            Combined prompt string
        """
        try:
            # Gemma3 multimodal prompt format
            img_start = "<start_of_image>"
            img_end = "<end_of_image>"
            
            # Include image metadata in prompt
            image_desc = f"Image {image_info.get('id', 'unknown')} ({image_info.get('processed_size', 'unknown size')})"
            
            combined_prompt = f"{img_start}{image_desc}{img_end}\n{text_prompt}"
            
            return combined_prompt
            
        except Exception as e:
            self.logger.error(f"Error creating image prompt: {e}")
            return text_prompt
    
    def batch_process_images(self, image_sources: List[Union[str, bytes, Dict]], 
                           batch_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Process multiple images in batch
        
        Args:
            image_sources: List of image sources (paths, bytes, or dicts with data)
            batch_id: Optional batch identifier
            
        Returns:
            List of processed image results
        """
        try:
            if batch_id is None:
                batch_id = generate_unique_id("batch")
            
            results = []
            
            for i, source in enumerate(image_sources):
                try:
                    image_id = f"{batch_id}_img_{i}"
                    
                    if isinstance(source, str):
                        # File path or base64
                        if source.startswith('data:') or len(source) > 1000:
                            result = self.process_base64_image(source, image_id)
                        else:
                            result = self.process_image_file(source, image_id)
                    elif isinstance(source, bytes):
                        # Raw bytes
                        result = self.process_image_bytes(source, image_id)
                    elif isinstance(source, dict) and 'data' in source:
                        # Dictionary with image data
                        if 'base64' in source:
                            result = self.process_base64_image(source['base64'], image_id)
                        elif 'bytes' in source:
                            result = self.process_image_bytes(source['bytes'], image_id)
                        else:
                            raise ValueError("Invalid image source dictionary")
                    else:
                        raise ValueError(f"Unsupported image source type: {type(source)}")
                    
                    results.append(result)
                    
                except Exception as e:
                    self.logger.error(f"Error processing image {i} in batch {batch_id}: {e}")
                    # Add error result
                    results.append({
                        'id': f"{batch_id}_img_{i}",
                        'error': str(e),
                        'processed': False
                    })
            
            self.logger.info(f"Batch processed {len(results)} images (batch_id: {batch_id})")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in batch processing: {e}")
            return []
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get image processing statistics
        
        Returns:
            Dictionary with processing statistics
        """
        return {
            'cache_size': len(self.image_cache),
            'cache_max_size': self.cache_max_size,
            'target_size': self.target_size,
            'max_image_size': self.max_image_size,
            'supported_formats': self.supported_formats,
            'preprocessing_enabled': True
        }
    
    def clear_cache(self):
        """Clear image processing cache"""
        self.image_cache.clear()
        self.logger.info("Image cache cleared")


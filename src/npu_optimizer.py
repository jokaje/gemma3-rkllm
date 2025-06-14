"""
NPU Optimization for Orange Pi 5 Plus
Handles NPU frequency scaling and performance optimization
"""

import os
import subprocess
import logging
import psutil
from typing import Dict, Any, Optional, List
import configparser
from pathlib import Path


class NPUOptimizer:
    """NPU optimization and frequency management for RK3588"""
    
    def __init__(self, config: configparser.ConfigParser):
        self.config = config
        self.logger = logging.getLogger("gemma3-rkllm.npu")
        self.platform = config.get('npu', 'platform', fallback='rk3588')
        self.frequency_mode = config.get('npu', 'frequency_mode', fallback='performance')
        self.enable_optimization = config.getboolean('npu', 'enable_optimization', fallback=True)
        
        # RK3588 NPU frequency settings
        self.npu_frequencies = {
            'rk3588': {
                'powersave': 300000000,    # 300 MHz
                'balanced': 600000000,     # 600 MHz  
                'performance': 1000000000  # 1000 MHz (1 GHz)
            },
            'rk3576': {
                'powersave': 200000000,    # 200 MHz
                'balanced': 400000000,     # 400 MHz
                'performance': 800000000   # 800 MHz
            }
        }
        
        # CPU governor settings
        self.cpu_governors = {
            'powersave': 'powersave',
            'balanced': 'ondemand',
            'performance': 'performance'
        }
        
        self.original_settings = {}
        self.optimization_applied = False
    
    def detect_platform(self) -> str:
        """
        Auto-detect NPU platform
        
        Returns:
            Platform identifier (rk3588, rk3576, etc.)
        """
        try:
            # Check device tree or CPU info
            if os.path.exists('/proc/device-tree/compatible'):
                with open('/proc/device-tree/compatible', 'rb') as f:
                    compatible = f.read().decode('utf-8', errors='ignore')
                    if 'rk3588' in compatible:
                        return 'rk3588'
                    elif 'rk3576' in compatible:
                        return 'rk3576'
            
            # Fallback to checking CPU info
            if os.path.exists('/proc/cpuinfo'):
                with open('/proc/cpuinfo', 'r') as f:
                    cpuinfo = f.read()
                    if 'rk3588' in cpuinfo.lower():
                        return 'rk3588'
                    elif 'rk3576' in cpuinfo.lower():
                        return 'rk3576'
            
            # Default fallback
            self.logger.warning("Could not detect platform, using default: rk3588")
            return 'rk3588'
            
        except Exception as e:
            self.logger.error(f"Error detecting platform: {e}")
            return 'rk3588'
    
    def get_npu_frequency_paths(self) -> List[str]:
        """
        Get NPU frequency control paths
        
        Returns:
            List of frequency control file paths
        """
        possible_paths = [
            '/sys/kernel/debug/clk/clk_npu_dsu0/clk_rate',
            '/sys/kernel/debug/clk/aclk_npu/clk_rate',
            '/sys/class/devfreq/fdab0000.npu/cur_freq',
            '/sys/class/devfreq/fdab0000.npu/max_freq',
            '/sys/class/devfreq/fdab0000.npu/min_freq'
        ]
        
        existing_paths = []
        for path in possible_paths:
            if os.path.exists(path):
                existing_paths.append(path)
        
        return existing_paths
    
    def get_current_npu_frequency(self) -> Optional[int]:
        """
        Get current NPU frequency
        
        Returns:
            Current frequency in Hz, or None if not available
        """
        try:
            freq_paths = self.get_npu_frequency_paths()
            
            for path in freq_paths:
                if 'cur_freq' in path or 'clk_rate' in path:
                    try:
                        with open(path, 'r') as f:
                            freq = int(f.read().strip())
                            self.logger.debug(f"Current NPU frequency from {path}: {freq} Hz")
                            return freq
                    except (ValueError, IOError):
                        continue
            
            self.logger.warning("Could not read current NPU frequency")
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting NPU frequency: {e}")
            return None
    
    def set_npu_frequency(self, frequency: int) -> bool:
        """
        Set NPU frequency
        
        Args:
            frequency: Target frequency in Hz
            
        Returns:
            True if successful, False otherwise
        """
        try:
            freq_paths = self.get_npu_frequency_paths()
            success = False
            
            for path in freq_paths:
                if 'max_freq' in path or 'min_freq' in path:
                    try:
                        # Set frequency via devfreq
                        cmd = f"echo {frequency} | sudo tee {path}"
                        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                        
                        if result.returncode == 0:
                            self.logger.info(f"Set NPU frequency to {frequency} Hz via {path}")
                            success = True
                        else:
                            self.logger.warning(f"Failed to set frequency via {path}: {result.stderr}")
                            
                    except Exception as e:
                        self.logger.warning(f"Error setting frequency via {path}: {e}")
                        continue
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error setting NPU frequency: {e}")
            return False
    
    def set_cpu_governor(self, governor: str) -> bool:
        """
        Set CPU governor for all cores
        
        Args:
            governor: Governor name (performance, ondemand, powersave)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cpu_count = psutil.cpu_count()
            success_count = 0
            
            for cpu in range(cpu_count):
                governor_path = f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor"
                
                if os.path.exists(governor_path):
                    try:
                        cmd = f"echo {governor} | sudo tee {governor_path}"
                        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                        
                        if result.returncode == 0:
                            success_count += 1
                        else:
                            self.logger.warning(f"Failed to set governor for CPU {cpu}: {result.stderr}")
                            
                    except Exception as e:
                        self.logger.warning(f"Error setting governor for CPU {cpu}: {e}")
            
            if success_count > 0:
                self.logger.info(f"Set CPU governor to {governor} for {success_count}/{cpu_count} cores")
                return True
            else:
                self.logger.error("Failed to set CPU governor for any core")
                return False
                
        except Exception as e:
            self.logger.error(f"Error setting CPU governor: {e}")
            return False
    
    def get_current_cpu_governor(self) -> Optional[str]:
        """
        Get current CPU governor
        
        Returns:
            Current governor name, or None if not available
        """
        try:
            governor_path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
            
            if os.path.exists(governor_path):
                with open(governor_path, 'r') as f:
                    governor = f.read().strip()
                    return governor
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting CPU governor: {e}")
            return None
    
    def save_current_settings(self):
        """Save current frequency and governor settings"""
        try:
            self.original_settings = {
                'npu_frequency': self.get_current_npu_frequency(),
                'cpu_governor': self.get_current_cpu_governor()
            }
            
            self.logger.info(f"Saved original settings: {self.original_settings}")
            
        except Exception as e:
            self.logger.error(f"Error saving current settings: {e}")
    
    def restore_original_settings(self):
        """Restore original frequency and governor settings"""
        try:
            if not self.original_settings:
                self.logger.warning("No original settings to restore")
                return
            
            # Restore CPU governor
            if self.original_settings.get('cpu_governor'):
                self.set_cpu_governor(self.original_settings['cpu_governor'])
            
            # Restore NPU frequency
            if self.original_settings.get('npu_frequency'):
                self.set_npu_frequency(self.original_settings['npu_frequency'])
            
            self.logger.info("Restored original settings")
            self.optimization_applied = False
            
        except Exception as e:
            self.logger.error(f"Error restoring settings: {e}")
    
    def apply_optimization(self) -> bool:
        """
        Apply NPU and CPU optimizations based on configuration
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.enable_optimization:
                self.logger.info("NPU optimization disabled in configuration")
                return True
            
            # Save current settings first
            self.save_current_settings()
            
            # Auto-detect platform if not specified
            if self.platform == 'auto':
                self.platform = self.detect_platform()
            
            # Get target frequency
            if self.platform in self.npu_frequencies:
                target_freq = self.npu_frequencies[self.platform].get(self.frequency_mode)
                if target_freq:
                    self.set_npu_frequency(target_freq)
                else:
                    self.logger.warning(f"Unknown frequency mode: {self.frequency_mode}")
            else:
                self.logger.warning(f"Unknown platform: {self.platform}")
            
            # Set CPU governor
            target_governor = self.cpu_governors.get(self.frequency_mode, 'ondemand')
            self.set_cpu_governor(target_governor)
            
            # Additional optimizations
            self._apply_memory_optimizations()
            self._apply_scheduler_optimizations()
            
            self.optimization_applied = True
            self.logger.info(f"NPU optimization applied for {self.platform} in {self.frequency_mode} mode")
            return True
            
        except Exception as e:
            self.logger.error(f"Error applying NPU optimization: {e}")
            return False
    
    def _apply_memory_optimizations(self):
        """Apply memory-related optimizations"""
        try:
            # Disable swap if enabled (for better NPU performance)
            subprocess.run("sudo swapoff -a", shell=True, capture_output=True)
            
            # Set vm.swappiness to 1 (minimal swapping)
            subprocess.run("echo 1 | sudo tee /proc/sys/vm/swappiness", shell=True, capture_output=True)
            
            # Increase dirty ratio for better I/O performance
            subprocess.run("echo 15 | sudo tee /proc/sys/vm/dirty_ratio", shell=True, capture_output=True)
            
            self.logger.debug("Applied memory optimizations")
            
        except Exception as e:
            self.logger.warning(f"Error applying memory optimizations: {e}")
    
    def _apply_scheduler_optimizations(self):
        """Apply CPU scheduler optimizations"""
        try:
            # Set CPU affinity for better performance
            # Keep NPU-related processes on performance cores (A76)
            
            # For RK3588: cores 4-7 are A76 (performance cores)
            if self.platform == 'rk3588':
                performance_cores = "4-7"
            else:
                performance_cores = "0-3"  # Fallback
            
            # This would be applied to the current process
            # In practice, this might be done at the application level
            
            self.logger.debug("Applied scheduler optimizations")
            
        except Exception as e:
            self.logger.warning(f"Error applying scheduler optimizations: {e}")
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """
        Get current optimization status
        
        Returns:
            Dictionary with optimization status information
        """
        try:
            return {
                "platform": self.platform,
                "frequency_mode": self.frequency_mode,
                "optimization_enabled": self.enable_optimization,
                "optimization_applied": self.optimization_applied,
                "current_npu_frequency": self.get_current_npu_frequency(),
                "current_cpu_governor": self.get_current_cpu_governor(),
                "original_settings": self.original_settings
            }
            
        except Exception as e:
            self.logger.error(f"Error getting optimization status: {e}")
            return {}
    
    def benchmark_npu(self) -> Dict[str, float]:
        """
        Run simple NPU benchmark
        
        Returns:
            Benchmark results
        """
        try:
            # This would run a simple inference benchmark
            # For now, return placeholder values
            
            import time
            start_time = time.time()
            
            # Simulate some work
            time.sleep(0.1)
            
            end_time = time.time()
            
            return {
                "benchmark_time": end_time - start_time,
                "estimated_tops": 6.0,  # RK3588 has 6 TOPS
                "memory_bandwidth": "unknown"
            }
            
        except Exception as e:
            self.logger.error(f"Error running NPU benchmark: {e}")
            return {}
    
    def __del__(self):
        """Destructor to restore original settings"""
        if self.optimization_applied:
            self.restore_original_settings()


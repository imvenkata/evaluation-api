"""
Caching utilities for the synthetic ground truth generation pipeline.
Provides fast, persistent caching for expensive operations.
"""

import hashlib
import json
import os
import pickle
import logging
from pathlib import Path
from typing import Any, Optional, Union, Dict
from functools import wraps

logger = logging.getLogger(__name__)

class SimpleCache:
    """Simple file-based cache with automatic key generation."""
    
    def __init__(self, cache_dir: str = "./cache", namespace: str = "default"):
        self.cache_dir = Path(cache_dir) / namespace
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _create_key(self, *args, **kwargs) -> str:
        """Create a stable cache key from arguments."""
        # Combine args and kwargs into a stable representation
        content = {
            'args': args,
            'kwargs': kwargs
        }
        
        # Convert to JSON string (sorted for consistency)
        json_str = json.dumps(content, sort_keys=True, default=str)
        
        # Create SHA256 hash
        hash_obj = hashlib.sha256(json_str.encode('utf-8'))
        return hash_obj.hexdigest()[:16]  # Use first 16 chars for shorter filenames
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        cache_file = self.cache_dir / f"{key}.pkl"
        
        if not cache_file.exists():
            return None
            
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except (pickle.PickleError, EOFError, FileNotFoundError) as e:
            logger.warning(f"Failed to load cache file {cache_file}: {e}")
            # Remove corrupted cache file
            try:
                cache_file.unlink()
            except FileNotFoundError:
                pass
            return None
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        cache_file = self.cache_dir / f"{key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
        except (pickle.PickleError, OSError) as e:
            logger.warning(f"Failed to save cache file {cache_file}: {e}")
    
    def get_or_compute(self, compute_func, *args, **kwargs) -> Any:
        """Get from cache or compute and store."""
        cache_key = self._create_key(*args, **kwargs)
        
        # Try to get from cache first
        result = self.get(cache_key)
        if result is not None:
            logger.debug(f"Cache hit for key {cache_key}")
            return result
        
        # Cache miss - compute the result
        logger.debug(f"Cache miss for key {cache_key}, computing...")
        result = compute_func(*args, **kwargs)
        
        # Store in cache
        self.set(cache_key, result)
        return result
    
    def clear(self) -> None:
        """Clear all cache files."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
            except FileNotFoundError:
                pass
        logger.info(f"Cleared cache directory: {self.cache_dir}")
    
    def size_info(self) -> Dict[str, Union[int, str]]:
        """Get cache size information."""
        files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in files)
        
        return {
            "file_count": len(files),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "cache_dir": str(self.cache_dir)
        }


def cached(cache_dir: str = "./cache", namespace: str = "default"):
    """Decorator to automatically cache function results."""
    def decorator(func):
        cache = SimpleCache(cache_dir, namespace)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return cache.get_or_compute(func, *args, **kwargs)
        
        # Add cache management methods to the function
        wrapper.cache_clear = cache.clear
        wrapper.cache_info = cache.size_info
        
        return wrapper
    return decorator


# Convenience functions for specific use cases
def create_chunk_cache_key(chunk_text: str, config_hash: str) -> str:
    """Create cache key for chunk-related operations."""
    content = f"{chunk_text}|{config_hash}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]

def create_config_hash(config) -> str:
    """Create a hash representing current configuration."""
    # Extract relevant config values
    config_dict = {
        'MIN_TOKEN_LENGTH': getattr(config, 'MIN_TOKEN_LENGTH', None),
        'MAX_TOKEN_LENGTH': getattr(config, 'MAX_TOKEN_LENGTH', None),
        'DUPLICATE_COSINE_SIM': getattr(config, 'DUPLICATE_COSINE_SIM', None),
        'TEMPERATURE': getattr(config, 'TEMPERATURE', None),
        'MAX_TOKENS': getattr(config, 'MAX_TOKENS', None),
        'QUERY_TYPES': getattr(config, 'QUERY_TYPES', None),
    }
    
    json_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]

def create_prompt_cache_key(golden_text: str, distractor_texts: list, query_type: str, 
                          temperature: float, max_tokens: int) -> str:
    """Create cache key for LLM prompt results."""
    content = {
        'golden': golden_text,
        'distractors': sorted(distractor_texts),  # Sort for consistency
        'type': query_type,
        'temp': temperature,
        'tokens': max_tokens
    }
    
    json_str = json.dumps(content, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]

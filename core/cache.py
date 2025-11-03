"""
Caching module for FinGPT service - MINIMAL FIX
Only fixes the generator caching bug, nothing else changed
"""
import hashlib
import json
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Optional, Iterator
import logging

from cachetools import TTLCache

logger = logging.getLogger(__name__)

# ===================================
# CACHE CONFIGURATION
# ===================================

# Financial data cache (1 hour TTL, max 500 items)
financial_data_cache = TTLCache(maxsize=500, ttl=3600)

# Forecast cache (30 minutes TTL, max 200 items)
forecast_cache = TTLCache(maxsize=200, ttl=1800)

# Model predictions cache (1 hour TTL, max 100 items)
prediction_cache = TTLCache(maxsize=100, ttl=3600)


# ===================================
# CACHE KEY GENERATOR
# ===================================

def generate_cache_key(*args, **kwargs) -> str:
    """
    Generate a unique cache key from function arguments
    """
    # Combine args and kwargs into a deterministic string
    key_dict = {
        'args': args,
        'kwargs': sorted(kwargs.items())
    }
    key_string = json.dumps(key_dict, sort_keys=True)
    
    # Generate hash
    cache_key = hashlib.md5(key_string.encode()).hexdigest()
    return cache_key


# ===================================
# CACHE DECORATORS
# ===================================

def cached_financial_data(func: Callable) -> Callable:
    """
    Decorator to cache financial data fetching
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Generate cache key
        cache_key = generate_cache_key(func.__name__, *args, **kwargs)
        
        # Check cache
        if cache_key in financial_data_cache:
            logger.info(f"✅ Cache HIT for financial data: {cache_key[:8]}...")
            return financial_data_cache[cache_key]
        
        # Cache miss - fetch data
        logger.info(f"❌ Cache MISS for financial data: {cache_key[:8]}...")
        result = func(*args, **kwargs)
        
        # Store in cache
        financial_data_cache[cache_key] = result
        logger.info(f"💾 Cached financial data: {cache_key[:8]}...")
        
        return result
    
    return wrapper


def cached_forecast(func: Callable) -> Callable:
    """
    🔥 FIXED: Properly cache generator functions
    
    The bug: Generators can only be consumed once. Caching the generator object
    meant the second request got an exhausted generator.
    
    The fix: Consume the generator, cache the full text, then yield from cache.
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Iterator[str]:
        cache_key = generate_cache_key(func.__name__, *args, **kwargs)
        
        # Check cache
        if cache_key in forecast_cache:
            logger.info(f"✅ Cache HIT for forecast: {cache_key[:8]}...")
            cached_text = forecast_cache[cache_key]
            
            # Return cached text as generator
            stream = kwargs.get('stream', True)
            if stream:
                # Simulate streaming by yielding in small chunks
                chunk_size = 10
                for i in range(0, len(cached_text), chunk_size):
                    yield cached_text[i:i+chunk_size]
            else:
                # For non-streaming, yield all at once
                yield cached_text
            return
        
        # Cache miss - generate new
        logger.info(f"❌ Cache MISS for forecast: {cache_key[:8]}...")
        
        # Consume generator and cache the full text
        full_text = ""
        for chunk in func(*args, **kwargs):
            full_text += chunk
            yield chunk  # Still stream to caller in real-time
        
        # Cache the complete text
        forecast_cache[cache_key] = full_text
        logger.info(f"💾 Cached forecast: {cache_key[:8]}... ({len(full_text)} chars)")
    
    return wrapper


def cached_prediction(func: Callable) -> Callable:
    """
    Decorator to cache model predictions
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        cache_key = generate_cache_key(func.__name__, *args, **kwargs)
        
        if cache_key in prediction_cache:
            logger.info(f"✅ Cache HIT for prediction: {cache_key[:8]}...")
            return prediction_cache[cache_key]
        
        logger.info(f"❌ Cache MISS for prediction: {cache_key[:8]}...")
        result = func(*args, **kwargs)
        
        prediction_cache[cache_key] = result
        logger.info(f"💾 Cached prediction: {cache_key[:8]}...")
        
        return result
    
    return wrapper


# ===================================
# CACHE MANAGEMENT
# ===================================

def clear_all_caches():
    """Clear all caches"""
    financial_data_cache.clear()
    forecast_cache.clear()
    prediction_cache.clear()
    logger.info("🧹 Cleared all caches")


def get_cache_stats() -> dict:
    """Get cache statistics"""
    return {
        "financial_data": {
            "size": len(financial_data_cache),
            "maxsize": financial_data_cache.maxsize,
            "ttl": financial_data_cache.ttl,
        },
        "forecast": {
            "size": len(forecast_cache),
            "maxsize": forecast_cache.maxsize,
            "ttl": forecast_cache.ttl,
        },
        "prediction": {
            "size": len(prediction_cache),
            "maxsize": prediction_cache.maxsize,
            "ttl": prediction_cache.ttl,
        },
        "total_cached_items": (
            len(financial_data_cache) + 
            len(forecast_cache) + 
            len(prediction_cache)
        ),
    }


def clear_expired_entries():
    """Manually trigger expiration of old entries"""
    # TTLCache automatically expires entries, but we can trigger it
    _ = len(financial_data_cache)
    _ = len(forecast_cache)
    _ = len(prediction_cache)
    logger.info("🔄 Triggered cache expiration check")
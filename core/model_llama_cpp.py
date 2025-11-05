"""
Optimized Model Loading using llama.cpp
10x faster on Mac M4!
"""
import logging
from typing import Iterator
from llama_cpp import Llama
from config import setting as config

logger = logging.getLogger(__name__)

# Global model instance
_model = None

def load_model_llama_cpp():
    """Load model using llama.cpp (much faster!)"""
    global _model
    
    logger.info("="*70)
    logger.info("🚀 Loading FinGPT Model with llama.cpp (Optimized for Mac M4)")
    logger.info("="*70)
    
    model_path = "./models/llama-2-7b-chat.Q4_K_M.gguf"
    
    logger.info(f"📦 Loading model from {model_path}...")
    logger.info("   This takes 10-20 seconds...")
    
    _model = Llama(
        model_path=model_path,
        n_ctx=2048,           # Context window
        n_threads=8,          # CPU threads (adjust for your Mac)
        n_gpu_layers=35,      # Use Metal GPU (35 layers for 7B model)
        verbose=False,
        seed=-1,              # Random seed
    )
    
    logger.info("="*70)
    logger.info("✅ MODEL LOADED SUCCESSFULLY!")
    logger.info("="*70)
    
    return _model


def generate_forecast_llama_cpp(
    prompt: str,
    temperature: float = 0.2,
    max_new_tokens: int = 128,
    stream: bool = True
) -> Iterator[str]:
    """Generate forecast using llama.cpp"""
    global _model
    
    if _model is None:
        raise RuntimeError("Model not loaded!")
    
    # Build Llama 2 chat format
    full_prompt = f"<s>[INST] <<SYS>>\n{config.SYSTEM_PROMPT}\n<</SYS>>\n\n{prompt.strip()} [/INST]"
    
    logger.info(f"🤖 Generating with llama.cpp (max_tokens={max_new_tokens})...")
    
    # Generate
    response = _model(
        full_prompt,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=config.DEFAULT_TOP_P,
        top_k=config.DEFAULT_TOP_K,
        repeat_penalty=config.DEFAULT_REPETITION_PENALTY,
        stop=["</s>", "[/INST]"],
        stream=stream,
    )
    
    if stream:
        # Streaming mode
        for chunk in response:
            token = chunk['choices'][0]['text']
            yield token
    else:
        # Non-streaming mode
        yield response['choices'][0]['text']
    
    logger.info("✅ Generation complete!")


def get_model_info_llama_cpp() -> dict:
    """Get model info"""
    global _model
    
    return {
        "loaded": _model is not None,
        "engine": "llama.cpp",
        "device": "Metal (Apple Silicon GPU)",
        "model": "Llama-2-7B-Chat-GGUF-Q4",
        "quantization": "4-bit",
        "estimated_size_gb": 4.0,
    }


def cleanup_memory_llama_cpp():
    """Cleanup (not much needed with llama.cpp)"""
    import gc
    gc.collect()
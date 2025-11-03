"""
Optimized Model Loading and Inference
Supports 4-bit/8-bit quantization for Mac M4
"""
import torch
import gc
import logging
from typing import Iterator, Optional
from threading import Thread
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
)

# Try to import BitsAndBytesConfig, but don't fail if not available
try:
    from transformers import BitsAndBytesConfig
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    logger.warning("⚠️  bitsandbytes not available - quantization disabled")

from peft import PeftModel
from config import setting as config  

logger = logging.getLogger(__name__)

# Global model and tokenizer
_model = None
_tokenizer = None
_device = None


def get_optimal_device() -> torch.device:
    """Determine the best device for the model"""
    if config.USE_MPS_IF_AVAILABLE and torch.backends.mps.is_available():
        logger.info("✅ Using MPS (Apple Silicon)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        logger.info("✅ Using CUDA")
        return torch.device("cuda")
    else:
        logger.info("⚠️  Using CPU (slower)")
        return torch.device("cpu")


def get_quantization_config() -> Optional[BitsAndBytesConfig]:
    """
    Get quantization configuration for memory optimization
    
    4-bit: ~4GB memory, 3-5x faster
    8-bit: ~7GB memory, 2-3x faster
    None: ~14GB memory, baseline speed
    """
    if not BITSANDBYTES_AVAILABLE:
        if config.USE_4BIT_QUANTIZATION or config.USE_8BIT_QUANTIZATION:
            logger.warning("⚠️  Quantization requested but bitsandbytes not installed")
            logger.warning("   Install with: pip install bitsandbytes")
            logger.warning("   Falling back to FP16 mode")
        return None
    
    if config.USE_4BIT_QUANTIZATION:
        logger.info("🚀 Using 4-bit quantization (NF4)")
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,  # Double quantization for better compression
        )
    elif config.USE_8BIT_QUANTIZATION:
        logger.info("🚀 Using 8-bit quantization")
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
    else:
        logger.info("📦 Using full precision (FP16)")
        return None


def cleanup_memory():
    """Aggressive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def load_model_and_tokenizer():
    """
    Load model with optimizations for Mac M4
    
    Optimizations:
    - 4-bit quantization (optional)
    - Low CPU memory usage
    - Proper device mapping
    - Gradient disabled
    """
    global _model, _tokenizer, _device
    
    logger.info("="*70)
    logger.info("🚀 Loading FinGPT Model (Optimized for Mac M4)")
    logger.info("="*70)
    
    # Clean memory before loading
    cleanup_memory()
    
    # Determine device
    _device = get_optimal_device()
    
    # Load tokenizer
    logger.info(f"📝 Loading tokenizer from {config.BASE_MODEL_PATH}...")
    _tokenizer = AutoTokenizer.from_pretrained(
        config.BASE_MODEL_PATH,
        use_fast=True
    )
    _tokenizer.pad_token = _tokenizer.eos_token
    _tokenizer.padding_side = "left"
    logger.info("✅ Tokenizer loaded")
    
    # Get quantization config
    quant_config = get_quantization_config()
    
    # Model loading kwargs
    model_kwargs = {
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    
    # Add quantization if enabled
    if quant_config:
        model_kwargs["quantization_config"] = quant_config
        model_kwargs["device_map"] = "auto"
    else:
        # Manual dtype setting for non-quantized
        dtype = torch.float16 if config.MODEL_DTYPE == "float16" else torch.bfloat16
        model_kwargs["torch_dtype"] = dtype
    
    # Load base model
    logger.info(f"📦 Loading base model from {config.BASE_MODEL_PATH}...")
    logger.info(f"   This may take 30-90 seconds...")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL_PATH,
        **model_kwargs
    )
    
    # Move to device if not using quantization (quantization handles device_map)
    if not quant_config:
        logger.info(f"📦 Moving model to {_device}...")
        base_model = base_model.to(_device)
    
    logger.info("✅ Base model loaded")
    
    # Load LoRA weights if available
    if config.LORA_WEIGHTS_PATH:
        logger.info(f"🎯 Loading LoRA weights from {config.LORA_WEIGHTS_PATH}...")
        try:
            _model = PeftModel.from_pretrained(
                base_model,
                config.LORA_WEIGHTS_PATH,
                torch_dtype=torch.float16
            )
            logger.info("✅ LoRA weights loaded")
        except Exception as e:
            logger.warning(f"⚠️  Failed to load LoRA: {e}")
            logger.info("📌 Using base model only")
            _model = base_model
    else:
        logger.info("📌 No LoRA weights specified, using base model")
        _model = base_model
    
    # Set to eval mode and disable gradients
    _model.eval()
    for param in _model.parameters():
        param.requires_grad = False
    
    # Final cleanup
    cleanup_memory()
    
    logger.info("="*70)
    logger.info("✅ MODEL LOADED SUCCESSFULLY!")
    logger.info("="*70)
    
    return _model, _tokenizer


def generate_forecast(
    prompt: str,
    temperature: float = None,
    max_new_tokens: int = None,
    stream: bool = True
) -> Iterator[str]:
    """
    Generate forecast using the model
    
    Args:
        prompt: Input prompt
        temperature: Sampling temperature (default from config)
        max_new_tokens: Max tokens to generate (default from config)
        stream: Whether to stream output
        
    Yields:
        Generated text chunks
    """
    global _model, _tokenizer, _device
    
    if _model is None or _tokenizer is None:
        raise RuntimeError("Model not loaded. Call load_model_and_tokenizer() first.")
    
    # Use defaults from config if not specified
    temperature = temperature or config.DEFAULT_TEMPERATURE
    max_new_tokens = max_new_tokens or config.DEFAULT_MAX_NEW_TOKENS
    
    # Build full prompt with Llama format
    full_prompt = f"{config.B_INST} {config.B_SYS}{config.SYSTEM_PROMPT}{config.E_SYS}{prompt} {config.E_INST}"
    
    # Tokenize with truncation
    inputs = _tokenizer(
        full_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,  # Llama context window
        padding=False
    )
    
    # Move to device (if not using device_map="auto")
    if not config.USE_4BIT_QUANTIZATION and not config.USE_8BIT_QUANTIZATION:
        inputs = {k: v.to(_device) for k, v in inputs.items()}
    
    # Generation config
    generation_config = {
        "max_new_tokens": max_new_tokens,
        "do_sample": config.DO_SAMPLE,
        "temperature": temperature,
        "top_p": config.DEFAULT_TOP_P,
        "top_k": config.DEFAULT_TOP_K,
        "repetition_penalty": config.DEFAULT_REPETITION_PENALTY,
        "num_beams": config.NUM_BEAMS,
        "pad_token_id": _tokenizer.pad_token_id,
        "eos_token_id": _tokenizer.eos_token_id,
    }
    
    if stream:
        # Streaming generation
        streamer = TextIteratorStreamer(
            _tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=180.0
        )
        
        generation_kwargs = {**inputs, **generation_config, "streamer": streamer}
        
        # Run generation in separate thread
        thread = Thread(
            target=lambda: _model.generate(**generation_kwargs),
            daemon=True
        )
        thread.start()
        
        # Yield tokens as they're generated
        token_count = 0
        for text in streamer:
            token_count += 1
            yield text
        
        thread.join(timeout=10)
        
        if config.LOG_GENERATION_TIME:
            logger.info(f"✅ Generated {token_count} tokens")
    
    else:
        # Non-streaming generation
        with torch.no_grad():
            outputs = _model.generate(**inputs, **generation_config)
        
        # Decode and yield full text
        text = _tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        yield text
    
    # Cleanup
    if config.LOW_MEMORY_MODE:
        del inputs
        cleanup_memory()


def get_model_info() -> dict:
    """Get information about the loaded model"""
    global _model, _device
    
    if _model is None:
        return {"loaded": False}
    
    # Try to estimate model size
    try:
        param_count = sum(p.numel() for p in _model.parameters())
        param_size_gb = param_count * 2 / (1024**3)  # Assuming FP16
    except:
        param_count = "Unknown"
        param_size_gb = "Unknown"
    
    return {
        "loaded": True,
        "device": str(_device),
        "parameters": param_count,
        "estimated_size_gb": param_size_gb,
        "quantization": "4-bit" if config.USE_4BIT_QUANTIZATION else (
            "8-bit" if config.USE_8BIT_QUANTIZATION else "FP16"
        ),
        "lora_enabled": config.LORA_WEIGHTS_PATH is not None
    }
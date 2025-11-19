"""
Configuration for FinGPT Server
Centralized settings for easy optimization
"""
import os
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# API KEYS
# =============================================================================
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
if not FINNHUB_API_KEY:
    raise ValueError("FINNHUB_API_KEY not found in .env file")

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
BASE_MODEL_PATH = "meta-llama/Llama-2-7b-chat-hf"
LORA_WEIGHTS_PATH = None  # Set to path if you have LoRA weights

# 🚀 OPTIMIZATION FLAGS
# ⚠️  Note: 4-bit/8-bit requires bitsandbytes, which may not work on Mac M4
# If you get "No module named 'bitsandbytes'" error, set both to False
USE_4BIT_QUANTIZATION = False  # Set to True if bitsandbytes is installed (3-5x faster)
USE_8BIT_QUANTIZATION = False  # Alternative: 8-bit for better quality
USE_FLASH_ATTENTION = False  # Enable if available (requires flash-attn)

# If both above are False, server uses FP16 mode (slower but works on all systems)

# Model precision
MODEL_DTYPE = "bfloat16"  

# =============================================================================
# GENERATION PARAMETERS
# =============================================================================
# 🎯 Optimized for speed vs quality balance
DEFAULT_MAX_NEW_TOKENS = 1024  # Reduced from 1024 for speed
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 50  # Reduced from 50
DEFAULT_REPETITION_PENALTY = 1.1

# Advanced generation settings
NUM_BEAMS = 1  # Keep at 1 for streaming
DO_SAMPLE = True

# =============================================================================
# MEMORY MANAGEMENT
# =============================================================================
LOW_MEMORY_MODE = True  # Enable aggressive memory cleanup
MIN_FREE_MEMORY_GB = 2.0  # Minimum free memory before generation
ENABLE_MEMORY_WARNINGS = True

# PyTorch MPS settings for M4
USE_MPS_IF_AVAILABLE = True
MPS_ALLOCATOR_POLICY = "garbage_collection"

# =============================================================================
# CACHE CONFIGURATION
# =============================================================================
# Financial data cache (1 hour TTL)
FINANCIAL_CACHE_SIZE = 500
FINANCIAL_CACHE_TTL = 3600

# Forecast cache (30 minutes TTL)
FORECAST_CACHE_SIZE = 200
FORECAST_CACHE_TTL = 1800

ENABLE_CACHING = True

# =============================================================================
# DATA FETCHING
# =============================================================================
DEFAULT_PAST_WEEKS = 4
MAX_NEWS_ITEMS = 8  # Reduced from 10 to save prompt tokens
INCLUDE_FINANCIALS_DEFAULT = False

# =============================================================================
# SERVER CONFIGURATION
# =============================================================================
HOST = "0.0.0.0"
PORT = 8000
RELOAD = False  # Set True only in development

# CORS
CORS_ORIGINS = ["*"]  # In production, specify your frontend URL

# =============================================================================
# LLAMA 2 PROMPT TEMPLATE
# =============================================================================
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SYSTEM_PROMPT = """You are FinGPT, a knowledgeable and friendly financial advisor with Wall Street expertise.

YOUR PERSONALITY:
- Conversational and approachable - like talking to a smart friend
- Enthusiastic about finance but realistic about risks
- Use analogies and real-world examples
- Balance optimism with honest risk assessment

RESPONSE STYLE:
- Start with a brief, friendly acknowledgment
- Break down complex topics into digestible sections
- Use natural language - avoid robotic corporate speak
- Include specific numbers and data when available
- End with a key takeaway or insight

FOR STOCK ANALYSIS:
[Positive Developments]:
1. Specific recent development with numbers/dates
2. Another concrete positive

[Potential Concerns]:
1. Specific risk or challenge
2. Another concern with context

[Prediction & Analysis]:
Clear forecast with reasoning and key factors

FOR OTHER QUESTIONS:
Answer conversationally in 2-4 sentences. Use examples. Be specific, not generic.

IMPORTANT:
- Never use placeholders like "[Recent development]"
- Use consistent numbering (1, 2, 3... throughout)
- Be specific with data: "gained 15.3%" not "went up"
- Keep disclaimers brief and natural"""

# =============================================================================
# LOGGING
# =============================================================================
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# =============================================================================
# PERFORMANCE MONITORING
# =============================================================================
ENABLE_PERFORMANCE_LOGGING = True
LOG_GENERATION_TIME = True
LOG_MEMORY_USAGE = True
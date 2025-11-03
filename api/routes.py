import os
from dotenv import load_dotenv
import traceback

import yfinance as yf
import finnhub
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import torch

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")
if not FINNHUB_API_KEY:
    raise ValueError("FINNHUB_API_KEY not found in environment variables")

# Initialize Finnhub Client
finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

# Constants for Llama format
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

# System prompt
SYSTEM_PROMPT = """You are a seasoned stock market analyst. Your task is to list the positive developments and potential concerns for companies based on relevant news and basic financials from the past weeks, then provide an analysis and prediction for the companies' stock price movement for the upcoming week. Your answer format should be as follows:\n\n[Positive Developments]:\n1. ...\n\n[Potential Concerns]:\n1. ...\n\n[Prediction & Analysis]:\n...\n"""

# Global model and tokenizer
model = None
tokenizer = None

# Model paths - ADJUST THESE
BASE_MODEL_PATH = "meta-llama/Llama-2-7b-chat-hf"
LORA_WEIGHTS_PATH = None  # Set to path if you have LoRA weights

# Cache 
from fingpt_service.core.cache import (
    clear_all_caches,
    get_cache_stats,
)
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models
class ForecasterRequest(BaseModel):
    ticker: str
    end_date: str
    past_weeks: int = 4
    include_financials: bool = False
    temperature: float = 0.2
    stream: bool = True

@app.post("/v1/chat/completions")
async def chat_completions(req: ForecasterRequest):
    """Main endpoint for stock forecasting"""
    
    if model is None or tokenizer is None:
        return JSONResponse(
            {"error": "Model not loaded yet. Please wait..."}, 
            status_code=503
        )
    
    if req.temperature < 0 or req.temperature > 2:
        return JSONResponse(
            {"error": "Temperature must be between 0 and 2"}, 
            status_code=400
        )
    
    # Fetch data and create prompt
    try:
        logger.info(f"📊 Fetching data for {req.ticker}...")
        user_prompt = fetch_data_for_prompt(
            req.ticker, 
            req.end_date, 
            req.past_weeks, 
            req.include_financials
        )
    except Exception as e:
        logger.error(f"❌ Data fetch error: {e}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            {"error": f"Data fetch error: {str(e)}"}, 
            status_code=400
        )

    # Stream response
    if req.stream:
        def sse():
            try:
                logger.info(f"🤖 Generating forecast for {req.ticker}...")
                for chunk in gen_reply(user_prompt, req.temperature, stream=True):
                    escaped_chunk = chunk.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
                    yield f'data: {{"id":"chatcmpl-fingpt","object":"chat.completion.chunk","choices":[{{"delta":{{"content":"{escaped_chunk}"}},"index":0}}]}}\n\n'
                yield 'data: [DONE]\n\n'
                logger.info(f"✅ Forecast complete for {req.ticker}")
            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                logger.error(f"❌ SSE Error: {error_msg}")
                logger.error(traceback.format_exc())
                yield f'data: {{"error":"{error_msg}"}}\n\n'
            finally:
                cleanup_memory()
        
        return StreamingResponse(sse(), media_type="text/event-stream")

    # Non-streaming response
    try:
        logger.info(f"🤖 Generating forecast for {req.ticker}...")
        text = "".join(list(gen_reply(user_prompt, req.temperature, stream=False)))
        logger.info(f"✅ Forecast complete for {req.ticker}")
        
        return JSONResponse({
            "id": "chatcmpl-fingpt",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop"
                }
            ]
        })
    except Exception as e:
        logger.error(f"❌ Error during generation: {e}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            {"error": f"{type(e).__name__}: {str(e)}"}, 
            status_code=500
        )
    finally:
        cleanup_memory()


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "FinGPT-Forecaster",
        "status": "running",
        "version": "1.3-debug-optimized",
        "optimized_for": "Mac M4 16GB - Enhanced Error Handling",
        "free_memory_gb": round(get_free_memory_gb(), 2)
    }

@app.get("/healthz")
async def healthz():
    """Health check endpoint"""
    model_loaded = model is not None and tokenizer is not None
    
    memory_info = {
        "free_memory_gb": round(get_free_memory_gb(), 2)
    }
    
    if torch.backends.mps.is_available():
        memory_info["device"] = "mps (Apple Silicon)"
    elif torch.cuda.is_available():
        memory_info["device"] = "cuda"
    else:
        memory_info["device"] = "cpu"
    
    return {
        "status": "ok" if model_loaded else "loading",
        "service": "FinGPT-Forecaster",
        "model_loaded": model_loaded,
        **memory_info
    }


@app.get("/v1/models")
async def list_models():
    """List available models"""
    return {
        "object": "list",
        "data": [
            {
                "id": "fingpt-forecaster-llama2-7b",
                "object": "model",
                "owned_by": "fingpt",
                "precision": "float16",
                "optimized_for": "Mac M4 16GB (enhanced debugging)"
            }
        ]
    }

# Cache endpoints
@app.get("/cache/stats")
async def cache_stats():
    """Get cache statistics"""
    stats = get_cache_stats()
    return {
        "status": "ok",
        "stats": stats,
    }

@app.post("/cache/clear")
async def clear_cache():
    """Clear all caches"""
    clear_all_caches()
    return {
        "status": "ok",
        "message": "All caches cleared",
    }

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global model, tokenizer
    logger.info("🛑 Shutting down...")
    del model
    del tokenizer
    cleanup_memory()
    logger.info("✅ Cleanup complete")
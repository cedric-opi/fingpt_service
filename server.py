"""
FinGPT Forecaster Server - Optimized for Mac M4
Clean, modular architecture with performance optimizations
"""
import logging
import time
from typing import Optional, List 
from typing import Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from config import setting as config
# from core.model import load_model_and_tokenizer, generate_forecast, get_model_info, cleanup_memory
from core.model_llama_cpp import (
    load_model_llama_cpp as load_model_and_tokenizer,
    generate_forecast_llama_cpp as generate_forecast,
    get_model_info_llama_cpp as get_model_info,
    cleanup_memory_llama_cpp as cleanup_memory,
)
from services.data_service import build_forecast_prompt
from core.cache import cached_forecast, cached_financial_data, get_cache_stats, clear_all_caches
from services.question_router import detect_question_type
from services import question_handlers
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="FinGPT Forecaster",
    description="Stock market forecasting using FinGPT (Optimized for Mac M4)",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class ForecastRequest(BaseModel):
    """Request model for stock forecasting"""
    ticker: str = Field(default="", description="Stock ticker symbol (optional if message provided)")
    message: str = Field(default="", description="User's question/message")
    end_date: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"), description="End date in YYYY-MM-DD format")
    past_weeks: int = Field(default=4, ge=1, le=12, description="Weeks to look back")
    include_financials: bool = Field(default=True, description="Include financial metrics")
    temperature: float = Field(default=0.2, ge=0.0, le=2.0, description="Sampling temperature")
    max_new_tokens: Optional[int] = Field(default=512, ge=50, le=1024, description="Max tokens to generate")
    stream: bool = Field(default=True, description="Stream the response")
    conversation_context: str = Field(default="")
    system_prompt: str = Field(default="")
    instructions: str = Field(default="")
    image_urls: Optional[List[str]] = None  # For future use with image inputs


class ForecastResponse(BaseModel):
    """Response model for non-streaming forecast"""
    id: str
    object: str
    choices: list
    
    class Config:
        schema_extra = {
            "example": {
                "id": "chatcmpl-fingpt",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "[Positive Developments]:\n1. ..."
                    },
                    "finish_reason": "stop"
                }]
            }
        }


# =============================================================================
# STARTUP / SHUTDOWN
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("🚀 Starting FinGPT Forecaster Server...")
    logger.info(f"   Quantization: {'4-bit' if config.USE_4BIT_QUANTIZATION else '8-bit' if config.USE_8BIT_QUANTIZATION else 'FP16'}")
    logger.info(f"   Caching: {'Enabled' if config.ENABLE_CACHING else 'Disabled'}")
    
    try:
        load_model_and_tokenizer()
        logger.info("✅ Server ready!")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down server...")
    cleanup_memory()
    logger.info("✅ Cleanup complete")


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """Root endpoint with server info"""
    model_info = get_model_info()
    
    return {
        "service": "FinGPT-Forecaster",
        "version": "2.0.0",
        "status": "running",
        "optimized_for": "Mac M4 16GB RAM",
        "model": model_info,
        "features": {
            "quantization": config.USE_4BIT_QUANTIZATION or config.USE_8BIT_QUANTIZATION,
            "caching": config.ENABLE_CACHING,
            "streaming": True
        }
    }


@app.get("/health")
@app.get("/healthz")
async def health_check():
    """Health check endpoint"""
    model_info = get_model_info()
    
    return {
        "status": "healthy" if model_info["loaded"] else "loading",
        "service": "FinGPT-Forecaster",
        "model_loaded": model_info["loaded"]
    }


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI compatibility)"""
    return {
        "object": "list",
        "data": [
            {
                "id": "fingpt-forecaster-llama2-7b",
                "object": "model",
                "owned_by": "fingpt",
                "precision": "4-bit" if config.USE_4BIT_QUANTIZATION else "FP16",
                "optimized_for": "Mac M4"
            }
        ]
    }


@app.post("/v1/chat/completions", response_model=None)
async def create_forecast(request: ForecastRequest):
    """
    Main endpoint - handles ANY financial question with optional images!
    """
    
    model_info = get_model_info()
    if not model_info["loaded"]:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded yet. Please wait..."
        )
    
    # Measure time
    start_time = time.time()
    
    try:
        # ✅ NEW: Analyze images if provided
        image_context = ""
        if request.image_urls and len(request.image_urls) > 0:
            logger.info(f"🖼️  Processing {len(request.image_urls)} image(s)...")
            from services.image_service import analyze_multiple_images
            image_context = analyze_multiple_images(request.image_urls)
            
            if image_context:
                logger.info(f"✅ Image analysis complete ({len(image_context)} chars)")
        
        # Detect question type from message or use ticker-based forecast
        if request.message:
            question_type, extracted_data = detect_question_type(request.message)
            logger.info(f"📊 Question type: {question_type}, Data: {extracted_data}")
        else:
            # Legacy: Direct ticker request
            question_type = "stock_forecast"
            extracted_data = {"ticker": request.ticker}
        
        # Build the prompt based on question type
        if question_type == "stock_forecast":
            ticker = extracted_data.get("ticker", request.ticker)
            
            if not ticker:
                # If no ticker but has image, treat as general analysis
                if image_context:
                    question_type = "general_chat"
                    extracted_data = {"query": request.message}
                else:
                    raise HTTPException(
                        status_code=400,
                        detail="No ticker found in message. Please specify a stock ticker (e.g., AAPL)"
                    )
        
        # Build prompt for each question type
        if question_type == "stock_forecast":
            # Build prompt (with caching if enabled)
            if config.ENABLE_CACHING:
                @cached_financial_data
                def get_cached_prompt(ticker, end_date, past_weeks, include_financials):
                    return build_forecast_prompt(ticker, end_date, past_weeks, include_financials)
                
                prompt = get_cached_prompt(
                    ticker,
                    request.end_date,
                    request.past_weeks,
                    request.include_financials
                )
            else:
                prompt = build_forecast_prompt(
                    ticker,
                    request.end_date,
                    request.past_weeks,
                    request.include_financials
                )
            
            # Add custom instructions if provided
            if request.instructions:
                prompt = f"{prompt}\n\nAdditional Instructions: {request.instructions}"
            
            # Add image context
            if image_context:
                prompt = f"{prompt}\n\n[Additional Visual Data from Chart]:\n{image_context}\n\nUse this chart data to enhance your analysis."
        
        elif question_type == "stock_comparison":
            tickers = extracted_data.get("tickers", [])
            if len(tickers) < 2:
                raise HTTPException(
                    status_code=400,
                    detail="Comparison requires at least 2 stock tickers"
                )
            
            ticker1, ticker2 = tickers[0], tickers[1]
            from services.question_handlers import create_comparison_prompt
            prompt = create_comparison_prompt(ticker1, ticker2, request.end_date)
            
            if image_context:
                prompt = f"{prompt}\n\n[Chart Analysis]:\n{image_context}"
        
        elif question_type == "market_analysis":
            from services.question_handlers import create_market_analysis_prompt
            prompt = create_market_analysis_prompt(extracted_data["query"])
            
            if image_context:
                prompt = f"{prompt}\n\n[Visual Market Data]:\n{image_context}"
        
        elif question_type == "financial_education":
            from services.question_handlers import create_education_prompt
            prompt = create_education_prompt(extracted_data["query"])
            
            if image_context:
                prompt = f"{prompt}\n\n[Visual Reference]:\n{image_context}"
        
        elif question_type == "investment_advice":
            from services.question_handlers import create_investment_advice_prompt
            prompt = create_investment_advice_prompt(extracted_data["query"])
            
            if image_context:
                prompt = f"{prompt}\n\n[Chart Analysis]:\n{image_context}"
        
        elif question_type == "general_chat":
            # Use system prompt from request OR fallback to default
            base_prompt = request.system_prompt or config.SYSTEM_PROMPT
            
            if request.instructions:
                base_prompt = f"{base_prompt}\n\nAdditional Instructions: {request.instructions}"
            
            # Build prompt with optional image context
            if image_context:
                prompt = f"{base_prompt}\n\n[Chart/Image Analysis]:\n{image_context}\n\nUser Question: {request.message}\n\nProvide a detailed analysis based on the chart data and question."
            else:
                if request.conversation_context:
                    prompt = f"{base_prompt}\n\nPrevious conversation context:\n{request.conversation_context}\n\nCurrent question: {request.message}\n\nAssistant:"
                else:
                    prompt = f"{base_prompt}\n\nUser: {request.message}\n\nAssistant:"
        
        else:
            # Fallback for any other type
            prompt = f"{config.SYSTEM_PROMPT}\n\nUser: {request.message}\n\nAssistant:"
        
        # ✅ UNIFIED GENERATION - All question types use this same code
        if request.stream:
            # Streaming response
            async def stream_generator():
                try:
                    # Use caching only for stock forecasts
                    if config.ENABLE_CACHING and question_type == "stock_forecast":
                        gen = cached_forecast(generate_forecast)(
                            prompt,
                            request.temperature,
                            request.max_new_tokens,
                            stream=True
                        )
                    else:
                        gen = generate_forecast(
                            prompt,
                            request.temperature,
                            request.max_new_tokens,
                            stream=True
                        )
                    
                    for chunk in gen:
                        escaped = chunk.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
                        yield f'data: {{"id":"chatcmpl-fingpt","object":"chat.completion.chunk","choices":[{{"delta":{{"content":"{escaped}"}},"index":0}}]}}\n\n'
                    
                    yield 'data: [DONE]\n\n'
                    
                    if config.LOG_GENERATION_TIME:
                        elapsed = time.time() - start_time
                        ticker_or_type = extracted_data.get("ticker", question_type)
                        logger.info(f"⏱️  Total time: {elapsed:.1f}s for {ticker_or_type}")
                
                except Exception as e:
                    error_msg = f"{type(e).__name__}: {str(e)}"
                    logger.error(f"❌ Stream error: {error_msg}")
                    yield f'data: {{"error":"{error_msg}"}}\n\n'
            
            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream"
            )
        
        else:
            # Non-streaming response
            if config.ENABLE_CACHING and question_type == "stock_forecast":
                gen = cached_forecast(generate_forecast)(
                    prompt,
                    request.temperature,
                    request.max_new_tokens,
                    stream=False
                )
            else:
                gen = generate_forecast(
                    prompt,
                    request.temperature,
                    request.max_new_tokens,
                    stream=False
                )
            
            text = "".join(gen)
            
            if config.LOG_GENERATION_TIME:
                elapsed = time.time() - start_time
                ticker_or_type = extracted_data.get("ticker", question_type)
                logger.info(f"⏱️  Total time: {elapsed:.1f}s for {ticker_or_type}")
            
            return JSONResponse({
                "id": "chatcmpl-fingpt",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": text
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": len(text.split()),
                    "total_tokens": len(prompt.split()) + len(text.split())
                }
            })
        
    except ValueError as e:
        logger.error(f"❌ Data error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"❌ Generation error: {e}")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")

# =============================================================================
# CACHE MANAGEMENT ENDPOINTS
# =============================================================================

@app.get("/cache/stats")
async def get_stats():
    """Get cache statistics"""
    if not config.ENABLE_CACHING:
        return {"error": "Caching is disabled"}
    
    stats = get_cache_stats()
    return {
        "status": "ok",
        "caching_enabled": True,
        "stats": stats
    }


@app.post("/cache/clear")
async def clear_cache():
    """Clear all caches"""
    if not config.ENABLE_CACHING:
        return {"error": "Caching is disabled"}
    
    clear_all_caches()
    return {
        "status": "ok",
        "message": "All caches cleared"
    }


# =============================================================================
# DEBUG ENDPOINTS
# =============================================================================

@app.get("/debug/model")
async def debug_model_info():
    """Get detailed model information"""
    return get_model_info()


@app.get("/debug/config")
async def debug_config():
    """Get current configuration (excluding secrets)"""
    return {
        "model": {
            "base_path": config.BASE_MODEL_PATH,
            "lora_path": config.LORA_WEIGHTS_PATH,
            "quantization": "4-bit" if config.USE_4BIT_QUANTIZATION else "8-bit" if config.USE_8BIT_QUANTIZATION else "FP16",
        },
        "generation": {
            "max_tokens": config.DEFAULT_MAX_NEW_TOKENS,
            "temperature": config.DEFAULT_TEMPERATURE,
            "top_p": config.DEFAULT_TOP_P,
            "top_k": config.DEFAULT_TOP_K,
        },
        "caching": {
            "enabled": config.ENABLE_CACHING,
            "financial_ttl": config.FINANCIAL_CACHE_TTL,
            "forecast_ttl": config.FORECAST_CACHE_TTL,
        },
        "memory": {
            "low_memory_mode": config.LOW_MEMORY_MODE,
            "use_mps": config.USE_MPS_IF_AVAILABLE,
        }
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "server:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.RELOAD
    )
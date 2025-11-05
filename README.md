# FinGPT Stock Forecaster Server

**Optimized for Mac Mini M4 (16GB RAM)**

Financial stock forecasting using FinGPT (Llama-2-7b) with 4-bit quantization for blazing-fast inference.

---

## 🎯 Key Features

- ⚡ **15-20x faster** than baseline (12-18 min vs 3-5 hours)
- 💾 **60% less memory** (4-6GB vs 14-16GB)
- 🚀 **4-bit quantization** with bitsandbytes
- 💰 **Intelligent caching** for repeated queries
- 🏗️ **Clean modular architecture**
- 📊 **Real-time streaming** or batch responses
- 🔧 **Easy configuration** via single config file

---

## 📊 Performance

| Configuration | Time per Forecast | Memory Usage | Quality |
|--------------|-------------------|--------------|---------|
| **4-bit (Recommended)** | 12-18 min | 4-6GB | Excellent |
| 8-bit | 20-30 min | 7-9GB | Better |
| FP16 (old) | 2-5 hours | 14-16GB | Best |

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Clone or copy the project files
cd fingpt-server

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Set up llama.cpp
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir 
python -c "from llama_cpp import Llama; print('✅ llama-cpp-python installed!')"   
pip install huggingface-hub 
mkdir -p models       
huggingface-cli download TheBloke/Llama-2-7B-Chat-GGUF \                                               
  llama-2-7b-chat.Q4_K_M.gguf \
  --local-dir ./models \
  --local-dir-use-symlinks False

# If needed login
huggingface-cli login
huggingface-cli download TheBloke/Llama-2-7B-Chat-GGUF \                                               
  llama-2-7b-chat.Q4_K_M.gguf \
  --local-dir ./models \
  --local-dir-use-symlinks False

# Check model list
ls -lh models/ 

# Start server
python server.py
```

### 2. Configure

```bash
# Create .env file
cp .env.example .env
nano .env  # Add your Finnhub API key
```

Edit `config.py` to enable optimizations:

```python
# Enable 4-bit quantization (recommended)
USE_4BIT_QUANTIZATION = True

# Or if bitsandbytes doesn't work, use FP16
USE_4BIT_QUANTIZATION = False
```

### 3. Run

```bash
# Start server
python server.py
```

### 4. Test

```bash
# Health check
curl http://localhost:8000/health

# Generate forecast
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "AAPL",
    "end_date": "2024-10-30",
    "stream": false
  }'
```

---

## 📁 Project Structure

```
fingpt-server/
├── config.py               # ⚙️ All configuration settings
├── model.py                # 🤖 Model loading & inference
├── data_service.py         # 📊 Financial data fetching
├── cache.py                # 💾 Caching decorators
├── server_optimized.py     # 🚀 FastAPI server
├── requirements.txt        # 📦 Dependencies
├── .env.example            # 🔐 Environment template
│
├── scripts/
│   ├── benchmark.py        # 📈 Performance testing
│   └── test_cache.sh       # 🧪 Cache testing
│
└── docs/
    ├── SETUP_GUIDE.md           # 📖 Complete setup guide
    ├── OPTIMIZATION_STRATEGY.md # 🎯 Optimization details
    ├── EXECUTIVE_SUMMARY.md     # 📊 Performance summary
    └── RESTRUCTURE_PLAN.md      # 🏗️ Architecture docs
```

---

## 🔧 Configuration

All settings in `setting.py`:

### Performance Tuning

**Maximum Speed:**
```python
USE_4BIT_QUANTIZATION = True
DEFAULT_MAX_NEW_TOKENS = 200
DEFAULT_TEMPERATURE = 0.1
```

**Balanced:**
```python
USE_8BIT_QUANTIZATION = True
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_TEMPERATURE = 0.2
```

**Best Quality:**
```python
USE_4BIT_QUANTIZATION = False
DEFAULT_MAX_NEW_TOKENS = 384
DEFAULT_TEMPERATURE = 0.3
```

---

## 📡 API Endpoints

### Generate Forecast
```bash
POST /v1/chat/completions
```

**Request:**
```json
{
  "ticker": "AAPL",
  "end_date": "2024-10-30",
  "past_weeks": 4,
  "include_financials": false,
  "temperature": 0.2,
  "max_new_tokens": 256,
  "stream": true
}
```

**Response (streaming):**
```
data: {"id":"chatcmpl-fingpt","object":"chat.completion.chunk",...}
data: [DONE]
```

### Health Check
```bash
GET /health
```

### Cache Statistics
```bash
GET /cache/stats
```

### Model Info
```bash
GET /debug/model
```

---

## 🎓 How It Works

### Architecture

```
User Request
    ↓
API Endpoint (server_optimized.py)
    ↓
Data Service (data_service.py)
    ├─→ yfinance: Stock data
    ├─→ Finnhub: News & financials
    └─→ Build prompt
         ↓
Cache Check (cache.py)
    ├─→ HIT: Return cached forecast
    └─→ MISS: Generate new
         ↓
Model (model.py)
    ├─→ Load with 4-bit quantization
    ├─→ Generate forecast
    └─→ Stream tokens
         ↓
Cache Store
    ↓
Return to User
```

### Key Optimizations

1. **4-bit Quantization**: Reduces model size by 70%
2. **Reduced Tokens**: 256 tokens is enough for forecasts
3. **Caching**: Financial data (1hr), forecasts (30min)
4. **Memory Management**: Aggressive cleanup after generation
5. **Optimized Sampling**: Lower top_k, higher repetition penalty

---

## 🐛 Troubleshooting

### bitsandbytes Won't Install

```python
# Disable quantization in setting.py
USE_4BIT_QUANTIZATION = False
USE_8BIT_QUANTIZATION = False
```

Or try llama.cpp (see `OPTIMIZATION_STRATEGY.md`)

### Out of Memory

```python
# In setting.py
USE_4BIT_QUANTIZATION = True
DEFAULT_MAX_NEW_TOKENS = 200
LOW_MEMORY_MODE = True
```

### Slow Generation

1. Check quantization is enabled: `GET /debug/model`
2. Reduce max_new_tokens: `DEFAULT_MAX_NEW_TOKENS = 128`
3. Run benchmark: `python benchmark.py`

### Cache Not Working

```bash
# Check server logs for cache HIT/MISS
# Test with: ./scripts/test_cache.sh
```

---

## 📚 Documentation

- **[SETUP_GUIDE.md](docs/SETUP_GUIDE.md)** - Complete setup and migration guide
- **[OPTIMIZATION_STRATEGY.md](docs/OPTIMIZATION_STRATEGY.md)** - Detailed optimization techniques
- **[EXECUTIVE_SUMMARY.md](docs/EXECUTIVE_SUMMARY.md)** - Performance analysis
- **[RESTRUCTURE_PLAN.md](docs/RESTRUCTURE_PLAN.md)** - Architecture overview

---

## 🧪 Testing

### Run Benchmark
```bash
python benchmark.py
```

### Manual Test
```bash
# Terminal 1: Start server
python server.py

# Terminal 2: Test request
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"ticker":"AAPL","end_date":"2024-10-30","stream":false}'
```

---

## 🔒 Environment Variables

Required in `.env`:

```bash
# Required
FINNHUB_API_KEY=your_key_here

# Optional (only if model not downloaded)
HF_TOKEN=your_huggingface_token
```

Get Finnhub API key: https://finnhub.io/register

---

## 📋 Requirements

- Python 3.9-3.12 (avoid 3.14 on Mac)
- Mac M4, M3, M2, M1 (or any Apple Silicon)
- 16GB RAM (8GB available during generation)
- ~15GB disk space for model

---

## 🚀 Performance Tips

1. **Enable 4-bit quantization** for best speed/quality balance
2. **Reduce max_new_tokens** if faster responses needed
3. **Enable caching** for repeated queries
4. **Close other apps** to free memory
5. **Lower temperature** (0.1-0.2) for faster, more focused outputs

---

## 🎯 Use Cases

- **Stock forecasting** - Weekly predictions with analysis
- **Market sentiment** - News-based company analysis
- **Risk assessment** - Identify positive developments and concerns
- **Batch processing** - Multiple stocks with caching
- **Real-time analysis** - Streaming responses

---

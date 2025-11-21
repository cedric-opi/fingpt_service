# FinGPT Stock Forecaster Server

Optimized for **Mac Mini M4 (16GB RAM)**

Financial stock forecasting using **FinGPT (Llama-2-7B)** with **4-bit quantization** for blazing-fast inference.

---

## 🎯 Key Features

* ⚡ **15–20x faster** than baseline (12–18 min vs 3–5 hours)
* 💾 **60% less memory** (4–6GB vs 14–16GB)
* 🚀 **4-bit quantization** with bitsandbytes
* 💰 Intelligent caching for repeated queries
* 🏗️ Clean modular architecture
* 📊 Real-time streaming or batch responses
* 🔧 Easy configuration with a single config file

---

# 🛠️ Installation & Execution Guide

## 🍎 macOS Installation (Apple Silicon Recommended)

### 1. Clone Project

```
git clone <repo-url>
cd fingpt-server
```

### 2. Create Virtual Environment

```
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```
pip install -r requirements.txt
```

### 4. Install llama.cpp with Metal Acceleration

```
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
python -c "from llama_cpp import Llama; print('✅ llama-cpp-python installed!')"
```

### 5. Download GGUF Model

```
pip install huggingface-hub
mkdir -p models
huggingface-cli download TheBloke/Llama-2-7B-Chat-GGUF \
  llama-2-7b-chat.Q4_K_M.gguf \
  --local-dir ./models \
  --local-dir-use-symlinks False
```

If required:

```
huggingface-cli login
```

### 6. Start Server

```
python server.py
```

---

## 🪟 Windows Installation

### 1. Clone Project

```
git clone <repo-url>
cd fingpt-server
```

### 2. Create Virtual Environment

```
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies

```
pip install -r requirements.txt
```

### 4. Install CPU-Only llama.cpp Wheel

```
pip install llama-cpp-python --no-cache-dir --force-reinstall
python -c "from llama_cpp import Llama; print('✅ llama-cpp-python installed on Windows!')"
```

### 5. Download Model

```
pip install huggingface-hub
mkdir models
huggingface-cli download TheBloke/Llama-2-7B-Chat-GGUF \
  llama-2-7b-chat.Q4_K_M.gguf \
  --local-dir ./models \
  --local-dir-use-symlinks False
```

If required:

```
huggingface-cli login
```

### 6. Start Server

```
python server.py
```

---

# 🔧 Configuration

### 1. Create `.env` File

```
cp .env.example .env
```

Add Finnhub API Key:

```
FINNHUB_API_KEY=your_key_here
```

### 2. Modify `config.py`

Best speed (recommended):

```
USE_4BIT_QUANTIZATION = True
```

Fallback (if bitsandbytes fails):

```
USE_4BIT_QUANTIZATION = False
```

---

# ▶️ Run Server

```
python server.py
```

---

# 🧪 Testing

### Health Check

```
curl http://localhost:8000/health
```

### Generate Forecast

```
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "AAPL",
    "end_date": "2024-10-30",
    "stream": false
  }'
```

---

# 📁 Project Structure

```
fingpt-server/
├── config.py
├── model.py
├── data_service.py
├── cache.py
├── server_optimized.py
├── requirements.txt
├── .env.example
│
├── scripts/
│   ├── benchmark.py
│   └── test_cache.sh
│
└── docs/
```

---

# 📡 API Endpoints

### **POST /v1/chat/completions** — Generate Forecast

```
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

### **GET /health** — Health Check

### **GET /cache/stats** — Cache Statistics

### **GET /debug/model** — Model Info

---

# 🐛 Troubleshooting

### bitsandbytes Fails

```
USE_4BIT_QUANTIZATION = False
USE_8BIT_QUANTIZATION = False
```

### Out of Memory

```
USE_4BIT_QUANTIZATION = True
DEFAULT_MAX_NEW_TOKENS = 200
LOW_MEMORY_MODE = True
```

### Slow Generation

* Ensure quantization enabled
* Reduce max_new_tokens
* Run benchmark:

```
python benchmark.py
```

### Cache Not Working

```
./scripts/test_cache.sh
```

---

# 🔒 Environment Variables

```
FINNHUB_API_KEY=your_key_here
HF_TOKEN=your_huggingface_token
```

---

# 🚀 Performance Tips

* Use 4-bit quantization
* Reduce max_new_tokens for speed
* Enable caching
* Free system memory
* Use low temperature (0.1–0.2)

---

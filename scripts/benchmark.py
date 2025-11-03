#!/usr/bin/env python3
"""
Benchmark script for FinGPT Server
Tests model performance and memory usage
"""
import time
import requests
import psutil
import json
from datetime import datetime

# Configuration
SERVER_URL = "http://localhost:8000"
TEST_TICKER = "AAPL"
TEST_DATE = "2024-10-30"

def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 3)

def test_health():
    """Test server health"""
    print("="*70)
    print("🏥 Testing Server Health")
    print("="*70)
    
    try:
        response = requests.get(f"{SERVER_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is healthy")
            print(json.dumps(response.json(), indent=2))
            return True
        else:
            print(f"❌ Server returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to server: {e}")
        return False

def test_model_info():
    """Get model information"""
    print("\n" + "="*70)
    print("🔍 Model Information")
    print("="*70)
    
    try:
        response = requests.get(f"{SERVER_URL}/debug/model", timeout=5)
        if response.status_code == 200:
            info = response.json()
            print(f"✅ Model loaded: {info.get('loaded')}")
            print(f"   Device: {info.get('device')}")
            print(f"   Quantization: {info.get('quantization')}")
            if 'estimated_size_gb' != 'Unknown':
                print(f"   Estimated Size: {info.get('estimated_size_gb')} GB")
            return info
    except requests.exceptions.RequestException as e:
        print(f"❌ Error getting model info: {e}")
        return None

def test_generation(stream=False, cached=False):
    """Test generation performance"""
    cache_str = "Cached" if cached else "Cold"
    stream_str = "Streaming" if stream else "Non-streaming"
    
    print("\n" + "="*70)
    print(f"🚀 Testing {cache_str} {stream_str} Generation")
    print("="*70)
    
    payload = {
        "ticker": TEST_TICKER,
        "end_date": TEST_DATE,
        "stream": stream,
        "max_new_tokens": 256
    }
    
    start_time = time.time()
    start_memory = get_memory_usage()
    
    try:
        if stream:
            response = requests.post(
                f"{SERVER_URL}/v1/chat/completions",
                json=payload,
                stream=True,
                timeout=600
            )
            
            chunks = []
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        data = line_str[6:]
                        if data != '[DONE]':
                            chunks.append(data)
            
            content_length = sum(len(c) for c in chunks)
            
        else:
            response = requests.post(
                f"{SERVER_URL}/v1/chat/completions",
                json=payload,
                timeout=600
            )
            result = response.json()
            content = result['choices'][0]['message']['content']
            content_length = len(content)
        
        end_time = time.time()
        end_memory = get_memory_usage()
        
        elapsed = end_time - start_time
        memory_used = end_memory - start_memory
        
        print(f"✅ Generation successful!")
        print(f"⏱️  Time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        print(f"💾 Memory Used: {memory_used:.2f} GB")
        print(f"📊 Content Length: {content_length} chars")
        print(f"⚡ Speed: {content_length/elapsed:.1f} chars/sec")
        
        return {
            "success": True,
            "time": elapsed,
            "memory_used": memory_used,
            "content_length": content_length
        }
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Generation failed: {e}")
        return {"success": False}

def test_cache():
    """Test cache performance"""
    print("\n" + "="*70)
    print("💾 Testing Cache Performance")
    print("="*70)
    
    # First request (cache miss)
    print("\n1️⃣  First request (should be cache MISS)...")
    result1 = test_generation(stream=False, cached=False)
    
    if not result1["success"]:
        print("❌ First request failed, skipping cache test")
        return
    
    # Wait a moment
    print("\n⏳ Waiting 2 seconds...")
    time.sleep(2)
    
    # Second request (cache hit)
    print("\n2️⃣  Second request (should be cache HIT)...")
    result2 = test_generation(stream=False, cached=True)
    
    if result2["success"]:
        speedup = result1["time"] / result2["time"]
        print("\n" + "="*70)
        print("📊 Cache Performance Summary")
        print("="*70)
        print(f"First Request:  {result1['time']:.1f}s")
        print(f"Second Request: {result2['time']:.1f}s")
        print(f"⚡ Speedup: {speedup:.1f}x faster!")
        
        if speedup > 3:
            print("✅ Cache is working GREAT!")
        elif speedup > 1.5:
            print("⚠️  Cache is working but not optimal")
        else:
            print("❌ Cache might not be working properly")

def main():
    """Run all benchmarks"""
    print("\n" + "="*70)
    print("🧪 FinGPT Performance Benchmark")
    print("="*70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Server: {SERVER_URL}")
    print("="*70)
    
    # Test 1: Health check
    if not test_health():
        print("\n❌ Server is not running. Please start the server first.")
        return
    
    # Test 2: Model info
    test_model_info()
    
    # Test 3: Non-streaming generation
    test_generation(stream=False, cached=False)
    
    # Test 4: Streaming generation
    test_generation(stream=True, cached=False)
    
    # Test 5: Cache performance
    test_cache()
    
    # Final summary
    print("\n" + "="*70)
    print("✅ BENCHMARK COMPLETE")
    print("="*70)
    print("\n💡 Tips:")
    print("   - If generation is slow, enable 4-bit quantization in config.py")
    print("   - If cache not working, check server logs")
    print("   - Reduce max_new_tokens for faster generation")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
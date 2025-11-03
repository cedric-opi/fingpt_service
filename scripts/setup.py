"""
Script to download and verify FinGPT model for Mac Mini M4
Run this ONCE before starting the server
"""
import os
import torch
from huggingface_hub import login, snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

print("Loading base model...")
base = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    dtype=torch.float16,
    device_map="cpu"  # Load on CPU to save GPU memory
)

print("Loading LoRA...")
model = PeftModel.from_pretrained(base, "./fingpt-forecaster_llama2-7b_lora")

print("Merging...")
merged = model.merge_and_unload()

print("Saving...")
merged.save_pretrained("./fingpt-merged")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
tokenizer.save_pretrained("./fingpt-merged")

print("✅ Done! Merged model saved to ./fingpt-merged")
print("   Update server.py:")
print("   BASE_MODEL_PATH = './fingpt-merged'")
print("   LORA_WEIGHTS_PATH = None")

def check_system():
    """Check if system is compatible"""
    print("🔍 Checking system compatibility...")
    
    # Check PyTorch
    print(f"✅ PyTorch version: {torch.__version__}")
    
    # Check MPS availability
    if torch.backends.mps.is_available():
        print("✅ Apple Silicon MPS backend is available")
    else:
        print("⚠️  MPS not available, will use CPU (slower)")
    
    # Check Python version
    import sys
    version = sys.version_info
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor == 14:
        print("⚠️  WARNING: Python 3.14 has compatibility issues. Recommend 3.11")
    
    return True


def download_base_model():
    """Download Llama-2-7b-chat-hf base model"""
    print("\n📥 Downloading Llama-2-7b-chat-hf base model...")
    print("Note: This is a large download (~13GB). First time only.")
    print("Files will be cached at ~/.cache/huggingface/")
    
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    
    try:
        # Check if already downloaded
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        model_dir = os.path.join(cache_dir, "models--meta-llama--Llama-2-7b-chat-hf")
        
        if os.path.exists(model_dir):
            print("✅ Base model already downloaded!")
            return True
        
        # Download tokenizer first (small)
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✅ Tokenizer downloaded")
        
        # Download model (this will take time)
        print("Downloading model weights (this may take 10-30 minutes)...")
        print("You can monitor progress below:")
        
        snapshot_download(
            repo_id=model_name,
            local_dir=None,  # Use default cache
            local_dir_use_symlinks=True,
        )
        
        print("✅ Base model downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading base model: {e}")
        print("\n💡 Troubleshooting:")
        print("1. Make sure you've accepted the Llama 2 license:")
        print("   https://huggingface.co/meta-llama/Llama-2-7b-chat-hf")
        print("2. Login to Hugging Face:")
        print("   Run: huggingface-cli login")
        print("   Or use: login() in Python")
        return False


def download_fingpt_lora():
    """Download or setup FinGPT LoRA weights"""
    print("\n📥 Setting up FinGPT LoRA weights...")
    
    lora_path = "./fingpt-forecaster_llama2-7b_lora"
    
    if os.path.exists(lora_path):
        print(f"✅ LoRA weights found at {lora_path}")
        return True
    
    print("⚠️  LoRA weights not found!")
    print("\n📝 To get FinGPT LoRA weights, you have two options:")
    print("\n1. Clone from FinGPT repository:")
    print("   git clone https://github.com/AI4Finance-Foundation/FinGPT.git")
    print("   cd FinGPT")
    print("   # Follow instructions to get fine-tuned weights")
    
    print("\n2. Use Hugging Face (if available):")
    print("   # Check https://huggingface.co/FinGPT for available models")
    
    print("\n3. OR fine-tune yourself:")
    print("   # Follow FinGPT training guide")
    print("   # https://github.com/AI4Finance-Foundation/FinGPT")
    
    print("\n❓ For now, you can also test with just the base model")
    print("   (Set LORA_WEIGHTS_PATH = None in server.py)")
    
    return False


def verify_setup():
    """Verify everything is set up correctly"""
    print("\n🔍 Verifying setup...")
    
    checks = []
    
    # Check base model
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        checks.append(("Base model tokenizer", True))
    except Exception as e:
        checks.append(("Base model tokenizer", False))
    
    # Check LoRA weights
    lora_path = "./fingpt-forecaster_llama2-7b_lora"
    checks.append(("LoRA weights", os.path.exists(lora_path)))
    
    # Check .env file
    env_path = ".env"
    checks.append((".env file", os.path.exists(env_path)))
    
    # Display results
    print("\n📋 Setup Checklist:")
    for name, status in checks:
        icon = "✅" if status else "❌"
        print(f"  {icon} {name}")
    
    all_good = all(status for _, status in checks)
    
    if all_good:
        print("\n🎉 Setup complete! You can now start the server:")
        print("   uvicorn server:app --reload --host 0.0.0.0 --port 8000")
    else:
        print("\n⚠️  Some items need attention. See messages above.")
    
    return all_good


def main():
    """Main setup flow"""
    print("=" * 60)
    print("   FinGPT Model Setup for Mac Mini M4 (16GB RAM)")
    print("=" * 60)
    
    # Step 1: Check system
    if not check_system():
        print("❌ System check failed")
        return
    
    # Step 2: Login to Hugging Face
    print("\n" + "=" * 60)
    print("🔑 Hugging Face Authentication")
    print("=" * 60)
    print("You need a Hugging Face account to download Llama 2.")
    print("1. Create account at: https://huggingface.co/join")
    print("2. Accept Llama 2 license: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf")
    print("3. Get your token: https://huggingface.co/settings/tokens")
    
    choice = input("\nHave you completed these steps? (y/n): ").lower()
    
    if choice == 'y':
        print("\nPlease login with your Hugging Face token:")
        try:
            login()
            print("✅ Login successful!")
        except Exception as e:
            print(f"❌ Login failed: {e}")
            print("You can also run: huggingface-cli login")
            return
    else:
        print("Please complete authentication first, then run this script again.")
        return
    
    # Step 3: Download base model
    print("\n" + "=" * 60)
    download_base_model()
    
    # Step 4: Setup LoRA weights
    print("\n" + "=" * 60)
    download_fingpt_lora()
    
    # Step 5: Verify everything
    print("\n" + "=" * 60)
    verify_setup()
    
    print("\n" + "=" * 60)
    print("Setup process complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
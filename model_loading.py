# Chứa hàm LLM load_forecaster_model 
model = None
tokenizer = None

def load_forecaster_model():
    """Tải Base Model và LoRA weights của FinGPT-Forecaster với Quantization 4-bit."""
    global model, tokenizer
    print("Loading FinGPT-Forecaster model with 4-bit quantization...")
    
    # --- BƯỚC 1: Cấu hình Quantization ---
    # Cấu hình để tải mô hình với 4-bit (NF4)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4", # Highly recommended by the FinGPT community
        bnb_4bit_compute_dtype=torch.float16, # Sử dụng fp16 cho tính toán (giữ tốc độ)
        bnb_4bit_use_double_quant=True
    )
    
    # --- BƯỚC 2: Tải Base Model với Config ---
    base_model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-2-7b-chat-hf',
        trust_remote_code=True,
        device_map="auto",
        # THAY THẾ torch_dtype=torch.float16 BẰNG quantization_config
        quantization_config=bnb_config, 
    )
    
    # Cần set model.config.use_cache = False nếu gặp lỗi
    base_model.config.use_cache = False 
    
    # 3. Tải Tokenizer
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
    
    # 4. Tải LoRA weights
    model = PeftModel.from_pretrained(base_model, 'FinGPT/fingpt-forecaster_dow30_llama2-7b_lora')
    model = model.eval()
    print("FinGPT-Forecaster model loaded successfully with 4-bit.")
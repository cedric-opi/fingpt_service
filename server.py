# Chứa logic API chính FastAPI endpoints và hàm sinh văn bản gen_reply
import os
from dotenv import load_dotenv

from datetime import datetime, timedelta
import yfinance as yf
import finnhub
from pydantic import BaseModel
from typing import List, Optional, Iterator
import re 
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse

# Thư viện cho LLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Hàm thu nhập dữ liệu
def fetch_data_for_prompt(ticker: str, end_date: str, past_weeks: int, include_financials: bool) -> str:
    """Thu thập dữ liệu từ yfinance và Finnhub, sau đó tạo prompt."""
    
    # Tính toán khoảng thời gian
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    start_dt = end_dt - timedelta(weeks=past_weeks)
    
    start_date_str = start_dt.strftime("%Y-%m-%d")
    end_date_str = end_dt.strftime("%Y-%m-%d")
    
    # --- 3.1. Thu thập Thông tin Cơ bản (Finnhub) ---
    try:
        profile = finnhub_client.company_profile2(symbol=ticker)
    except Exception as e:
        print(f"Error fetching company profile: {e}")
        profile = {} # Dùng dictionary rỗng nếu lỗi
        
    # --- 3.2. Thu thập Dữ liệu Giá Cổ phiếu (yfinance) ---
    stock_data = yf.Ticker(ticker).history(start=start_date_str, end=end_date_str)
    
    if stock_data.empty:
        raise ValueError(f"No stock data found for {ticker} from {start_date_str} to {end_date_str}")
    
    start_price = stock_data['Close'].iloc[0]
    end_price = stock_data['Close'].iloc[-1]
    
    price_change = "increased" if end_price > start_price else "decreased"

    # --- 3.3. Thu thập Tin tức (Finnhub) ---
    # Giới hạn tin tức trong phạm vi 1 tháng (4 tuần) trước ngày kết thúc
    news_list = finnhub_client.company_news(ticker, _from=start_date_str, to=end_date_str)
    
    news_string = ""
    for item in news_list[:10]: # Giới hạn 10 tin tức quan trọng nhất
        news_string += f"[Headline]: {item['headline']}\n[Summary]: {item['summary']}\n\n"
    
    # --- 3.4. Thu thập Tài chính Cơ bản (Finnhub - Tùy chọn) ---
    financials_string = ""
    if include_financials:
        # Finnhub có nhiều endpoint tài chính, ta dùng basic_financials (ví dụ)
        basic_financials = finnhub_client.company_basic_financials(ticker, 'all')
        if basic_financials and 'metric' in basic_financials:
            metrics = basic_financials['metric']
            # Chọn lọc một số chỉ số quan trọng
            selected_metrics = {
                'marketCapitalization': metrics.get('marketCapitalization', 'N/A'),
                'beta': metrics.get('beta', 'N/A'),
                'peTTM': metrics.get('peTTM', 'N/A'),
            }
            financials_string = "\n".join([f"{k}: {v}" for k, v in selected_metrics.items()])
            financials_string = f"Some recent basic financials of {profile.get('name', ticker)}, reported at {end_date_str}, are presented below:\n\n[Basic Financials]:\n{financials_string}\n"
    
    # --- 3.5. Xây dựng Prompt (Dùng template từ README) ---
    
    # Điền dữ liệu vào template
    prompt_template = f"""
[Company Introduction]:

{profile.get('name', ticker)} is a leading entity in the {profile.get('finnhubIndustry', 'N/A')} sector. Incorporated and publicly traded since {profile.get('ipo', 'N/A')}, the company has established its reputation as one of the key players in the market. As of today, {profile.get('name', ticker)} has a market capitalization of {profile.get('marketCapitalization', 0.0):.2f} in {profile.get('currency', 'USD')}, with {profile.get('shareOutstanding', 0.0):.2f} shares outstanding. {profile.get('name', ticker)} operates primarily in the {profile.get('country', 'N/A')}, trading under the ticker {ticker} on the {profile.get('exchange', 'N/A')}. As a dominant force in the {profile.get('finnhubIndustry', 'N/A')} space, the company continues to innovate and drive progress within the industry.

From {start_date_str} to {end_date_str}, {profile.get('name', ticker)}'s stock price {price_change} from {start_price:.2f} to {end_price:.2f}. Company news during this period are listed below:

{news_string}

{financials_string}

Based on all the information before {end_date_str}, let's first analyze the positive developments and potential concerns for {ticker}. Come up with 2-4 most important factors respectively and keep them concise. Most factors should be inferred from company-related news. Then make your prediction of the {ticker} stock price movement for next week ({start_date_str} to {end_date_str}). Provide a summary analysis to support your prediction.
"""
    # Loại bỏ khoảng trắng thừa
    return prompt_template.strip()

# Khởi tạo FastAPI
app = FastAPI()

# Biến môi trường cho Finnhub API Key
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")

# Khởi tạo Finnhub Client
# TODO: Kiểm tra nếu FINNHUB_API_KEY không tồn tại
finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)


# Constants cho Llama format (từ README)
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

# SYSTEM PROMPT (từ README)
SYSTEM_PROMPT = """You are a seasoned stock market analyst. Your task is to list the positive developments and potential concerns for companies based on relevant news and basic financials from the past weeks, then provide an analysis and prediction for the companies' stock price movement for the upcoming week. Your answer format should be as follows:\n\n[Positive Developments]:\n1. ...\n\n[Potential Concerns]:\n1. ...\n\n[Prediction & Analysis]:\n...\n"""

# Hàm sinh văn bản thực tế
def gen_reply(full_prompt: str, temperature: float = 0.2) -> Iterator[str]:
    # ... (Tạo final_prompt và tokenize inputs) ...
    
    # Sử dụng generator để truyền dữ liệu sinh ra
    class StreamingGenerator:
        def __init__(self):
            self.queue = Queue()
            self.stop_signal = object()
        
        def put(self, token_ids):
            # Decode token IDs thành văn bản
            text = tokenizer.decode(token_ids, skip_special_tokens=True)
            self.queue.put(text)
        
        def end(self):
            self.queue.put(self.stop_signal)
            
        def __iter__(self):
            while True:
                item = self.queue.get()
                if item is self.stop_signal:
                    break
                yield item

    generator = StreamingGenerator()
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, output_handler=generator)

    def generate_and_stream():
        # Chạy trong một thread hoặc async task để không chặn FastAPI
        model.generate(
            **inputs, 
            max_length=4096, 
            do_sample=True,
            temperature=temperature,
            eos_token_id=tokenizer.eos_token_id,
            streamer=streamer # Dùng streamer thay vì nhận res
        )
        generator.end()

    # Bắt đầu sinh và trả về iterator
    import threading
    threading.Thread(target=generate_and_stream).start()
    return generator # Trả về generator để StreamingResponse dùng

# --- Cập nhật hàm chat trong server.py ---
class ForecasterRequest(BaseModel):
    ticker: str
    end_date: str # Định dạng YYYY-MM-DD
    past_weeks: int
    include_financials: bool
    # Các trường khác: model, temperature, stream (như đã có)

@app.post("/v1/chat/completions")
def chat(req: ForecasterRequest):
    
    # Tạo full prompt
    try:
        user_prompt = fetch_data_for_prompt(
            req.ticker, 
            req.end_date, 
            req.past_weeks, 
            req.include_financials
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    # ... Phần còn lại của logic streaming (như bạn đã viết) ...
    if req.stream:
        def sse():
            for ch in gen_reply(user_prompt, req.temperature): # Truyền prompt vào
                # Định dạng SSE cho AI SDK của Vercel
                yield f'data: {{"id":"dummy","object":"chat.completion.chunk","choices":[{{"delta":{{"content":"{ch}"}}}}]}}\n\n'
            yield 'data: [DONE]\n\n'
        return StreamingResponse(sse(), media_type="text/event-stream")

    # ... Logic non-streaming nếu cần ...
    text = "".join(list(gen_reply(user_prompt, req.temperature)))
    return JSONResponse({
        "id": "chatcmpl-fingpt",
        "object": "chat.completion",
        "choices": [
            {"message": {"role": "assistant", "content": text}}
        ]
    })

@app.get("/healthz")
def healthz():
    return {"status": "ok", "service": "FinGPT-Forecaster"}
"""
Route different types of financial questions to appropriate handlers
"""
import re
from typing import Tuple

def detect_question_type(user_message: str) -> Tuple[str, dict]:
    """
    Detect what type of financial question this is
    
    Returns: (question_type, extracted_data)
    """
    message_lower = user_message.lower()
    
    # Stock analysis patterns
    stock_patterns = [
        r'\b([A-Z]{1,5})\b.*(?:stock|price|forecast|predict|analysis)',
        r'(?:analyze|forecast|predict).*\b([A-Z]{1,5})\b',
        r'what.*(?:think|predict).*\b([A-Z]{1,5})\b',
    ]
    
    for pattern in stock_patterns:
        match = re.search(pattern, user_message)
        if match:
            ticker = match.group(1)
            return ("stock_forecast", {"ticker": ticker})
    
    # Comparison questions
    if any(word in message_lower for word in ['compare', 'vs', 'versus', 'better']):
        tickers = re.findall(r'\b([A-Z]{2,5})\b', user_message)
        if len(tickers) >= 2:
            return ("stock_comparison", {"tickers": tickers[:2]})
    
    # Market/sector questions
    if any(word in message_lower for word in ['market', 'sector', 'industry', 'trend']):
        return ("market_analysis", {"query": user_message})
    
    # Educational/general financial questions
    if any(word in message_lower for word in ['what is', 'explain', 'how does', 'why']):
        return ("financial_education", {"query": user_message})
    
    # Portfolio/investment advice
    if any(word in message_lower for word in ['should i', 'invest', 'buy', 'sell', 'portfolio']):
        return ("investment_advice", {"query": user_message})
    
    # Default to general chat
    return ("general_chat", {"query": user_message})
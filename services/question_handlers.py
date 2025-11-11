"""
Handlers for different types of financial questions
"""
from typing import Iterator
from core.model_llama_cpp import generate_forecast_llama_cpp
from services.data_service import build_forecast_prompt, fetch_company_profile, fetch_stock_data
from config import setting as config
import yfinance as yf
from datetime import datetime, timedelta

def handle_stock_forecast(ticker: str, end_date: str) -> Iterator[str]:
    """Handle stock forecast questions"""
    prompt = build_forecast_prompt(ticker, end_date, past_weeks=4, include_financials=True)
    yield from generate_forecast_llama_cpp(prompt, stream=True)


def handle_stock_comparison(ticker1: str, ticker2: str, end_date: str) -> Iterator[str]:
    """
    Compare two stocks with comprehensive data for better forecasting
    """
    try:
        # Fetch comprehensive data for both stocks
        stock1_data = _fetch_comparison_data(ticker1, end_date)
        stock2_data = _fetch_comparison_data(ticker2, end_date)
        
        # Build detailed comparison prompt
        prompt = f"""You are FinGPT, a financial forecasting expert. Compare these two stocks for investment over the next 1 week.

{'='*60}
STOCK 1: {ticker1}
{'='*60}
Company: {stock1_data['name']}
Sector: {stock1_data['sector']}
Industry: {stock1_data['industry']}

RECENT PRICE PERFORMANCE (4 weeks):
- Current Price: ${stock1_data['current_price']:.2f}
- 4-Week Change: {stock1_data['price_change_pct']:+.2f}%
- 52-Week Range: ${stock1_data['year_low']:.2f} - ${stock1_data['year_high']:.2f}
- Average Volume: {stock1_data['avg_volume']:,.0f}

FINANCIAL METRICS:
- Market Cap: ${stock1_data['market_cap']:,.0f}
- P/E Ratio: {stock1_data['pe_ratio']}
- EPS: ${stock1_data['eps']}
- Profit Margin: {stock1_data['profit_margin']}%
- Revenue Growth: {stock1_data['revenue_growth']}%
- Debt/Equity: {stock1_data['debt_to_equity']}

RECENT NEWS SENTIMENT: {stock1_data['news_summary']}

{'='*60}
STOCK 2: {ticker2}
{'='*60}
Company: {stock2_data['name']}
Sector: {stock2_data['sector']}
Industry: {stock2_data['industry']}

RECENT PRICE PERFORMANCE (4 weeks):
- Current Price: ${stock2_data['current_price']:.2f}
- 4-Week Change: {stock2_data['price_change_pct']:+.2f}%
- 52-Week Range: ${stock2_data['year_low']:.2f} - ${stock2_data['year_high']:.2f}
- Average Volume: {stock2_data['avg_volume']:,.0f}

FINANCIAL METRICS:
- Market Cap: ${stock2_data['market_cap']:,.0f}
- P/E Ratio: {stock2_data['pe_ratio']}
- EPS: ${stock2_data['eps']}
- Profit Margin: {stock2_data['profit_margin']}%
- Revenue Growth: {stock2_data['revenue_growth']}%
- Debt/Equity: {stock2_data['debt_to_equity']}

RECENT NEWS SENTIMENT: {stock2_data['news_summary']}

{'='*60}
ANALYSIS REQUIRED:
{'='*60}

Please provide a comprehensive comparison covering:

[Positive Developments]:
- List positive factors for each stock (2-3 points each)

[Potential Concerns]:
- List risk factors and concerns for each stock (2-3 points each)

[Price Performance Comparison]:
- Which stock has stronger momentum?
- Volume trends and liquidity

[Valuation Comparison]:
- Which offers better value based on P/E, EPS, margins?
- Growth vs value perspective

[1-Week Forecast]:
For {ticker1}: [Up/Down/Neutral] - Explain why
For {ticker2}: [Up/Down/Neutral] - Explain why

[Investment Recommendation]:
- Which stock is better for 1-week holding?
- Confidence level (High/Medium/Low)
- Risk assessment
- Final verdict with clear reasoning

Be specific, data-driven, and provide actionable insights."""

        yield from generate_forecast_llama_cpp(prompt, temperature=0.3, max_new_tokens=512, stream=True)
        
    except Exception as e:
        # Fallback to simpler comparison if data fetch fails
        error_msg = f"⚠️ Unable to fetch complete data: {str(e)}\n\n"
        yield error_msg
        
        fallback_prompt = f"""Compare {ticker1} and {ticker2} stocks based on general financial analysis.

Analyze:
1. Business models and competitive advantages
2. Market position in their sectors
3. General investment outlook for next week
4. Which stock appears more attractive and why

Provide a clear comparison and recommendation."""
        
        yield from generate_forecast_llama_cpp(fallback_prompt, stream=True)


def _fetch_comparison_data(ticker: str, end_date: str) -> dict:
    """
    Fetch comprehensive data for stock comparison
    """
    stock = yf.Ticker(ticker)
    info = stock.info
    
    # Calculate date range for historical data
    end = datetime.strptime(end_date, "%Y-%m-%d") if isinstance(end_date, str) else end_date
    start = end - timedelta(weeks=4)
    
    # Fetch historical data
    hist = stock.history(start=start, end=end)
    
    # Calculate metrics
    current_price = hist['Close'].iloc[-1] if len(hist) > 0 else info.get('currentPrice', 0)
    start_price = hist['Close'].iloc[0] if len(hist) > 0 else current_price
    price_change_pct = ((current_price - start_price) / start_price * 100) if start_price > 0 else 0
    
    # Fetch recent news sentiment (simplified)
    news = stock.news[:3] if hasattr(stock, 'news') and stock.news else []
    news_summary = _summarize_news(news) if news else "No recent news available"
    
    return {
        # Company info
        'name': info.get('longName', ticker),
        'sector': info.get('sector', 'N/A'),
        'industry': info.get('industry', 'N/A'),
        
        # Price data
        'current_price': current_price,
        'price_change_pct': price_change_pct,
        'year_high': info.get('fiftyTwoWeekHigh', 0),
        'year_low': info.get('fiftyTwoWeekLow', 0),
        'avg_volume': hist['Volume'].mean() if len(hist) > 0 else info.get('averageVolume', 0),
        
        # Financial metrics
        'market_cap': info.get('marketCap', 0),
        'pe_ratio': f"{info.get('trailingPE', 0):.2f}" if info.get('trailingPE') else 'N/A',
        'eps': f"{info.get('trailingEps', 0):.2f}" if info.get('trailingEps') else 'N/A',
        'profit_margin': f"{info.get('profitMargins', 0) * 100:.2f}" if info.get('profitMargins') else 'N/A',
        'revenue_growth': f"{info.get('revenueGrowth', 0) * 100:.2f}" if info.get('revenueGrowth') else 'N/A',
        'debt_to_equity': f"{info.get('debtToEquity', 0):.2f}" if info.get('debtToEquity') else 'N/A',
        
        # News sentiment
        'news_summary': news_summary
    }


def _summarize_news(news_list: list) -> str:
    """Summarize recent news headlines"""
    if not news_list:
        return "No recent news"
    
    headlines = [item.get('title', '') for item in news_list[:3]]
    return "Recent headlines: " + " | ".join(headlines[:2])


def handle_market_analysis(query: str) -> Iterator[str]:
    """Handle market/sector analysis questions"""
    prompt = f"""{config.SYSTEM_PROMPT}

User Question: {query}

Provide a comprehensive market analysis addressing the user's question. Include:
- Current market trends
- Key factors affecting the market/sector
- Outlook for the coming week
- Investment implications"""
    
    yield from generate_forecast_llama_cpp(prompt, stream=True)


def handle_financial_education(query: str) -> Iterator[str]:
    """Handle educational financial questions"""
    prompt = f"""You are a helpful financial educator. Explain financial concepts clearly and simply.

Question: {query}

Provide a clear, educational answer that:
1. Defines key terms
2. Explains the concept with examples
3. Shows real-world applications
4. Keeps it simple for beginners"""
    
    yield from generate_forecast_llama_cpp(prompt, stream=True)


def handle_investment_advice(query: str) -> Iterator[str]:
    """Handle investment advice questions (with disclaimer)"""
    prompt = f"""You are a financial analyst providing educational investment guidance.

Question: {query}

Provide thoughtful guidance that:
1. Analyzes the investment question objectively
2. Discusses pros and cons
3. Mentions key considerations
4. Emphasizes this is educational, not personalized advice

IMPORTANT: Always remind users to consult with a licensed financial advisor before making investment decisions."""
    
    yield from generate_forecast_llama_cpp(prompt, stream=True)


def create_investment_advice_prompt(query: str) -> str:
    """Create prompt for investment advice questions"""
    return f"""You are a professional financial advisor providing investment guidance.

User Question: {query}

Provide balanced, informative advice that includes:
1. Key factors to consider
2. Potential risks and rewards
3. General market context
4. A clear recommendation with reasoning

Remember: Always include appropriate disclaimers about doing your own research and consulting financial professionals.

Answer:"""

def create_comparison_prompt(ticker1: str, ticker2: str, end_date: str) -> str:
    """
    Create prompt for comparing two stocks (returns prompt string, doesn't stream)
    """
    try:
        # Fetch comprehensive data for both stocks
        stock1_data = _fetch_comparison_data(ticker1, end_date)
        stock2_data = _fetch_comparison_data(ticker2, end_date)
        
        # Build detailed comparison prompt
        return f"""You are FinGPT, a financial forecasting expert. Compare these two stocks for investment over the next 1 week.

{'='*60}
STOCK 1: {ticker1}
{'='*60}
Company: {stock1_data['name']}
Sector: {stock1_data['sector']}
Industry: {stock1_data['industry']}

RECENT PRICE PERFORMANCE (4 weeks):
- Current Price: ${stock1_data['current_price']:.2f}
- 4-Week Change: {stock1_data['price_change_pct']:+.2f}%
- 52-Week Range: ${stock1_data['year_low']:.2f} - ${stock1_data['year_high']:.2f}
- Average Volume: {stock1_data['avg_volume']:,.0f}

FINANCIAL METRICS:
- Market Cap: ${stock1_data['market_cap']:,.0f}
- P/E Ratio: {stock1_data['pe_ratio']}
- EPS: ${stock1_data['eps']}
- Profit Margin: {stock1_data['profit_margin']}%
- Revenue Growth: {stock1_data['revenue_growth']}%
- Debt/Equity: {stock1_data['debt_to_equity']}

RECENT NEWS SENTIMENT: {stock1_data['news_summary']}

{'='*60}
STOCK 2: {ticker2}
{'='*60}
Company: {stock2_data['name']}
Sector: {stock2_data['sector']}
Industry: {stock2_data['industry']}

RECENT PRICE PERFORMANCE (4 weeks):
- Current Price: ${stock2_data['current_price']:.2f}
- 4-Week Change: {stock2_data['price_change_pct']:+.2f}%
- 52-Week Range: ${stock2_data['year_low']:.2f} - ${stock2_data['year_high']:.2f}
- Average Volume: {stock2_data['avg_volume']:,.0f}

FINANCIAL METRICS:
- Market Cap: ${stock2_data['market_cap']:,.0f}
- P/E Ratio: {stock2_data['pe_ratio']}
- EPS: ${stock2_data['eps']}
- Profit Margin: {stock2_data['profit_margin']}%
- Revenue Growth: {stock2_data['revenue_growth']}%
- Debt/Equity: {stock2_data['debt_to_equity']}

RECENT NEWS SENTIMENT: {stock2_data['news_summary']}

{'='*60}
ANALYSIS REQUIRED:
{'='*60}

Please provide a comprehensive comparison covering:

[Positive Developments]:
- List positive factors for each stock (2-3 points each)

[Potential Concerns]:
- List risk factors and concerns for each stock (2-3 points each)

[Price Performance Comparison]:
- Which stock has stronger momentum?
- Volume trends and liquidity

[Valuation Comparison]:
- Which offers better value based on P/E, EPS, margins?
- Growth vs value perspective

[1-Week Forecast]:
For {ticker1}: [Up/Down/Neutral] - Explain why
For {ticker2}: [Up/Down/Neutral] - Explain why

[Investment Recommendation]:
- Which stock is better for 1-week holding?
- Confidence level (High/Medium/Low)
- Risk assessment
- Final verdict with clear reasoning

Be specific, data-driven, and provide actionable insights."""
        
    except Exception as e:
        # Fallback to simpler comparison if data fetch fails
        return f"""Compare {ticker1} and {ticker2} stocks based on general financial analysis.

Analyze:
1. Business models and competitive advantages
2. Market position in their sectors
3. General investment outlook for next week
4. Which stock appears more attractive and why

Provide a clear comparison and recommendation.

Note: Unable to fetch detailed data ({str(e)}), providing general analysis."""


def create_market_analysis_prompt(query: str) -> str:
    """Create prompt for market/sector analysis questions"""
    return f"""{config.SYSTEM_PROMPT}

User Question: {query}

Provide a comprehensive market analysis addressing the user's question. Include:
- Current market trends
- Key factors affecting the market/sector
- Outlook for the coming week
- Investment implications

Answer:"""


def create_education_prompt(query: str) -> str:
    """Create prompt for educational financial questions"""
    return f"""You are a helpful financial educator. Explain financial concepts clearly and simply.

Question: {query}

Provide a clear, educational answer that:
1. Defines key terms
2. Explains the concept with examples
3. Shows real-world applications
4. Keeps it simple for beginners

Answer:"""
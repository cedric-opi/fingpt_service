"""
Data Service - Handles financial data fetching
Separated from model logic for better organization
"""
import logging
from datetime import datetime, timedelta
import yfinance as yf
import finnhub
from config import setting as config

logger = logging.getLogger(__name__)

# Initialize Finnhub client
finnhub_client = finnhub.Client(api_key=config.FINNHUB_API_KEY)


def fetch_company_profile(ticker: str) -> dict:
    """Fetch company profile from Finnhub"""
    try:
        profile = finnhub_client.company_profile2(symbol=ticker)
        return profile
    except Exception as e:
        logger.warning(f"⚠️  Error fetching company profile for {ticker}: {e}")
        return {}


def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> tuple:
    """
    Fetch stock price data from yfinance
    
    Returns:
        (start_price, end_price, price_change_direction)
    """
    import time
    import requests
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Use yf.download without show_errors parameter
            stock_data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False
            )
            
            # If empty, try with custom session
            if stock_data.empty:
                logger.warning(f"⚠️  Download empty, trying with custom session...")
                session = requests.Session()
                session.headers.update({
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                })
                
                ticker_obj = yf.Ticker(ticker, session=session)
                stock_data = ticker_obj.history(start=start_date, end=end_date)
            
            if stock_data.empty:
                if attempt < max_retries - 1:
                    logger.warning(f"⚠️  Attempt {attempt + 1}/{max_retries} failed for {ticker}, retrying in 2s...")
                    time.sleep(2)
                    continue
                else:
                    raise ValueError(
                        f"No stock data found for {ticker} "
                        f"from {start_date} to {end_date}. "
                        f"Ticker may be invalid or Yahoo Finance is blocking requests."
                    )
            
            # Success - extract prices
            start_price = float(stock_data['Close'].iloc[0])
            end_price = float(stock_data['Close'].iloc[-1])
            price_change = "increased" if end_price > start_price else "decreased"
            
            logger.info(f"✅ Successfully fetched data for {ticker}: ${start_price:.2f} → ${end_price:.2f}")
            return start_price, end_price, price_change
            
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"⚠️  Attempt {attempt + 1}/{max_retries} failed: {e}")
                time.sleep(2)
            else:
                logger.error(f"❌ All attempts failed for {ticker}: {e}")
                raise ValueError(f"Failed to fetch stock data for {ticker}: {str(e)}")
            
def fetch_company_news(ticker: str, start_date: str, end_date: str, max_items: int = None) -> str:
    """
    Fetch company news from Finnhub
    
    Returns:
        Formatted news string
    """
    max_items = max_items or config.MAX_NEWS_ITEMS
    
    try:
        news_list = finnhub_client.company_news(ticker, _from=start_date, to=end_date)
    except Exception as e:
        logger.warning(f"⚠️  Error fetching news for {ticker}: {e}")
        news_list = []
    
    news_string = ""
    for item in news_list[:max_items]:
        news_string += f"[Headline]: {item['headline']}\n"
        news_string += f"[Summary]: {item['summary']}\n\n"
    
    return news_string.strip()


def fetch_basic_financials(ticker: str, company_name: str, end_date: str) -> str:
    """
    Fetch basic financial metrics from Finnhub
    
    Returns:
        Formatted financials string or empty string
    """
    try:
        basic_financials = finnhub_client.company_basic_financials(ticker, 'all')
        
        if not basic_financials or 'metric' not in basic_financials:
            return ""
        
        metrics = basic_financials['metric']
        selected_metrics = {
            'marketCapitalization': metrics.get('marketCapitalization', 'N/A'),
            'beta': metrics.get('beta', 'N/A'),
            'peTTM': metrics.get('peTTM', 'N/A'),
        }
        
        financials_lines = [f"{k}: {v}" for k, v in selected_metrics.items()]
        financials_text = "\n".join(financials_lines)
        
        return (
            f"Some recent basic financials of {company_name}, "
            f"reported at {end_date}, are presented below:\n\n"
            f"[Basic Financials]:\n{financials_text}\n"
        )
    
    except Exception as e:
        logger.warning(f"⚠️  Error fetching financials for {ticker}: {e}")
        return ""


def build_forecast_prompt(
    ticker: str,
    end_date: str,
    past_weeks: int = None,
    include_financials: bool = None
) -> str:
    """
    Build complete prompt for stock forecasting
    
    Args:
        ticker: Stock ticker symbol
        end_date: End date in YYYY-MM-DD format
        past_weeks: Number of weeks to look back
        include_financials: Whether to include financial metrics
        
    Returns:
        Formatted prompt string ready for model
    """
    # Use defaults from config if not specified
    past_weeks = past_weeks or config.DEFAULT_PAST_WEEKS
    include_financials = include_financials if include_financials is not None else config.INCLUDE_FINANCIALS_DEFAULT
    
    # Calculate date range
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    start_dt = end_dt - timedelta(weeks=past_weeks)
    
    start_date_str = start_dt.strftime("%Y-%m-%d")
    end_date_str = end_dt.strftime("%Y-%m-%d")
    
    logger.info(f"📊 Fetching data for {ticker} from {start_date_str} to {end_date_str}")
    
    # Fetch all data
    profile = fetch_company_profile(ticker)
    company_name = profile.get('name', ticker)
    
    start_price, end_price, price_change = fetch_stock_data(
        ticker, start_date_str, end_date_str
    )
    
    news_string = fetch_company_news(ticker, start_date_str, end_date_str)
    
    financials_string = ""
    if include_financials:
        financials_string = fetch_basic_financials(ticker, company_name, end_date_str)
    
    # Build the prompt
    prompt = f"""[Company Introduction]:

{company_name} is a leading entity in the {profile.get('finnhubIndustry', 'N/A')} sector. Incorporated and publicly traded since {profile.get('ipo', 'N/A')}, the company has established its reputation as one of the key players in the market. As of today, {company_name} has a market capitalization of {profile.get('marketCapitalization', 0.0):.2f} in {profile.get('currency', 'USD')}, with {profile.get('shareOutstanding', 0.0):.2f} shares outstanding. {company_name} operates primarily in the {profile.get('country', 'N/A')}, trading under the ticker {ticker} on the {profile.get('exchange', 'N/A')}. As a dominant force in the {profile.get('finnhubIndustry', 'N/A')} space, the company continues to innovate and drive progress within the industry.

From {start_date_str} to {end_date_str}, {company_name}'s stock price {price_change} from {start_price:.2f} to {end_price:.2f}. Company news during this period are listed below:

{news_string}

{financials_string}

Based on all the information before {end_date_str}, let's first analyze the positive developments and potential concerns for {ticker}. Come up with 2-4 most important factors respectively and keep them concise. Most factors should be inferred from company-related news. Then make your prediction of the {ticker} stock price movement for next week. Provide a summary analysis to support your prediction."""
    
    return prompt.strip()
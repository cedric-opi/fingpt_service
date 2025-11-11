"""
Image Analysis Service
Analyzes financial charts, documents using GPT-4 Vision
"""
import logging
import requests
from typing import Optional, List
import os

logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def analyze_financial_image(image_url: str) -> Optional[str]:
    """
    Analyze a financial chart/document using GPT-4 Vision
    
    Args:
        image_url: URL of the image to analyze
        
    Returns:
        Analysis text or None if failed
    """
    if not OPENAI_API_KEY:
        logger.warning("⚠️ OPENAI_API_KEY not set, skipping image analysis")
        return None
    
    try:
        logger.info(f"📊 Analyzing image: {image_url[:50]}...")
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}",
            },
            json={
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """Analyze this financial chart or document. Extract:
- Key numbers and metrics
- Trends (up/down, percentage changes)
- Time periods shown
- Notable patterns or anomalies
- Any tickers or company names visible

Be specific and concise. Focus on actionable insights."""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 500,
                "temperature": 0.2,
            },
            timeout=30
        )
        
        response.raise_for_status()
        data = response.json()
        
        analysis = data["choices"][0]["message"]["content"]
        logger.info(f"✅ Image analyzed successfully ({len(analysis)} chars)")
        
        return analysis
        
    except Exception as e:
        logger.error(f"❌ Image analysis failed: {e}")
        return None


def analyze_multiple_images(image_urls: List[str]) -> str:
    """
    Analyze multiple images and combine results
    
    Args:
        image_urls: List of image URLs
        
    Returns:
        Combined analysis text
    """
    if not image_urls:
        return ""
    
    analyses = []
    
    for i, url in enumerate(image_urls, 1):
        analysis = analyze_financial_image(url)
        if analysis:
            analyses.append(f"📊 Image {i} Analysis:\n{analysis}")
    
    if not analyses:
        return ""
    
    return "\n\n".join(analyses)
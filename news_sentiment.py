"""
News Sentiment Analysis Module.

Uses AWS Bedrock (Claude Haiku) to analyze financial news and generate
sentiment signals that complement the topological trading features.
"""

import os
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass
import pandas as pd

# News fetching (using free sources)
try:
    import feedparser
    HAS_FEEDPARSER = True
except ImportError:
    HAS_FEEDPARSER = False

try:
    from gnews import GNews
    HAS_GNEWS = True
except ImportError:
    HAS_GNEWS = False

from bedrock_client import BedrockClient, ClaudeModel


@dataclass
class SentimentResult:
    """Structured sentiment analysis result."""
    ticker: str
    sentiment_score: float  # -1 (bearish) to 1 (bullish)
    confidence: float       # 0 to 1
    key_themes: list[str]
    news_explains_deviation: bool
    summary: str
    article_count: int


class NewsSentimentAnalyzer:
    """
    Analyzes news sentiment for stocks using Claude Haiku.
    
    Designed to be cost-effective for high-volume analysis.
    """
    
    SYSTEM_PROMPT = """You are a financial news analyst specializing in quantitative trading signals.
Your task is to analyze news articles and extract sentiment that could affect stock prices.

Be objective, precise, and focus on:
1. Material financial information (earnings, guidance, deals, lawsuits)
2. Competitive positioning changes
3. Management/leadership changes
4. Regulatory or macroeconomic impacts
5. Technical or product developments

Ignore marketing fluff and focus on actionable intelligence."""

    def __init__(self):
        self.client = BedrockClient()
        self.news_client = GNews(language='en', country='US', max_results=10) if HAS_GNEWS else None
        
    def fetch_news(self, ticker: str, company_name: Optional[str] = None, days_back: int = 7) -> list[dict]:
        """
        Fetch recent news for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            company_name: Full company name for better search
            days_back: How many days of news to fetch
            
        Returns:
            List of news articles with title, description, date
        """
        articles = []
        search_term = company_name or ticker
        
        # Method 1: GNews (if available)
        if self.news_client:
            try:
                results = self.news_client.get_news(f"{search_term} stock")
                for item in results:
                    articles.append({
                        "title": item.get("title", ""),
                        "description": item.get("description", ""),
                        "source": item.get("publisher", {}).get("title", "Unknown"),
                        "date": item.get("published date", ""),
                    })
            except Exception as e:
                print(f"GNews error for {ticker}: {e}")
        
        # Method 2: RSS feeds from major financial sources
        if HAS_FEEDPARSER and len(articles) < 5:
            rss_feeds = [
                f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
            ]
            
            for feed_url in rss_feeds:
                try:
                    feed = feedparser.parse(feed_url)
                    for entry in feed.entries[:5]:
                        articles.append({
                            "title": entry.get("title", ""),
                            "description": entry.get("summary", ""),
                            "source": feed.feed.get("title", "RSS"),
                            "date": entry.get("published", ""),
                        })
                except Exception as e:
                    print(f"RSS error for {ticker}: {e}")
                    
        return articles
    
    def analyze_sentiment(
        self,
        ticker: str,
        articles: Optional[list[dict]] = None,
        price_deviation: Optional[float] = None,
    ) -> SentimentResult:
        """
        Analyze sentiment for a ticker using Claude Haiku.
        
        Args:
            ticker: Stock ticker symbol
            articles: List of news articles (fetched if not provided)
            price_deviation: The residual/deviation from expected price (from topology model)
            
        Returns:
            SentimentResult with scores and analysis
        """
        # Fetch news if not provided
        if articles is None:
            articles = self.fetch_news(ticker)
            
        if not articles:
            return SentimentResult(
                ticker=ticker,
                sentiment_score=0.0,
                confidence=0.0,
                key_themes=[],
                news_explains_deviation=False,
                summary="No recent news found.",
                article_count=0,
            )
        
        # Format articles for the prompt
        articles_text = "\n\n".join([
            f"**{a['title']}** ({a['source']})\n{a['description']}"
            for a in articles[:10]  # Limit to 10 articles
        ])
        
        deviation_context = ""
        if price_deviation is not None:
            direction = "outperforming" if price_deviation > 0 else "underperforming"
            deviation_context = f"""
CONTEXT: This stock is currently {direction} its sector peers by {abs(price_deviation):.2%} 
based on our correlation model. Assess whether the news explains this deviation.
"""
        
        prompt = f"""Analyze the following news articles for {ticker} and provide a JSON response:

{articles_text}

{deviation_context}

Respond with this exact JSON structure:
{{
    "sentiment_score": <float from -1.0 (very bearish) to 1.0 (very bullish)>,
    "confidence": <float from 0.0 to 1.0 indicating how confident you are>,
    "key_themes": [<list of 2-4 key themes as strings>],
    "news_explains_deviation": <boolean - does news explain the price deviation?>,
    "summary": "<2-3 sentence summary of the overall sentiment and key drivers>"
}}"""

        try:
            result = self.client.invoke_for_json(
                prompt=prompt,
                model=ClaudeModel.HAIKU,  # Use Haiku for cost efficiency
                system_prompt=self.SYSTEM_PROMPT,
                max_tokens=512,
            )
            
            return SentimentResult(
                ticker=ticker,
                sentiment_score=float(result.get("sentiment_score", 0)),
                confidence=float(result.get("confidence", 0)),
                key_themes=result.get("key_themes", []),
                news_explains_deviation=result.get("news_explains_deviation", False),
                summary=result.get("summary", ""),
                article_count=len(articles),
            )
            
        except Exception as e:
            print(f"Sentiment analysis error for {ticker}: {e}")
            return SentimentResult(
                ticker=ticker,
                sentiment_score=0.0,
                confidence=0.0,
                key_themes=[],
                news_explains_deviation=False,
                summary=f"Analysis failed: {str(e)}",
                article_count=len(articles),
            )
    
    def analyze_batch(
        self,
        tickers: list[str],
        deviations: Optional[dict[str, float]] = None,
    ) -> pd.DataFrame:
        """
        Analyze sentiment for multiple tickers.
        
        Args:
            tickers: List of ticker symbols
            deviations: Optional dict mapping ticker to price deviation
            
        Returns:
            DataFrame with sentiment results for all tickers
        """
        results = []
        
        for ticker in tickers:
            deviation = deviations.get(ticker) if deviations else None
            result = self.analyze_sentiment(ticker, price_deviation=deviation)
            results.append({
                "ticker": result.ticker,
                "sentiment_score": result.sentiment_score,
                "confidence": result.confidence,
                "key_themes": ", ".join(result.key_themes),
                "news_explains_deviation": result.news_explains_deviation,
                "summary": result.summary,
                "article_count": result.article_count,
            })
            
        return pd.DataFrame(results)


# Convenience function
def get_sentiment(ticker: str, price_deviation: Optional[float] = None) -> SentimentResult:
    """Quick sentiment analysis for a single ticker."""
    analyzer = NewsSentimentAnalyzer()
    return analyzer.analyze_sentiment(ticker, price_deviation=price_deviation)


if __name__ == "__main__":
    # Test the analyzer
    print("Testing News Sentiment Analyzer...")
    
    analyzer = NewsSentimentAnalyzer()
    
    # Test with a popular stock
    test_ticker = "NVDA"
    print(f"\nFetching news for {test_ticker}...")
    articles = analyzer.fetch_news(test_ticker)
    print(f"Found {len(articles)} articles")
    
    if articles:
        print("\nAnalyzing sentiment...")
        result = analyzer.analyze_sentiment(test_ticker, articles, price_deviation=0.05)
        print(f"\nResults for {result.ticker}:")
        print(f"  Sentiment Score: {result.sentiment_score:+.2f}")
        print(f"  Confidence: {result.confidence:.0%}")
        print(f"  Key Themes: {', '.join(result.key_themes)}")
        print(f"  News Explains Deviation: {result.news_explains_deviation}")
        print(f"  Summary: {result.summary}")


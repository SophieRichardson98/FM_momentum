import pandas as pd
import numpy as np
import requests
import os
import json
from textblob import TextBlob
from datetime import datetime, timedelta
import yfinance as yf

# ------------------- Cache Management -------------------

def cache_news_path():
    return "news_cache.json"

def cache_prices_path():
    return "price_cache.csv"

def load_news_cache():
    if os.path.exists(cache_news_path()):
        with open(cache_news_path(), 'r') as f:
            return json.load(f)
    return {}

def save_news_cache(cache):
    with open(cache_news_path(), 'w') as f:
        json.dump(cache, f)

def load_price_cache():
    if os.path.exists(cache_prices_path()):
        return pd.read_csv(cache_prices_path(), parse_dates=['date'])
    return pd.DataFrame(columns=['ticker', 'date', 'open', 'close'])

def save_price_cache(df):
    df.to_csv(cache_prices_path(), index=False)

# ------------------- Data Prefetching for Single Stock -------------------

def prefetch_price_data(stock, start_date, end_date):
    """
    Download daily open/close for one stock, cache to price_cache.csv.
    """
    try:
        df = yf.download(
            stock,
            start=start_date.strftime('%Y-%m-%d'),
            end=(end_date + timedelta(days=1)).strftime('%Y-%m-%d'),
            interval='1d',
            progress=False
        )
        df = df.dropna(subset=['Open', 'Close'])
        df['ticker'] = stock
        df['date'] = df.index.strftime('%Y-%m-%d')
        out = df[['ticker','date','Open','Close']].rename(columns={'Open':'open','Close':'close'})
    except Exception as e:
        print(f"Download failed for {stock}")
    save_price_cache(out)
    return out

# ------------------- Sentiment Analysis -------------------

def get_sentiment(stock, date, api_key, base_url, news_cache, offline_mode=False):
    key = f"{stock}_{date.strftime('%Y-%m-%d')}"
    if key in news_cache:
        return news_cache[key]
    if offline_mode:
        np.random.seed(hash(key) % 2**32)
        s = np.random.uniform(-1,1)
    else:
        params = {
            'q': stock,
            'from': date.strftime('%Y-%m-%d'),
            'to': (date + timedelta(days=1)).strftime('%Y-%m-%d'),
            'sortBy': 'relevancy',
            'apiKey': api_key,
            'language': 'en',
            'pageSize': 10
        }
        try:
            resp = requests.get(base_url, params=params)
            data = resp.json()
            scores = []
            for art in data.get('articles', []):
                txt = f"{art.get('title','')} {art.get('description','')}".strip()
                if txt:
                    scores.append(TextBlob(txt).sentiment.polarity)
            s = float(np.mean(scores)) if scores else 0.0
        except Exception as e:
            print(f"NewsAPI error: {e}")
            s = 0.0
    news_cache[key] = s
    save_news_cache(news_cache)
    return s

# ------------------- Trading Simulation for Single Stock -------------------

def get_stock_prices(stock, date, price_cache):
    row = price_cache[(price_cache['ticker']==stock) & (price_cache['date']==date.strftime('%Y-%m-%d'))]
    if not row.empty:
        return float(row['open'].iloc[0]), float(row['close'].iloc[0])
    return np.nan, np.nan

def run_sentiment_strategy(
    stock,
    start_date,
    end_date,
    initial_portfolio_value,
    api_key,
    base_url,
    offline_mode=True
):
    dates = pd.date_range(start_date, end_date)
    portfolio = initial_portfolio_value
    news_cache = load_news_cache()
    price_cache = load_price_cache()
    # ensure price data available
    if price_cache.empty:
        price_cache = prefetch_price_data(stock, start_date, end_date)
    results = []
    for dt in dates:
        sent = get_sentiment(stock, dt, api_key, base_url, news_cache, offline_mode)
        o, c = get_stock_prices(stock, dt, price_cache)
        if np.isnan(o) or np.isnan(c):
            ret = 0.0
        else:
            if sent >= 0:
                ret = (c - o)/o
            else:
                ret = (o - c)/o
        portfolio *= (1 + ret)
        results.append({
            'date': dt.strftime('%Y-%m-%d'),
            'sentiment': sent,
            'open': o,
            'close': c,
            'daily_return': ret,
            'portfolio_value': portfolio
        })
    return pd.DataFrame(results)

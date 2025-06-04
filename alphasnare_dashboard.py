import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import yfinance as yf
from datetime import datetime, timedelta
import sqlite3
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import time
from scipy.stats import linregress
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
import torch
import os
import json

st.set_page_config(page_title="Sentiment Tracker", layout="wide", initial_sidebar_state="expanded")

# ------------------ STYLE ------------------
st.markdown("""
    <style>
        .big-title {
            font-size:48px !important;
            color:#007ACC;
            font-weight: 700;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='big-title'>📊 AlphaSnare Sentiment Dashboard</div>", unsafe_allow_html=True)

# ------------------ STATE & CONFIG ------------------
WEIGHT_FILE = "weights.json"
def load_weights():
    default = {"Finviz": 1.0, "Yahoo Finance": 1.0, "MarketWatch": 1.0, "WSJ": 2.0, "Economist": 2.0}
    if os.path.exists(WEIGHT_FILE):
        with open(WEIGHT_FILE, 'r') as f:
            return json.load(f)
    return default

def save_weights(weights):
    with open(WEIGHT_FILE, 'w') as f:
        json.dump(weights, f)

weights = load_weights()
st.sidebar.subheader("🛠 Developer Tools: Source Weights")
for key in weights:
    weights[key] = st.sidebar.slider(f"{key} Weight", 0.1, 5.0, weights[key], 0.1)
save_weights(weights)

# ------------------ SENTIMENT ENGINES ------------------
def analyze_sentiment_textblob(text):
    return TextBlob(text).sentiment.polarity * 100

@st.cache_resource
def load_finbert():
    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    return model, tokenizer

def analyze_sentiment_finbert(text_list):
    model, tokenizer = load_finbert()
    scores = []
    for text in text_list:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=1).detach().numpy()[0]
        score = (probs[2] - probs[0]) * 100  # pos - neg
        scores.append(score)
    return scores

# ------------------ SCRAPER UTILS ------------------
def fetch_headlines_finviz(ticker):
    headlines = []
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        table = soup.find(id="news-table")
        rows = table.find_all("tr")
        for row in rows:
            text = row.a.get_text()
            headlines.append((text, "Finviz"))
    except Exception:
        pass
    return headlines

def archive_and_scrape(url):
    try:
        search_url = f"https://archive.md/search/?q={url}"
        r = requests.get(search_url, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        archive_tag = soup.find("a", href=True)
        archive_url = archive_tag['href'] if archive_tag else None
        if not archive_url:
            submit = requests.post("https://archive.md/submit/", data={"url": url}, timeout=15)
            time.sleep(5)
            soup = BeautifulSoup(submit.text, "html.parser")
            archive_tag = soup.find("a", href=True)
            archive_url = archive_tag['href'] if archive_tag else None
        if not archive_url:
            return None
        r = requests.get(archive_url, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        paragraphs = soup.find_all("p")
        text = " ".join([p.get_text() for p in paragraphs if len(p.get_text()) > 40])
        return text
    except:
        return None

def fetch_articles_from_wsj_economist(ticker):
    results = []
    sources = {
        "WSJ": f"https://www.wsj.com/search?query={ticker}",
        "Economist": f"https://www.economist.com/search?q={ticker}",
        "Yahoo Finance": f"https://finance.yahoo.com/quote/{ticker}?p={ticker}",
        "MarketWatch": f"https://www.marketwatch.com/investing/stock/{ticker}"
    }
    for source, url in sources.items():
        text = archive_and_scrape(url)
        if text:
            results.append((text[:500], source))
    return results

# ------------------ TREND & REGRESSION ------------------
def sentiment_derivative(df):
    df_sorted = df.sort_values("Date")
    df_sorted["Trend"] = df_sorted["Sentiment Score"].diff()
    return df_sorted

def run_regression(x, y):
    slope, intercept, r, p, stderr = linregress(x, y)
    return {"slope": slope, "intercept": intercept, "r": r, "r2": r**2, "p": p, "stderr": stderr}

# ------------------ APP ------------------
st.sidebar.title("Configuration")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL)", "AAPL")
date_range = st.sidebar.date_input("Select Date Range", [datetime.today() - timedelta(days=730), datetime.today()])

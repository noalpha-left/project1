# Alphasnare Sentiment Dashboard - Full Unified Streamlit Module

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import yfinance as yf
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
from scipy.stats import linregress

st.set_page_config(page_title="Alphasnare Dashboard", layout="wide")

# ------------------- Helper Functions -------------------
def run_regression(x, y):
    slope, intercept, r, p, stderr = linregress(x, y)
    return {"slope": slope, "intercept": intercept, "r": r, "r2": r**2, "p": p, "stderr": stderr}

def sentiment_derivative(df):
    df = df.sort_values("Date")
    df["Trend"] = df["Sentiment Score"].rolling(window=3).mean().diff()
    return df

def fetch_headlines_finviz(ticker):
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.text, "html.parser")
    try:
        news_table = soup.find(id='news-table')
        rows = news_table.findAll('tr')
        return [(row.a.get_text(), "Finviz") for row in rows if row.a][:15]
    except:
        return []

def archive_and_scrape(url):
    try:
        archive_req = requests.get(f"https://archive.md/submit/?url={url}")
        archive_soup = BeautifulSoup(archive_req.text, "html.parser")
        archive_url = archive_soup.find("a", href=True)["href"]
        r = requests.get(archive_url)
        soup = BeautifulSoup(r.text, "html.parser")
        return soup.get_text(separator=' ').strip()
    except:
        return ""

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

def analyze_sentiment_textblob(text):
    return TextBlob(text).sentiment.polarity * 100

def analyze_sentiment_finbert(texts):
    return [TextBlob(t).sentiment.polarity * 100 for t in texts]  # Replace with FinBERT later if needed

# ------------------- App UI Setup -------------------
st.title("🧠 Alphasnare Sentiment Tracker")
st.sidebar.title("Configuration")
ticker = st.sidebar.text_input("Enter Ticker", value="AAPL")
date_range = st.sidebar.date_input("Select Date Range", [datetime.today() - timedelta(days=730), datetime.today()])
weights = {
    "WSJ": 2.0,
    "Economist": 2.0,
    "Finviz": 1.0,
    "Yahoo Finance": 1.5,
    "MarketWatch": 1.5
}
sentiment_filter = st.sidebar.slider("Minimum Sentiment Score", -100.0, 100.0, -100.0)
use_finbert = st.sidebar.checkbox("Use FinBERT for Sentiment", value=False)

if ticker:
    headlines = fetch_headlines_finviz(ticker)
    wsj_articles = fetch_articles_from_wsj_economist(ticker)
    all_data = headlines + wsj_articles
    today = datetime.now()
    records = []

    texts = [x[0] for x in all_data]
    sources = [x[1] for x in all_data]
    sentiments = analyze_sentiment_finbert(texts) if use_finbert else [analyze_sentiment_textblob(t) for t in texts]

    for i, (text, source) in enumerate(zip(texts, sources)):
        date = today.date() - timedelta(days=i)
        weight = weights.get(source, 1.0)
        score = sentiments[i] * weight
        records.append({"Date": date, "Text": text, "Sentiment Score": score, "Source": source})

    df = pd.DataFrame(records)
    df = df[df["Sentiment Score"] >= sentiment_filter]
    df = sentiment_derivative(df)
    df = df.sort_values("Date")

    st.subheader(f"📈 Sentiment Overview for {ticker}")
    if not df.empty:
        st.markdown(f"**Avg Sentiment:** {df['Sentiment Score'].mean():.2f}%")
        st.markdown(f"**Positive Articles:** {(df['Sentiment Score'] > 0).sum()}")
        st.markdown(f"**Negative Articles:** {(df['Sentiment Score'] < 0).sum()}")
        st.markdown(f"**Sources Used:** {df['Source'].nunique()}")
        st.dataframe(df)

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(df["Date"], df["Sentiment Score"], label="Sentiment")
        ax.plot(df["Date"], df["Trend"], label="Trend Derivative", linestyle="--")
        ax.set_title("Sentiment & Trend Over Time")
        ax.set_xlabel("Date")
        ax.xaxis.set_major_locator(plt.MaxNLocator(6))
        ax.legend()
        st.pyplot(fig)

        reg = run_regression(df["Sentiment Score"], range(len(df)))
        st.markdown("### 📐 Regression Stats")
        st.markdown(f"- **R**: {round(reg['r'], 3)}")
        st.markdown(f"- **R²**: {round(reg['r2'], 3)}")
        st.markdown(f"- **p-value**: {round(reg['p'], 5)}")
        st.markdown(f"- **Std Error**: {round(reg['stderr'], 3)}")

        df["Signal"] = df["Sentiment Score"].rolling(3).mean().shift(1)
        df["Position"] = 0
        df.loc[df["Signal"] > 20, "Position"] = 1
        df.loc[df["Signal"] < -20, "Position"] = -1
        df["Returns"] = df["Sentiment Score"].pct_change()
        df["Strategy"] = df["Returns"] * df["Position"].shift(1)
        df = df.fillna(0)
        df["Cumulative"] = (1 + df["Returns"]).cumprod()
        df["Strategy Cum"] = (1 + df["Strategy"]).cumprod()

        col1, col2 = st.columns([3, 1])
        with col2:
            st.subheader("💡 Financials")
            try:
                ticker_obj = yf.Ticker(ticker)
                fin_df = ticker_obj.financials.T
                st.markdown("**Yearly Revenue (B):** " + str(round(fin_df['Total Revenue'].iloc[-1]/1e9, 2)))
                st.markdown("**EBITDA (B):** " + str(round(fin_df['EBITDA'].iloc[-1]/1e9, 2)))
                st.markdown("**Net Margin (%):** " + str(round((fin_df['Net Income'].iloc[-1] / fin_df['Total Revenue'].iloc[-1]) * 100, 2)))
            except:
                st.markdown("Financial data unavailable.")
        with col1:
            st.subheader("📊 Backtest: Cumulative Strategy")
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(df["Date"], df["Cumulative"], label="Market", color="gray")
            ax.plot(df["Date"], df["Strategy Cum"], label="Strategy", color="blue")
            ax.set_title("Cumulative Returns vs Strategy")
            ax.xaxis.set_major_locator(plt.MaxNLocator(6))
            ax.legend()
            st.pyplot(fig)

        try:
            fin_df = ticker_obj.financials.T
            earnings = fin_df[['Total Revenue', 'EBITDA']].dropna()
            if not earnings.empty:
                x = earnings['Total Revenue'].values.astype(float)
                y = earnings['EBITDA'].values.astype(float)
                fin_reg = run_regression(x, y)
                st.markdown("**📊 Earnings Regression (EBITDA vs Revenue)**")
                st.markdown(f"- **R**: {round(fin_reg['r'], 3)}")
                st.markdown(f"- **R²**: {round(fin_reg['r2'], 3)}")
        except:
            st.markdown("_Financial regression failed._")

        st.markdown(f"**🟢 Live Signal:** {'Bullish' if reg['slope'] > 0 else 'Bearish' if reg['slope'] < 0 else 'Neutral'}")

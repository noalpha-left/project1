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

st.set_page_config(page_title="Sentiment Tracker", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
    <style>
        .big-title {
            font-size:48px !important;
            color:#007ACC;
            font-weight: 700;
        }
        .metric-style div {
            font-size: 18px !important;
        }
        .block-container {
            padding: 2rem 3rem 3rem;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='big-title'>📊 AlphaSnare Sentiment Dashboard</div>", unsafe_allow_html=True)

# ---------------------------- DB UTILS ----------------------------
def init_db():
    conn = sqlite3.connect("sentiment_data.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS sentiment (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            ticker TEXT,
            source TEXT,
            text TEXT,
            sentiment_score REAL,
            weighted_score REAL
        )
    """)
    conn.commit()
    conn.close()

def insert_record(date, ticker, source, text, sentiment_score, weighted_score):
    conn = sqlite3.connect("sentiment_data.db")
    c = conn.cursor()
    c.execute("""
        INSERT INTO sentiment (date, ticker, source, text, sentiment_score, weighted_score)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (date, ticker, source, text, sentiment_score, weighted_score))
    conn.commit()
    conn.close()

def load_sentiment_history(ticker=None):
    conn = sqlite3.connect("sentiment_data.db")
    query = "SELECT * FROM sentiment"
    if ticker:
        query += " WHERE ticker = ?"
        df = pd.read_sql_query(query, conn, params=(ticker,))
    else:
        df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# ------------------------ FETCH DATA -----------------------------
def fetch_headlines(ticker):
    headlines = []
    headers = {'User-Agent': 'Mozilla/5.0'}

    try:
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        news_table = soup.find(id='news-table')
        rows = news_table.findAll('tr')
        headlines = [row.a.get_text() for row in rows if row.a][:15]
        if headlines:
            return headlines, "Finviz"
    except:
        pass

    try:
        url = f"https://finance.yahoo.com/quote/{ticker}?p={ticker}"
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        tags = soup.find_all("h3")
        headlines = [tag.get_text() for tag in tags if len(tag.get_text()) > 20][:15]
        if headlines:
            return headlines, "Yahoo Finance"
    except:
        pass

    try:
        url = f"https://www.marketwatch.com/investing/stock/{ticker}"
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        articles = soup.find_all("h3")
        headlines = [a.get_text() for a in articles if len(a.get_text()) > 20][:15]
        if headlines:
            return headlines, "MarketWatch"
    except:
        pass

    return [], "None"

# ------------------------ ARCHIVE.SCRAPE -------------------------
def archive_and_scrape(url):
    try:
        search_url = f"https://archive.md/search/?q={url}"
        r = requests.get(search_url, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        archived_links = soup.select("a[href*='archive.today'], a[href*='archive.md']")

        if archived_links:
            archive_url = archived_links[0]['href']
        else:
            submit = requests.post("https://archive.md/submit/", data={"url": url}, timeout=15)
            time.sleep(5)
            soup = BeautifulSoup(submit.text, "html.parser")
            archive_tag = soup.find("a", href=True)
            archive_url = archive_tag['href'] if archive_tag else None

        if not archive_url:
            return None, None

        r = requests.get(archive_url, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        paragraphs = soup.find_all("p")
        article_text = " ".join([p.get_text() for p in paragraphs if len(p.get_text()) > 40])
        return article_text, archive_url

    except Exception as e:
        print("Archive scrape failed:", e)
        return None, None

# ------------------------ SENTIMENT ENGINES ----------------------
def analyze_sentiment(text_list):
    sentiments = []
    for text in text_list:
        blob = TextBlob(text)
        score = blob.sentiment.polarity
        sentiments.append(score * 100)
    return sentiments

@st.cache_resource
def load_finbert():
    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    return model, tokenizer

def finbert_sentiment(text_list):
    model, tokenizer = load_finbert()
    scores = []
    for text in text_list:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=1).detach().numpy()[0]
        sentiment_score = (probs[2] - probs[0]) * 100
        scores.append(sentiment_score)
    return scores

# ---------------------------- APP LOGIC --------------------------

init_db()

st.sidebar.title("Configuration")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL)", "AAPL")
days_back = st.sidebar.slider("Days of headlines to analyze", 1, 30, 7)
sentiment_filter = st.sidebar.slider("Minimum Sentiment Score", -100.0, 100.0, -100.0)
date_range = st.sidebar.date_input("Select Date Range", [datetime.today() - timedelta(days=30), datetime.today()])
use_finbert = st.sidebar.checkbox("Use FinBERT for Sentiment", value=False)

st.sidebar.markdown("---")
st.sidebar.subheader("🗞️ Manual Article Sentiment")
manual_url = st.sidebar.text_input("Paste WSJ or Economist URL")

if st.sidebar.button("Analyze URL"):
    with st.spinner("Archiving and analyzing article..."):
        article_text, archive_link = archive_and_scrape(manual_url)
        if article_text:
            score = finbert_sentiment([article_text])[0] if use_finbert else analyze_sentiment([article_text])[0]
            today = datetime.now().date()
            insert_record(today, "MANUAL", "WSJ/Economist", article_text, score, score)
            st.success(f"Archived from: {archive_link}")
            st.write(article_text[:1000] + "...")
            st.metric("Sentiment Score (%)", round(score, 2))
        else:
            st.error("Failed to fetch or archive article. Try another URL.")

if st.sidebar.button("Analyze"):
    with st.spinner("Processing data sources..."):
        headlines, source_used = fetch_headlines(ticker)
        if not headlines:
            st.error("No headlines found. Try another ticker.")
        else:
            today = datetime.now()
            sentiments = finbert_sentiment(headlines) if use_finbert else analyze_sentiment(headlines)
            records = []
            for i, text in enumerate(headlines):
                date = today.date() - timedelta(days=i)
                score = sentiments[i]
                weight = 1 / (i + 1)
                weighted_score = score * weight
                insert_record(date, ticker, source_used, text, score, weighted_score)
                records.append({"Date": date, "Text": text, "Sentiment Score": score, "Weighted Score": weighted_score})

            df = pd.DataFrame(records)
            df = df[df["Sentiment Score"] >= sentiment_filter]

            if isinstance(date_range, list) and len(date_range) == 2:
                df = df[(df["Date"] >= pd.to_datetime(date_range[0]).date()) & (df["Date"] <= pd.to_datetime(date_range[1]).date())]

            st.subheader(f"📈 Sentiment Overview for {ticker} from {source_used}")
            st.dataframe(df, use_container_width=True)

            avg_sent = round(df["Sentiment Score"].mean(), 2)
            avg_weighted = round(df["Weighted Score"].mean(), 2)
            col1, col2 = st.columns(2)
            col1.metric("Avg Sentiment (%)", avg_sent)
            col2.metric("Weighted Avg (%)", avg_weighted)

            sentiment_time_df = df.groupby("Date")["Sentiment Score"].mean().sort_index().reset_index()
sentiment_time_df["Date"] = pd.to_datetime(sentiment_time_df["Date"]).dt.date
price_data = yf.download(ticker, period="6mo")
            price_data.reset_index(inplace=True)
            price_data["Date"] = price_data["Date"].dt.date
            merged = pd.merge(sentiment_time_df, price_data[["Date", "Close"]], on="Date", how="inner")

            st.subheader("📉 Sentiment vs. Price Trend")
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            ax1.plot(merged["Date"], merged["Sentiment Score"], color='blue', label='Sentiment')
            ax2.plot(merged["Date"], merged["Close"], color='green', label='Price')
            ax1.set_ylabel("Sentiment (%)")
            ax2.set_ylabel("Close Price ($)")
            ax1.set_xlabel("Date")
            fig.autofmt_xdate()
            st.pyplot(fig)

            if not merged.empty:
                slope, intercept, r_value, p_value, std_err = linregress(merged["Sentiment Score"], merged["Close"])
                st.markdown(f"**📐 Correlation Stats:**")
                st.markdown(f"- R: `{round(r_value, 3)}`")
                st.markdown(f"- R²: `{round(r_value**2, 3)}`")
                st.markdown(f"- p-value: `{round(p_value, 5)}`")
                st.markdown(f"- Std. Error: `{round(std_err, 3)}`")

                st.subheader("📊 Backtest: Sentiment Strategy")
                merged["Signal"] = merged["Sentiment Score"].rolling(3).mean().shift(1)
                merged["Position"] = 0
                merged.loc[merged["Signal"] > 20, "Position"] = 1
                merged.loc[merged["Signal"] < -20, "Position"] = -1
                merged["Returns"] = merged["Close"].pct_change()
                merged["Strategy"] = merged["Returns"] * merged["Position"].shift(1)
                merged[["Returns", "Strategy"]] = merged[["Returns", "Strategy"]].fillna(0)
                merged["Cumulative Market"] = (1 + merged["Returns"]).cumprod()
                merged["Cumulative Strategy"] = (1 + merged["Strategy"]).cumprod()

                fig, ax = plt.subplots()
                ax.plot(merged["Date"], merged["Cumulative Market"], label="Market", color="gray")
                ax.plot(merged["Date"], merged["Cumulative Strategy"], label="Strategy", color="blue")
                ax.set_title("Backtest: Cumulative Returns")
                ax.legend()
                st.pyplot(fig)

            st.subheader("🔤 Most Common Words")
            word_text = " ".join(df["Text"])
            wc = WordCloud(width=800, height=300, background_color='white').generate(word_text)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)

            st.subheader("📈 Stock Price Performance (2Y)")
            long_price = yf.download(ticker, period="2y")
            st.line_chart(long_price['Close'])

            st.subheader("🚦 Trading Signal (Beta)")
            if avg_weighted > 30:
                st.success("Bullish sentiment detected.")
            elif avg_weighted < -30:
                st.error("Bearish sentiment detected.")
            else:
                st.warning("Neutral zone.")
else:
    st.info("Enter a ticker and click 'Analyze' to begin.")

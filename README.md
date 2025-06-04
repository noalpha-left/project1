# AlphaSnare: Sentiment-Driven Alpha Insights Dashboard

AlphaSnare is a financial sentiment analysis dashboard built with Streamlit. It scrapes financial news headlines, analyzes sentiment using TextBlob and FinBERT, and visualizes insights including price correlation, backtesting performance, and sentiment trends.

## 🔧 Features

- Multi-source news headline scraping (Finviz, Yahoo Finance, MarketWatch)
- Manual article scraping from WSJ/The Economist via Archive.md
- Sentiment analysis using:
  - TextBlob (default)
  - FinBERT (toggle option)
- Sentiment vs. stock price overlay
- Linear regression and correlation stats (R, R², p-value)
- Backtested signal strategy based on rolling sentiment average
- WordCloud and historical sentiment visualization
- Live trading signal simulation

## 📦 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
python -m textblob.download_corpora
```

## 🚀 Running the App

```bash
streamlit run alphasnare_dashboard.py
```

## 🧠 Notes

- Use the FinBERT toggle in the sidebar to switch between sentiment engines.
- The app stores results in a local SQLite database.
- Ideal for analysts, traders, and data-driven investors.

## 🖼 Preview

<img src="preview.png" alt="dashboard screenshot" width="800"/>

---

© 2025 AlphaSnare

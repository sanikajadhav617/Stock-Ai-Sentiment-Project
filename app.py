from flask import Flask, render_template, request
import requests
from textblob import TextBlob
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

app = Flask(__name__)

API_KEY = "355e4c8a262249faa521048af76d5e67"


@app.route('/')
def welcome():
    return render_template("welcome.html")


@app.route('/input')
def input_page():
    return render_template("input.html")


@app.route('/analyze', methods=['POST'])
def analyze():

    stocks = request.form['stock'].upper().split(",")
    results = []

    for stock in stocks:

        stock = stock.strip()

        # ---------------- SENTIMENT + HEADLINES ----------------
        url = f"https://newsapi.org/v2/everything?q={stock}&apiKey={API_KEY}"
        response = requests.get(url)
        news = response.json()
        articles = news.get("articles", [])[:5]

        sentiment_total = 0
        count = 0
        headlines = []

        for a in articles:
            title = a.get("title", "")
            if title:
                headlines.append(title)
                sentiment_total += TextBlob(title).sentiment.polarity
                count += 1

        avg_sentiment = sentiment_total / count if count > 0 else 0

        # ---------------- STOCK DATA ----------------
        df = yf.download(stock, period="6mo", auto_adjust=True)

        if df.empty:
            continue

        df = df.copy()
        df.columns = df.columns.get_level_values(0)

        df = df[['Close']].copy()
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df.dropna(inplace=True)

        df['MA'] = df['Close'].rolling(5).mean()

        df.dropna(inplace=True)

        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df.dropna(inplace=True)

        if len(df) < 20:
            continue

        X = df[['Close', 'MA']].values
        y = df['Target'].values

        split = int(len(df) * 0.8)

        X_train = X[:split]
        X_test = X[split:]
        y_train = y[:split]
        y_test = y[split:]

        model = LogisticRegression()
        model.fit(X_train, y_train)

        acc = accuracy_score(y_test, model.predict(X_test))

        latest_close = float(df['Close'].iloc[-1])
        latest_ma = float(df['MA'].iloc[-1])

        latest_input = np.array([[latest_close, latest_ma]])

        ml_pred = model.predict(latest_input)[0]

        combined_score = (ml_pred * 0.6) + (avg_sentiment * 0.4)

        if combined_score > 0.3:
            decision = "BUY"
        elif combined_score < -0.3:
            decision = "SELL"
        else:
            decision = "HOLD"

        confidence = round(abs(combined_score) * 100, 2)

        volatility = df['Close'].pct_change().std()

        if volatility < 0.01:
            risk = "Low Risk"
        elif volatility < 0.02:
            risk = "Medium Risk"
        else:
            risk = "High Risk"

        # -------- PRICE CHART --------
        plt.figure()
        plt.plot(df['Close'].values)
        plt.plot(df['MA'].values)
        plt.legend(["Close", "MA"])
        plt.title(stock + " Price Trend")
        plt.savefig("static/chart.png")
        plt.close()

        results.append({
            "stock": stock,
            "decision": decision,
            "confidence": confidence,
            "risk": risk,
            "accuracy": round(acc * 100, 2),
            "sentiment": round(avg_sentiment, 3),
            "headlines": headlines
        })

    return render_template("dashboard.html", results=results)


if __name__ == "__main__":
    app.run(debug=True)

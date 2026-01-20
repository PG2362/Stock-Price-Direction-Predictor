
import requests
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np

API_KEY = "0V0Z98H00MNX1SJY"
SYMBOL = "AAPL"
def fetch_daily_data(symbol=SYMBOL):
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": SYMBOL,
        "apikey" : API_KEY,
        "outputsize": "100"
    }

    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame.from_dict(
        data["Time Series (Daily)"], orient="index"
    )

    df = df.astype(float)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df

df = fetch_daily_data()
df.columns = ["open", "high", "low", "close", "volume"]
df["return"] = df["close"].pct_change()
df["target"] = (df["close"].shift(-1)>df["close"]).astype(int)

df["sma_5"] = df["close"].rolling(5).mean()
df["sma_10"] = df["close"].rolling(10).mean()
df["ema_10"] = df["close"].ewm(span=10, adjust=False).mean()
df["volatility_5"] = df["return"].rolling(5).std()
df["volume_change"] = df["volume"].pct_change()



for lag in [1, 2, 3, 5]:
    df[f"return_lag_{lag}"] = df["return"].shift(lag)
df = df.dropna()
print(df.tail())
features = [
    "return",
    "sma_5",
    "sma_10",
    "ema_10",
    "volatility_5",
    "volume_change",
    "return_lag_1",
    "return_lag_2",
    "return_lag_3",
    "return_lag_5"
]

X = df[features]
y = df["target"]

split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, preds))
print("AUC:", roc_auc_score(y_test, probs))
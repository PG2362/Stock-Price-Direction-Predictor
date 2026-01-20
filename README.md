# Stock-Price-Direction-Predictor
# Stock Price Direction Prediction using Logistic Regression

This project builds an end-to-end machine learning pipeline to predict the **next-day price direction** (up or down) of a stock using historical market data and technical indicators.

It demonstrates real-world skills in **API data ingestion, feature engineering, time-series modeling, and model evaluation**.

---

## Project Overview

- Fetches daily stock price data using the **Alpha Vantage API**
- Performs data cleaning and preprocessing with **pandas**
- Engineers technical indicators such as moving averages, volatility, and lagged returns
- Trains a **logistic regression classifier** to predict whether the stock price will increase the next day
- Evaluates performance using **accuracy** and **ROC-AUC**

---

## Features Engineered

The model uses the following features:

- Daily return  
- 5-day Simple Moving Average (SMA)  
- 10-day Simple Moving Average (SMA)  
- 10-day Exponential Moving Average (EMA)  
- 5-day rolling volatility  
- Volume percentage change  
- Lagged returns (1, 2, 3, and 5 days)

---

## Tech Stack

- Python  
- pandas, NumPy  
- scikit-learn  
- requests (API calls)  
- Alpha Vantage API  

---

## Model

- Algorithm: **Logistic Regression**
- Train/Test Split: **80/20 (time-series split to prevent data leakage)**
- Metrics:
  - Accuracy
  - ROC-AUC score

---

## How It Works

1. Pull daily stock data from Alpha Vantage  
2. Convert JSON to structured DataFrame  
3. Generate technical indicators  
4. Create binary target variable:
   - `1` if next day closing price increases  
   - `0` otherwise  
5. Train model on historical data  
6. Evaluate on unseen future data  



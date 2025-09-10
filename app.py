from flask import Flask, render_template, jsonify
import requests
from dotenv import load_dotenv
import os
from config import COMDAILY_API_URL, COMDAILY_AUTH, SYMBOL
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from datetime import datetime, timedelta
import threading
import time

#this file is the correct and working current retake project with an implemented working ai

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

load_dotenv()

app = Flask(__name__)

# Global cache for predictions
prediction_cache = {}
cache_lock = threading.Lock()
CACHE_DURATION = 3600  # 1 hour in seconds

def get_cache_key(symbol):
    current_hour = datetime.now().strftime("%Y%m%d%H")
    return f"{symbol}_{current_hour}"

from functools import lru_cache

@lru_cache(maxsize=10)
def fetch_data(endpoint):
    url = f"{COMDAILY_API_URL}/{endpoint}/{SYMBOL}"
    print(f"Attempting to fetch from: {url}")
    try:
        res = requests.get(url, auth=COMDAILY_AUTH)
        print(f"Response status: {res.status_code}")
        res.raise_for_status()
        if endpoint in ["income-statement", "key-metrics", "price/annual", "cash-flow-statement", "ratios"]:
            return res.json().get("message", [])
        return res.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as e:
        print(f"Other error occurred: {e}")
    return []

def fetch_key_metrics(symbol):
    data = fetch_data("key-metrics")
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    return df

def fetch_annual_prices(symbol):
    data = fetch_data("price/annual")
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    elif 'calendarYear' in df.columns:
        df['date'] = pd.to_datetime(df['calendarYear'].astype(str) + '-12-31')
    else:
        return pd.DataFrame()
    df.set_index('date', inplace=True)
    return df

def fetch_cash_flow(symbol):
    data = fetch_data("cash-flow-statement")
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    elif 'calendarYear' in df.columns:
        df['date'] = pd.to_datetime(df['calendarYear'].astype(str) + '-12-31')
    else:
        return pd.DataFrame()
    df.set_index('date', inplace=True)
    return df

def fetch_income_statement(symbol):
    data = fetch_data("income-statement")
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    elif 'calendarYear' in df.columns:
        df['date'] = pd.to_datetime(df['calendarYear'].astype(str) + '-12-31')
    else:
        return pd.DataFrame()
    df.set_index('date', inplace=True)
    return df

def fetch_ratios(symbol):
    data = fetch_data("ratios")
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    elif 'calendarYear' in df.columns:
        df['date'] = pd.to_datetime(df['calendarYear'].astype(str) + '-12-31')
    else:
        return pd.DataFrame()
    df.set_index('date', inplace=True)
    return df

def create_multivariate_sequences(features, target, seq_length=3):
    X, y = [], []
    for i in range(len(features) - seq_length):
        X.append(features[i:i+seq_length])
        y.append(target[i+seq_length])
    return np.array(X), np.array(y)

def extrapolate_features(features_scaled, feature_cols, years_to_predict=10):
    np.random.seed(42)
    growth_rates = []
    for i in range(features_scaled.shape[1]):
        feature_values = features_scaled[:, i]
        if len(feature_values) > 1:
            growth_rate = np.mean(np.diff(feature_values))
        else:
            growth_rate = 0
        growth_rates.append(growth_rate)
    future_features = []
    last_values = features_scaled[-1, :]
    for year in range(years_to_predict):
        noise_factor = np.random.normal(1, 0.05, len(growth_rates))
        new_values = last_values + np.array(growth_rates) * noise_factor
        new_values = np.clip(new_values, 0, 1)
        future_features.append(new_values)
        last_values = new_values
    return np.array(future_features)

def predict_future_prices(model, features_scaled, target_scaler, seq_length=3, years_to_predict=10):
    last_sequence = features_scaled[-seq_length:].copy()
    future_features = extrapolate_features(features_scaled, None, years_to_predict)
    predictions = []
    current_sequence = last_sequence.copy()
    for year in range(years_to_predict):
        X_pred = current_sequence.reshape(1, seq_length, current_sequence.shape[1])
        pred_scaled = model.predict(X_pred, verbose=0)
        predictions.append(pred_scaled[0, 0])
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1] = future_features[year]
    predictions_array = np.array(predictions).reshape(-1, 1)
    predictions_unscaled = target_scaler.inverse_transform(predictions_array)
    return predictions_unscaled.flatten()

def create_future_dates(last_date, years=10):
    future_dates = []
    for i in range(1, years + 1):
        future_date = last_date + timedelta(days=365 * i)
        future_dates.append(future_date)
    return future_dates

def build_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(32, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

def generate_ai_predictions():
    cache_key = get_cache_key(SYMBOL)
    with cache_lock:
        if cache_key in prediction_cache:
            cached_result, timestamp = prediction_cache[cache_key]
            if time.time() - timestamp < CACHE_DURATION:
                print("Returning cached predictions")
                return cached_result
    try:
        print("Generating new AI predictions...")
        df_metrics = fetch_key_metrics(SYMBOL)
        df_prices = fetch_annual_prices(SYMBOL)
        df_cash_flow = fetch_cash_flow(SYMBOL)
        df_income = fetch_income_statement(SYMBOL)
        df_ratios = fetch_ratios(SYMBOL)

        if df_prices.empty or 'close' not in df_prices.columns:
            return {"error": "No price data available"}

        df_merged = df_prices[['close']].copy()

        if not df_metrics.empty and 'revenuePerShare' in df_metrics.columns:
            df_merged = df_merged.join(df_metrics[['revenuePerShare']], how='inner')

        if not df_cash_flow.empty:
            cash_flow_cols = [col for col in ['operatingCashFlow', 'freeCashFlow'] if col in df_cash_flow.columns]
            if cash_flow_cols:
                df_merged = df_merged.join(df_cash_flow[cash_flow_cols], how='inner')

        if not df_income.empty:
            income_cols = [col for col in ['netIncome', 'totalRevenue', 'grossProfit'] if col in df_income.columns]
            if income_cols:
                df_merged = df_merged.join(df_income[income_cols], how='inner')

        if not df_ratios.empty:
            ratio_cols = [col for col in ['currentRatio', 'debtEquityRatio', 'returnOnEquity'] if col in df_ratios.columns]
            if ratio_cols:
                df_merged = df_merged.join(df_ratios[ratio_cols], how='inner')

        df_model = df_merged.dropna()

        if len(df_model) < 5:
            return {"error": f"Insufficient data. Only {len(df_model)} years available, need at least 5."}

        df_model = df_model.sort_index()

        feature_cols = [col for col in df_model.columns if col != 'close']
        target_col = 'close'

        if not feature_cols:
            df_model['price_change'] = df_model['close'].pct_change().fillna(0)
            df_model['price_ma'] = df_model['close'].rolling(window=2).mean().fillna(df_model['close'])
            feature_cols = ['price_change', 'price_ma']
            df_model = df_model.dropna()

        feature_scaler = MinMaxScaler(feature_range=(0, 1))
        target_scaler = MinMaxScaler(feature_range=(0, 1))

        features_scaled = feature_scaler.fit_transform(df_model[feature_cols])
        target_values = df_model[target_col].values.reshape(-1, 1)
        target_scaled = target_scaler.fit_transform(target_values)
        target_scaled_flat = target_scaled.flatten()

        seq_length = min(3, len(features_scaled) - 2)
        X, y = create_multivariate_sequences(features_scaled, target_scaled_flat, seq_length)

        if len(X) == 0:
            return {"error": "Not enough data to create training sequences"}

        if len(X) > 2:
            train_size = max(1, int(len(X) * 0.8))
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
        else:
            X_train, X_test = X, np.array([])
            y_train, y_test = y, np.array([])

        input_shape = (X_train.shape[1], X_train.shape[2])
        model = build_model(input_shape)

        print("Training model...")
        model.fit(X_train, y_train, epochs=30, batch_size=1, verbose=0)

        future_predictions = predict_future_prices(model, features_scaled, target_scaler, seq_length, 10)

        last_date = df_model.index[-1]
        future_dates = create_future_dates(last_date, 10)

        current_price = float(df_model[target_col].iloc[-1])
        final_price = float(future_predictions[-1])
        total_return = ((final_price - current_price) / current_price) * 100
        annual_return = ((final_price / current_price) ** (1/10) - 1) * 100

        predictions_data = []
        for i, (date, price) in enumerate(zip(future_dates, future_predictions)):
            predictions_data.append({
                "year": i + 1,
                "date": date.strftime("%Y-%m-%d"),
                "predicted_price": round(float(price), 2)
            })

        price_changes = df_model[target_col].pct_change().dropna()
        volatility = np.std(price_changes) if len(price_changes) > 0 else 0.2

        confidence_intervals = []
        for i, pred in enumerate(future_predictions):
            conf = 1.96 * volatility * np.sqrt(i + 1)
            lower_bound = max(0, pred * (1 - conf))
            upper_bound = pred * (1 + conf)
            confidence_intervals.append({
                "year": i + 1,
                "lower_bound": round(float(lower_bound), 2),
                "upper_bound": round(float(upper_bound), 2)
            })

        result = {
            "success": True,
            "symbol": SYMBOL,
            "current_price": round(current_price, 2),
            "predicted_final_price": round(final_price, 2),
            "total_return": round(total_return, 2),
            "annual_return": round(annual_return, 2),
            "predictions": predictions_data,
            "confidence_intervals": confidence_intervals,
            "volatility": round(volatility * 100, 2),
            "data_points_used": len(df_model),
            "features_used": feature_cols,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        with cache_lock:
            prediction_cache[cache_key] = (result, time.time())
            current_time = time.time()
            keys_to_remove = [k for k, (_, timestamp) in prediction_cache.items() 
                              if current_time - timestamp > CACHE_DURATION]
            for k in keys_to_remove:
                del prediction_cache[k]

        return result

    except Exception as e:
        print(f"Error in AI prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": f"Prediction failed: {str(e)}"}

def compare_historical_to_latest(income_data):
    if not income_data or len(income_data) < 2:
        return None
    try:
        oldest = income_data[-1]
        latest = income_data[0]
        comparison = {
            "start_year": oldest["date"][:4],
            "end_year": latest["date"][:4],
            "revenue": "↑" if float(latest["revenue"]) > float(oldest["revenue"]) else "↓",
            "netIncome": "↑" if float(latest["netIncome"]) > float(oldest["netIncome"]) else "↓",
            "eps": "↑" if float(latest.get("eps", 0)) > float(oldest.get("eps", 0)) else "↓"
        }
        return comparison
    except (KeyError, ValueError):
        return None

@app.route("/")
def index():
    try:
        print("=== Starting index route ===")
        
        income = fetch_data("income-statement")
        print(f"Income data: {len(income) if income else 0} items")
        
        metrics = fetch_data("key-metrics")
        print(f"Metrics data: {len(metrics) if metrics else 0} items")
        
        trend_comparison = compare_historical_to_latest(income) if income else None
        print(f"Trend comparison: {'Available' if trend_comparison else 'Not available'}")
        
        print("Generating AI predictions...")
        ai_predictions = generate_ai_predictions()
        print(f"AI predictions result: {type(ai_predictions)}")
        
        print("=== Rendering template ===")
        return render_template("index.html", 
                             income=income, 
                             metrics=metrics, 
                             trend_comparison=trend_comparison,
                             ai_predictions=ai_predictions,
                             symbol=SYMBOL)
    
    except Exception as e:
        print(f"Error in index route: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"<h1>Error</h1><p>{str(e)}</p><pre>{traceback.format_exc()}</pre>"

@app.route("/api/predictions")
def api_predictions():
    predictions = generate_ai_predictions()
    return jsonify(predictions)

@app.route("/predictions")
def predictions_page():
    ai_predictions = generate_ai_predictions()
    return render_template("predictions.html", 
                           ai_predictions=ai_predictions,
                           symbol=SYMBOL)

@app.route("/clear-cache")
def clear_cache():
    global prediction_cache
    with cache_lock:
        prediction_cache.clear()
    fetch_data.cache_clear()
    return jsonify({"message": "Cache cleared successfully"})

if __name__ == "__main__":
    app.run(debug=True)

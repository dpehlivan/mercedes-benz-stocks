import requests
import pandas as pd
import numpy as np
from config import COMDAILY_API_URL, COMDAILY_AUTH, SYMBOL
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def fetch_key_metrics(symbol):
    url = f"{COMDAILY_API_URL}/key-metrics/{symbol}"
    res = requests.get(url, auth=COMDAILY_AUTH)
    res.raise_for_status()
    json_data = res.json()
    data = json_data.get("message", [])
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

def fetch_annual_prices(symbol):
    url = f"{COMDAILY_API_URL}/price/annual/{symbol}"
    res = requests.get(url, auth=COMDAILY_AUTH)
    res.raise_for_status()
    json_data = res.json()
    data = json_data.get("message", [])
    df = pd.DataFrame(data)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    elif 'calendarYear' in df.columns:
        df['date'] = pd.to_datetime(df['calendarYear'].astype(str) + '-12-31')
    else:
        raise ValueError("No date or calendarYear column found in annual price data")
    df.set_index('date', inplace=True)
    return df

def fetch_cash_flow(symbol):
    url = f"{COMDAILY_API_URL}/cash-flow-statement/{symbol}"
    res = requests.get(url, auth=COMDAILY_AUTH)
    res.raise_for_status()
    json_data = res.json()
    data = json_data.get("message", [])
    df = pd.DataFrame(data)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    elif 'calendarYear' in df.columns:
        df['date'] = pd.to_datetime(df['calendarYear'].astype(str) + '-12-31')
    else:
        raise ValueError("No date or calendarYear column found in cash flow data")
    df.set_index('date', inplace=True)
    return df

def fetch_income_statement(symbol):
    url = f"{COMDAILY_API_URL}/income-statement/{symbol}"
    res = requests.get(url, auth=COMDAILY_AUTH)
    res.raise_for_status()
    json_data = res.json()
    data = json_data.get("message", [])
    df = pd.DataFrame(data)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    elif 'calendarYear' in df.columns:
        df['date'] = pd.to_datetime(df['calendarYear'].astype(str) + '-12-31')
    else:
        raise ValueError("No date or calendarYear column found in income statement data")
    df.set_index('date', inplace=True)
    return df

def fetch_ratios(symbol):
    url = f"{COMDAILY_API_URL}/ratios/{symbol}"
    res = requests.get(url, auth=COMDAILY_AUTH)
    res.raise_for_status()
    json_data = res.json()
    data = json_data.get("message", [])
    df = pd.DataFrame(data)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    elif 'calendarYear' in df.columns:
        df['date'] = pd.to_datetime(df['calendarYear'].astype(str) + '-12-31')
    else:
        raise ValueError("No date or calendarYear column found in ratios data")
    df.set_index('date', inplace=True)
    return df

def scale_columns(df, columns):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[columns])
    scaled_df = pd.DataFrame(scaled_data, columns=[f"{col}_scaled" for col in columns], index=df.index)
    return scaled_df, scaler

def create_multivariate_sequences(features, target, seq_length=3):
    """
    Create sequences of features and corresponding targets for LSTM.
    seq_length=3 means using 3 years of data to predict the next year's close.
    """
    X, y = [], []
    for i in range(len(features) - seq_length):
        X.append(features[i:i+seq_length])
        y.append(target[i+seq_length])
    return np.array(X), np.array(y)

def extrapolate_features(features_scaled, feature_cols, years_to_predict=10):
    """
    Extrapolate features for future years using trend analysis.
    This is a simple approach - you might want to use more sophisticated methods.
    """
    # Calculate growth rates for each feature based on historical data
    growth_rates = []
    for i in range(features_scaled.shape[1]):
        feature_values = features_scaled[:, i]
        # Calculate average year-over-year growth rate
        if len(feature_values) > 1:
            growth_rate = np.mean(np.diff(feature_values))
        else:
            growth_rate = 0
        growth_rates.append(growth_rate)
    
    # Generate future features
    future_features = []
    last_values = features_scaled[-1, :]
    
    for year in range(years_to_predict):
        # Apply growth rates with some randomness to simulate uncertainty
        noise_factor = np.random.normal(1, 0.1, len(growth_rates))  # 10% noise
        new_values = last_values + np.array(growth_rates) * noise_factor
        
        # Ensure values stay within [0, 1] range (since data is scaled)
        new_values = np.clip(new_values, 0, 1)
        
        future_features.append(new_values)
        last_values = new_values
    
    return np.array(future_features)

def predict_future_prices(model, features_scaled, target_scaler, seq_length=3, years_to_predict=10):
    """
    Predict stock prices for the next 10 years using the trained model.
    """
    # Get the last seq_length years of data as starting point
    last_sequence = features_scaled[-seq_length:].copy()
    
    # Extrapolate features for future years
    future_features = extrapolate_features(features_scaled, None, years_to_predict)
    
    predictions = []
    current_sequence = last_sequence.copy()
    
    for year in range(years_to_predict):
        # Reshape for prediction
        X_pred = current_sequence.reshape(1, seq_length, current_sequence.shape[1])
        
        # Make prediction
        pred_scaled = model.predict(X_pred, verbose=0)
        predictions.append(pred_scaled[0, 0])
        
        # Update sequence for next prediction
        # Remove the oldest year and add the new features with predicted influence
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1] = future_features[year]
    
    # Convert predictions back to original scale
    predictions_array = np.array(predictions).reshape(-1, 1)
    predictions_unscaled = target_scaler.inverse_transform(predictions_array)
    
    return predictions_unscaled.flatten()

def create_future_dates(last_date, years=10):
    """
    Create future dates for the next 10 years.
    """
    future_dates = []
    for i in range(1, years + 1):
        future_date = last_date + timedelta(days=365 * i)
        future_dates.append(future_date)
    return future_dates

def main():
    try:
        # Fetch all data
        print("Fetching data...")
        df_metrics = fetch_key_metrics(SYMBOL)
        df_prices = fetch_annual_prices(SYMBOL)
        df_cash_flow = fetch_cash_flow(SYMBOL)
        df_income = fetch_income_statement(SYMBOL)
        df_ratios = fetch_ratios(SYMBOL)

        # Merge data on date index (inner join)
        print("Merging datasets...")
        df_merged = df_metrics.join(df_prices[['close']], how='inner', lsuffix='_metrics', rsuffix='_price')
        df_merged = df_merged.join(df_cash_flow, how='inner', lsuffix='', rsuffix='_cashflow')
        df_merged = df_merged.join(df_income, how='inner', lsuffix='', rsuffix='_income')
        df_merged = df_merged.join(df_ratios, how='inner', lsuffix='', rsuffix='_ratios')

        # Define columns to scale (adjust as needed)
        columns_to_scale = ['revenuePerShare', 'close']
        cash_flow_cols = ['operatingCashFlow', 'freeCashFlow']
        income_cols = ['netIncome', 'totalRevenue', 'grossProfit']
        ratio_cols = ['currentRatio', 'debtEquityRatio', 'returnOnEquity']

        # Collect all columns that exist in df_merged
        for col_list in [cash_flow_cols, income_cols, ratio_cols]:
            columns_to_scale.extend([col for col in col_list if col in df_merged.columns])

        columns_to_scale = [col for col in columns_to_scale if col in df_merged.columns]
        print(f"Using columns: {columns_to_scale}")

        # Drop rows with missing values in these columns to ensure clean data
        df_model = df_merged[columns_to_scale].dropna()
        
        if df_model.empty:
            raise ValueError("No data available after removing missing values")
        
        # Sort by date to ensure chronological order
        df_model = df_model.sort_index()
        print(f"Data shape after cleaning: {df_model.shape}")

        # Scale features and target separately
        feature_cols = [col for col in columns_to_scale if col != 'close']
        target_col = 'close'

        if target_col not in df_model.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")

        feature_scaler = MinMaxScaler(feature_range=(0, 1))
        target_scaler = MinMaxScaler(feature_range=(0, 1))

        features_scaled = feature_scaler.fit_transform(df_model[feature_cols])
        # Fix: Ensure target is properly shaped
        target_values = df_model[target_col].values.reshape(-1, 1)
        target_scaled = target_scaler.fit_transform(target_values)
        
        # Flatten target_scaled for sequence creation
        target_scaled_flat = target_scaled.flatten()

        # Create sequences (e.g., use 3 years of data to predict next year's close)
        seq_length = 3
        if len(features_scaled) <= seq_length:
            raise ValueError(f"Not enough data points. Need at least {seq_length + 1} years of data")
            
        X, y = create_multivariate_sequences(features_scaled, target_scaled_flat, seq_length)
        print(f"Created {len(X)} sequences for training")

        if len(X) == 0:
            raise ValueError("No sequences created. Check if you have enough data points")

        # Train-test split (80% train, 20% test)
        train_size = max(1, int(len(X) * 0.8))  # Ensure at least 1 training sample
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        if len(X_train) == 0:
            raise ValueError("No training data available")

        # Build LSTM model
        print("Building LSTM model...")
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        print("Training the model...")
        history = model.fit(X_train, y_train, epochs=100, batch_size=1, 
                          validation_data=(X_test, y_test) if len(X_test) > 0 else None, 
                          verbose=1)

        # Predict and inverse transform (only if we have test data)
        if len(X_test) > 0:
            predictions = model.predict(X_test)
            
            # Ensure correct shapes for inverse transformation
            predictions_reshaped = predictions.reshape(-1, 1)
            y_test_reshaped = y_test.reshape(-1, 1)
            
            predictions_unscaled = target_scaler.inverse_transform(predictions_reshaped)
            y_test_unscaled = target_scaler.inverse_transform(y_test_reshaped)

            # Evaluate model performance
            rmse = np.sqrt(mean_squared_error(y_test_unscaled, predictions_unscaled))
            print(f"Test RMSE: {rmse:.4f}")
        else:
            print("No test data available for evaluation")
            predictions_unscaled = np.array([])
            y_test_unscaled = np.array([])

        # NEW: Make 10-year predictions
        print("\nMaking 10-year predictions...")
        future_predictions = predict_future_prices(model, features_scaled, target_scaler, seq_length, 10)
        
        # Create future dates
        last_date = df_model.index[-1]
        future_dates = create_future_dates(last_date, 10)
        
        # Print predictions
        print(f"\n10-Year Stock Price Predictions for {SYMBOL}:")
        print("=" * 50)
        for i, (date, price) in enumerate(zip(future_dates, future_predictions)):
            print(f"Year {i+1} ({date.year}): ${price:.2f}")
        
        # Calculate potential returns - FIX: Extract scalar value from Series
        current_price = float(df_model[target_col].iloc[-1])  # Convert to float
        final_price = float(future_predictions[-1])  # Ensure it's a float
        total_return = ((final_price - current_price) / current_price) * 100
        annual_return = ((final_price / current_price) ** (1/10) - 1) * 100
        
        print(f"\nCurrent Price: ${current_price:.2f}")
        print(f"Predicted Price in 10 years: ${final_price:.2f}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Annualized Return: {annual_return:.2f}%")

        # Plot results
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Historical actual vs predicted (only if we have test data)
        if len(predictions_unscaled) > 0:
            plt.subplot(2, 1, 1)
            plt.plot(y_test_unscaled.flatten(), label='Actual Close Price', marker='o')
            plt.plot(predictions_unscaled.flatten(), label='Predicted Close Price', marker='s')
            plt.title('Historical Stock Price Prediction (Test Set)')
            plt.xlabel('Time Period')
            plt.ylabel('Price ($)')
            plt.legend()
            plt.grid(True)

            # Plot 2: Future predictions
            plt.subplot(2, 1, 2)
        else:
            # If no test data, use single plot
            plt.subplot(1, 1, 1)
        
        # Historical data (last 5 years for context, or all available data if less)
        n_historical = min(5, len(df_model))
        historical_dates = df_model.index[-n_historical:]
        historical_prices = df_model[target_col].iloc[-n_historical:].values
        
        # Combine historical and future data for plotting
        all_dates = list(historical_dates) + future_dates
        all_prices = list(historical_prices) + list(future_predictions)
        
        plt.plot(historical_dates, historical_prices, 'b-o', label='Historical Prices', linewidth=2)
        plt.plot(future_dates, future_predictions, 'r--s', label='10-Year Predictions', linewidth=2)
        plt.axvline(x=last_date, color='gray', linestyle=':', alpha=0.7, label='Prediction Start')
        
        plt.title(f'10-Year Stock Price Prediction for {SYMBOL}')
        plt.xlabel('Year')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # Additional analysis: Confidence intervals (simple approach)
        print(f"\nPrediction Analysis:")
        print("=" * 30)
        
        # Calculate volatility from price changes
        price_changes = df_model[target_col].pct_change().dropna()
        if len(price_changes) > 0:
            volatility = np.std(price_changes)
            print(f"Historical Annual Volatility: {volatility*100:.2f}%")
            
            # Simple confidence intervals (this is a basic approach)
            confidence_95 = 1.96 * volatility * np.sqrt(np.arange(1, 11))
            
            print(f"\n95% Confidence Intervals:")
            for i, (year, pred, conf) in enumerate(zip(range(1, 11), future_predictions, confidence_95)):
                lower_bound = pred * (1 - conf)
                upper_bound = pred * (1 + conf)
                print(f"Year {year}: ${max(0, lower_bound):.2f} - ${upper_bound:.2f} (Prediction: ${pred:.2f})")
        else:
            print("Cannot calculate volatility - insufficient price change data")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please check your data and API configuration")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

# this file is just a test file for running the ai so it doesnt ruin the original code... its a mess.
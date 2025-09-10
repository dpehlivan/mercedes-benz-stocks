from flask import Flask, render_template
import requests
from dotenv import load_dotenv
import os
from config import COMDAILY_API_URL, COMDAILY_AUTH, SYMBOL



#this file is the original edited project python file
#its here just in case i crash the whole code

# import pandas as pd 
# from sklearn.preprocessing import MinMaxScaler
# import numpy as np

load_dotenv()

app = Flask(__name__)

def fetch_data(endpoint):
    url = f"{COMDAILY_API_URL}/{endpoint}/{SYMBOL}"
    print(f"Attempting to fetch from: {url}")  # Debug print
    try:
        res = requests.get(url, auth=COMDAILY_AUTH)
        print(f"Response status: {res.status_code}")  # Debug print
        res.raise_for_status()  # Raises an error for bad responses
        
        # Return the correct data structure based on the endpoint
        if endpoint == "income-statement":
            return res.json().get("message", [])  # Extract the list from the message key
        elif endpoint == "key-metrics":
            return res.json().get("message", [])  # Extract the list from the message key
        return res.json()  # For other endpoints, return the full response

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as e:
        print(f"Other error occurred: {e}")
    return []  # Return an empty list on failure




#incorrect extraction of data to train AI model
#     def download_stock_data(endpoint):
#     url =  f"{COMDAILY_API_URL}/{endpoint}/{SYMBOL}"
#     response = requests.get(url)
#     data = response.json()
#     stock_data = pd.DataFrame(data['historical'])
#     stock_data = stock_data.sort_values('date')  # Ensure chronological order
#     return stock_data 






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
    income = fetch_data("income-statement")
    metrics = fetch_data("key-metrics")
    trend_comparison = compare_historical_to_latest(income) if income else None
    return render_template("index_og.html", income=income, metrics=metrics, trend_comparison=trend_comparison)

if __name__ == "__main__":
    app.run(debug=True)

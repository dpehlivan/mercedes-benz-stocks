import requests
from config import COMDAILY_API_URL, COMDAILY_AUTH, SYMBOL

def fetch_and_print_raw_json(endpoint, symbol):
    url = f"{COMDAILY_API_URL}/{endpoint}/{symbol}"
    print(f"Fetching URL: {url}")
    try:
        response = requests.get(url, auth=COMDAILY_AUTH)
        print(f"Response status code: {response.status_code}")
        response.raise_for_status()
        
        # Print raw JSON text
        print("Raw JSON text:")
        print(response.text)  # raw text of the response
        
        # Parse JSON
        json_data = response.json()
        print("Parsed JSON object:")
        print(json_data)
        
        return json_data
    
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
    except Exception as ex:
        print(f"Unexpected error: {ex}")

# Usage
endpoint = "key-metrics"
json_data = fetch_and_print_raw_json(endpoint, SYMBOL)


#i forgot which endpoint was key metrics at ... found it!
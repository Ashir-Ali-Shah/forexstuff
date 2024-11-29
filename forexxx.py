import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests

# App title
st.markdown("# 📈 Forex Trade Signal Generator")

# Currency Pair Selection
currency_pairs = {
    "EUR/USD": ("EUR", "USD"),
    "GBP/USD": ("GBP", "USD"),
    "USD/JPY": ("USD", "JPY"),
    "AUD/USD": ("AUD", "USD"),
    "NZD/USD": ("NZD", "USD"),
    "USD/CAD": ("USD", "CAD"),
    "USD/CHF": ("USD", "CHF"),
}

# Sidebar layout
st.sidebar.markdown("## ⚙️ Settings")
selected_pair = st.sidebar.selectbox("🌍 Select Currency Pair", options=list(currency_pairs.keys()))
selected_indicator = st.sidebar.selectbox(
    "📊 Select Indicator",
    options=["Live Exchange Rate", "Bitcoin Conversion"]
)
account_balance = st.sidebar.number_input("💰 Account Balance (USD)", min_value=0.0, value=1000.0, step=100.0)
risk_percentage = st.sidebar.slider("📉 Risk Percentage (%)", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
risk_reward_ratio = st.sidebar.slider("🎯 Risk/Reward Ratio", min_value=1.0, max_value=5.0, value=2.0, step=0.1)

# Alpha Vantage API key provided
ALPHA_VANTAGE_API_KEY = "GQN1R1CGXAO32GI3"  # Your provided API key

# Fetch live exchange rate using Alpha Vantage API
def fetch_live_rate(pair):
    try:
        base, quote = currency_pairs[pair]
        url = f"https://www.alphavantage.co/query"
        params = {
            "function": "FX_INTRADAY",
            "from_symbol": base,
            "to_symbol": quote,
            "interval": "15min",
            "apikey": ALPHA_VANTAGE_API_KEY
        }
        
        # Request data from Alpha Vantage
        response = requests.get(url, params=params)
        response.raise_for_status()  # This will raise an error for bad HTTP responses
        
        # Parse the response
        data = response.json()
        
        # Debugging: Print the raw response data to check the structure
        st.write(data)
        
        # Check if the response contains the 'Time Series FX (15min)' data
        if "Time Series FX (15min)" not in data:
            raise ValueError(f"No data found for {base}/{quote}. The API response is: {data}")
        
        # Get the most recent exchange rate
        last_time = list(data["Time Series FX (15min)"].keys())[0]
        exchange_rate = data["Time Series FX (15min)"][last_time]["4. close"]
        return float(exchange_rate)
    
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching live exchange rate: {e}")
        return None
    except ValueError as e:
        st.error(f"Value error: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None

# Fetch Bitcoin conversion rate (for demo)
def fetch_btc_rate(currency):
    try:
        url = f"https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies={currency.lower()}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if currency.lower() in data["bitcoin"]:
            return data["bitcoin"][currency.lower()]
        else:
            raise ValueError(f"Bitcoin rate for {currency} not found.")
    except Exception as e:
        st.error(f"Error fetching Bitcoin rate: {e}")
        return None

# Generate signal based on the selected indicator
def generate_signal(pair, indicator):
    try:
        if indicator == "Live Exchange Rate":
            rate = fetch_live_rate(pair)
            if rate is not None:
                return f"1 {currency_pairs[pair][0]} = {rate:.4f} {currency_pairs[pair][1]}", rate
        elif indicator == "Bitcoin Conversion":
            btc_rate = fetch_btc_rate(currency_pairs[pair][1])
            if btc_rate is not None:
                return f"1 BTC = {btc_rate:.2f} {currency_pairs[pair][1]}", btc_rate
        return "No Signal", None
    except Exception as e:
        st.error(f"Error generating signal: {e}")
        return "No Signal", None

# Plot the signal or conversion data
def plot_signal(rate, indicator, pair):
    try:
        if rate is None:
            raise ValueError("Rate data is missing for charting.")
        base, quote = currency_pairs[pair]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar([indicator], [rate], color="blue")
        ax.set_title(f"{indicator} for {pair}")
        ax.set_ylabel(f"Rate in {quote}")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error generating chart: {e}")

# Main Execution
signal, rate = generate_signal(selected_pair, selected_indicator)

if rate is not None:
    st.metric("Signal", signal)
    plot_signal(rate, selected_indicator, selected_pair)
else:
    st.error("Failed to generate a valid rate or signal.")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from forex_python.converter import CurrencyRates
import requests
import time

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

# Initialize forex-python converters
currency_converter = CurrencyRates()

# Fetch live exchange rate with retries and error handling
def fetch_live_rate(pair):
    try:
        base, quote = currency_pairs[pair]
        rate = currency_converter.get_rate(base, quote)
        if rate is None:
            raise ValueError("API returned no rate for the given currency pair.")
        return rate
    except Exception as e:
        st.error(f"Error fetching live exchange rate with forex-python: {e}")
        # Fallback to alternative API if forex-python fails
        return fetch_live_rate_alternative(pair)

# Alternative: Use ExchangeRate-API (Backup Plan)
def fetch_live_rate_alternative(pair):
    try:
        base, quote = currency_pairs[pair]
        api_key = "YOUR_EXCHANGERATE_API_KEY"  # Replace with your ExchangeRate-API key
        url = f"https://v6.exchangerate-api.com/v6/{api_key}/latest/{base}"
        response = requests.get(url)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        data = response.json()
        if "conversion_rates" in data and quote in data["conversion_rates"]:
            return data["conversion_rates"][quote]
        else:
            raise ValueError(f"Rate for {base} to {quote} not found in API response.")
    except Exception as e:
        st.error(f"Error fetching live exchange rate from ExchangeRate-API: {e}")
        return None

# Fetch Bitcoin conversion rate
def fetch_btc_rate(currency):
    try:
        btc_rate = btc_converter.get_latest_price(currency)
        return btc_rate
    except Exception as e:
        st.error(f"Error fetching Bitcoin rate: {e}")
        return None

# Display live rate or Bitcoin conversion
def generate_signal(pair, indicator):
    try:
        base, quote = currency_pairs[pair]
        if indicator == "Live Exchange Rate":
            rate = fetch_live_rate(pair)
            if rate is not None:
                return f"1 {base} = {rate:.4f} {quote}", rate
        elif indicator == "Bitcoin Conversion":
            btc_rate = fetch_btc_rate(quote)
            if btc_rate is not None:
                return f"1 BTC = {btc_rate:.2f} {quote}", btc_rate
        return "No Signal", None
    except Exception as e:
        st.error(f"Error generating signal: {e}")
        return "No Signal", None

# Plot the live rate or BTC conversion
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

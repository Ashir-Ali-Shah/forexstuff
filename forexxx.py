import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from forex_python.converter import CurrencyRates
import yfinance as yf

# Set page config
st.set_page_config(page_title="Forex Trade Signal Generator", page_icon="\ud83d\udcc8", layout="wide", initial_sidebar_state="expanded")

# App title
st.markdown("# \ud83d\udcc8 Forex Trade Signal Generator")

# Currency Pair Selection
currency_pairs = {
    "XAUUSD": "GLD",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X",
    "NZDUSD": "NZDUSD=X",
    "USDCAD": "USDCAD=X",
    "USDCHF": "USDCHF=X",
}

# Sidebar
st.sidebar.markdown("## ⚙️ Settings")
selected_pair = st.sidebar.selectbox("🌍 Select Currency Pair", options=list(currency_pairs.keys()))
account_balance = st.sidebar.number_input("💰 Account Balance (USD)", min_value=0.0, value=1000.0, step=100.0)
risk_percentage = st.sidebar.slider("📉 Risk Percentage (%)", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
risk_reward_ratio = st.sidebar.slider("🎯 Risk/Reward Ratio", min_value=1.0, max_value=5.0, value=2.0, step=0.1)

# Forex Statistics
st.markdown("## 📊 Forex Statistics")
currency_rates = CurrencyRates()
try:
    base_currency = selected_pair[:3]
    quote_currency = selected_pair[3:]
    forex_rate = currency_rates.get_rate(base_currency, quote_currency)
    st.write(f"💱 **Exchange Rate ({base_currency}/{quote_currency})**: {forex_rate}")
except Exception as e:
    st.error(f"Error fetching Forex rate: {e}")

# Fetch forex data from Yahoo Finance
def fetch_forex_data(pair):
    try:
        data = yf.download(tickers=pair, period="30d", interval="1h", progress=False)
        if data.empty:
            raise ValueError(f"No data fetched for {pair}")
        data.reset_index(inplace=True)
        data.rename(columns={"Datetime": "datetime", "Open": "open", "High": "high", "Low": "low", "Close": "close"}, inplace=True)
        data.set_index("datetime", inplace=True)
        return data
    except Exception as e:
        st.error(f"Error fetching forex data: {e}")
        return pd.DataFrame()

# Calculate signal strength
def calculate_signal_strength(data):
    try:
        if data.empty or len(data["close"]) < 10:
            return "Unknown"
        recent_closes = data["close"].tail(10)
        volatility = np.std(recent_closes)
        if volatility < 0.5:
            return "Strong"
        elif volatility < 1.0:
            return "Moderate"
        else:
            return "Weak"
    except Exception as e:
        st.error(f"Error calculating signal strength: {e}")
        return "Unknown"

# Lot Size Calculation
def calculate_lot_size(balance, risk_percent, entry_price, stop_loss):
    try:
        risk_amount = balance * (risk_percent / 100)
        pip_risk = abs(entry_price - stop_loss)
        if pip_risk == 0:
            return 0
        lot_size = risk_amount / pip_risk
        return round(lot_size, 2)
    except Exception as e:
        st.error(f"Error calculating lot size: {e}")
        return 0

# Main Execution
ticker_symbol = currency_pairs[selected_pair]
data = fetch_forex_data(ticker_symbol)

if not data.empty:
    try:
        entry_price = data["close"].iloc[-1]
        stop_loss = entry_price - 2
        take_profit = entry_price + (2 * risk_reward_ratio)
        trade_type = "Buy" if take_profit > entry_price else "Sell"
        signal_strength = calculate_signal_strength(data)
        lot_size = calculate_lot_size(account_balance, risk_percentage, entry_price, stop_loss)

        st.markdown("### 📊 Trade Signal")
        st.write(f"**Trade Type**: {trade_type}")
        st.write(f"**Signal Strength**: {signal_strength}")
        st.write(f"**Entry Price**: {entry_price}")
        st.write(f"**Stop Loss**: {stop_loss}")
        st.write(f"**Take Profit**: {take_profit}")
        st.write(f"**Lot Size**: {lot_size}")

    except Exception as e:
        st.error(f"An error occurred: {e}")

if 'trade_history' in st.session_state and not st.session_state.trade_history.empty:
    st.markdown("### 📜 Trade History")
    st.dataframe(st.session_state.trade_history)

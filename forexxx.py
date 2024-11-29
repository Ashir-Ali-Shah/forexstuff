import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import ccxt
from matplotlib.dates import DateFormatter, AutoDateLocator

# App title
st.markdown("# 📈 Forex Trade Signal Generator")

# Currency Pair Selection
currency_pairs = {
    "EUR/USD": "EUR/USDT",
    "GBP/USD": "GBP/USDT",
    "USD/JPY": "USD/JPY",
    "AUD/USD": "AUD/USDT",
    "NZD/USD": "NZD/USDT",
    "USD/CAD": "USD/CAD",
    "USD/CHF": "USD/CHF",
}

# Sidebar layout
st.sidebar.markdown("## ⚙️ Settings")
selected_pair = st.sidebar.selectbox("🌍 Select Currency Pair", options=list(currency_pairs.keys()))
selected_indicator = st.sidebar.selectbox(
    "📊 Select Indicator",
    options=["SMA (10/50)", "EMA (10/50)", "RSI (14)"]
)
account_balance = st.sidebar.number_input("💰 Account Balance (USD)", min_value=0.0, value=1000.0, step=100.0)
risk_percentage = st.sidebar.slider("📉 Risk Percentage (%)", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
risk_reward_ratio = st.sidebar.slider("🎯 Risk/Reward Ratio", min_value=1.0, max_value=5.0, value=2.0, step=0.1)
chart_type = st.sidebar.radio("📈 Chart Type", options=["Candlestick", "Line Chart"])

# Fetch real-time forex data using ccxt
@st.cache_data
def fetch_forex_data(pair):
    try:
        exchange = ccxt.binance()
        market = currency_pairs[pair].replace("/", "")
        data = exchange.fetch_ohlcv(market, timeframe="15m", limit=130)
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("datetime", inplace=True)
        df.drop(columns=["timestamp"], inplace=True)
        return df
    except Exception as e:
        st.error(f"Error fetching forex data: {e}")
        return pd.DataFrame()

# Simple Moving Average
def calculate_sma(data, window):
    return data["close"].rolling(window=window).mean()

# Exponential Moving Average
def calculate_ema(data, window):
    return data["close"].ewm(span=window, adjust=False).mean()

# Relative Strength Index
def calculate_rsi(data, window):
    delta = data["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to calculate lot size
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

# Generate Trade Signal
def generate_signal(data, indicator):
    try:
        if data.empty:
            raise ValueError("Data is empty for signal generation.")

        if indicator == "SMA (10/50)":
            data["SMA_10"] = calculate_sma(data, 10)
            data["SMA_50"] = calculate_sma(data, 50)
            signal_condition = data["SMA_10"].iloc[-1] > data["SMA_50"].iloc[-1]
        elif indicator == "EMA (10/50)":
            data["EMA_10"] = calculate_ema(data, 10)
            data["EMA_50"] = calculate_ema(data, 50)
            signal_condition = data["EMA_10"].iloc[-1] > data["EMA_50"].iloc[-1]
        elif indicator == "RSI (14)":
            data["RSI"] = calculate_rsi(data, 14)
            signal_condition = data["RSI"].iloc[-1] < 30  # Buy signal when RSI is oversold

        last_close = data["close"].iloc[-1]
        if signal_condition:
            return "Buy", last_close
        else:
            return "Sell", last_close
    except Exception as e:
        st.error(f"Error generating trade signal: {e}")
        return "No Signal", 0.0

# Plotting function
def plot_chart(data, signal, entry_price, stop_loss, take_profit, chart_type):
    try:
        if data.empty:
            raise ValueError("Data is empty for charting.")

        if chart_type == "Candlestick":
            fig, ax = plt.subplots(figsize=(12, 8))
            mpf.plot(data, type="candle", style="charles", ax=ax, mav=(10, 50), volume=False)
            ax.axhline(entry_price, color="orange", linestyle="--", label=f"Entry: {signal}")
            ax.axhline(stop_loss, color="red", linestyle="--", label="Stop Loss")
            ax.axhline(take_profit, color="green", linestyle="--", label="Take Profit")
        else:
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.plot(data.index, data["close"], label="Close Price", color="blue")
            ax.axhline(entry_price, color="orange", linestyle="--", label="Entry Price")
            ax.axhline(stop_loss, color="red", linestyle="--", label="Stop Loss")
            ax.axhline(take_profit, color="green", linestyle="--", label="Take Profit")

        ax.set_title(f"{selected_pair} Trade Signal")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid()
        ax.xaxis.set_major_locator(AutoDateLocator())
        ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d %H:%M"))
        fig.autofmt_xdate()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error generating chart: {e}")

# Main Execution
data = fetch_forex_data(selected_pair)

if not data.empty:
    try:
        signal, entry_price = generate_signal(data, selected_indicator)
        if entry_price == 0 or pd.isna(entry_price):
            st.error("Failed to generate a valid entry price.")
        else:
            stop_loss = float(entry_price * 0.995)
            take_profit = float(entry_price + (entry_price - stop_loss) * risk_reward_ratio)
            lot_size = calculate_lot_size(account_balance, risk_percentage, entry_price, stop_loss)

            col1, col2, col3 = st.columns(3)
            col1.metric("Signal", signal)
            col2.metric("Entry Price", f"${entry_price:.2f}")
            col3.metric("Lot Size", f"{lot_size:.2f}")

            st.markdown("### 📊 Trade Details")
            st.write(f"**Stop Loss:** ${stop_loss:.2f}")
            st.write(f"**Take Profit:** ${take_profit:.2f}")
            st.write(f"**Risk Percentage:** {risk_percentage}%")
            st.write(f"**Risk/Reward Ratio:** {risk_reward_ratio}")

            st.markdown("---")
            st.markdown("### 📈 Trade Chart")
            plot_chart(data, signal, entry_price, stop_loss, take_profit, chart_type)
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.error("Failed to fetch data for the selected pair.")

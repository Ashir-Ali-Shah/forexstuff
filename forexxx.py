import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from ta.trend import SMAIndicator
from matplotlib.dates import DateFormatter, AutoDateLocator

# Currency Pair Selection
currency_pairs = {
    "XAUUSD": "Gold",
    "EURUSD": "Euro/USD",
    "GBPUSD": "Pound/USD",
    "USDJPY": "USD/Yen",
    "AUDUSD": "AUD/USD",
    "NZDUSD": "NZD/USD",
    "USDCAD": "USD/CAD",
    "USDCHF": "USD/CHF",
}

# Streamlit Sidebar
st.sidebar.title("Trade Signal Generator")
selected_pair = st.sidebar.selectbox("Select Currency Pair", options=list(currency_pairs.keys()))
account_balance = st.sidebar.number_input("Account Balance (USD)", min_value=0.0, value=1000.0, step=100.0)
risk_percentage = st.sidebar.slider("Risk Percentage (%)", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
risk_reward_ratio = st.sidebar.slider("Risk/Reward Ratio", min_value=1.0, max_value=5.0, value=2.0, step=0.1)

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

# Generate mock data for demonstration purposes
@st.cache_data
def fetch_data():
    np.random.seed(42)
    date_range = pd.date_range(end=pd.Timestamp.now(), periods=100, freq="15T")
    prices = np.cumsum(np.random.randn(len(date_range)) * 0.1 + 0.5) + 100
    data = pd.DataFrame({"Datetime": date_range, "Close": prices})
    data["Open"] = data["Close"] + np.random.randn(len(data)) * 0.1
    data["High"] = data[["Close", "Open"]].max(axis=1) + np.random.rand(len(data)) * 0.2
    data["Low"] = data[["Close", "Open"]].min(axis=1) - np.random.rand(len(data)) * 0.2
    data.set_index("Datetime", inplace=True)
    return data

# Generate Trade Signal
def generate_signal(data):
    try:
        short_window = 10
        long_window = 50

        data["SMA10"] = SMAIndicator(data["Close"], window=short_window).sma_indicator()
        data["SMA50"] = SMAIndicator(data["Close"], window=long_window).sma_indicator()

        if data.empty or pd.isna(data["Close"].iloc[-1]):
            raise ValueError("Data is invalid or insufficient for generating signals.")

        last_close = data["Close"].iloc[-1]
        if pd.isna(last_close) or not isinstance(last_close, (int, float)):
            raise ValueError("Invalid close price.")

        if pd.isna(data["SMA10"].iloc[-1]) or pd.isna(data["SMA50"].iloc[-1]):
            raise ValueError("Moving averages could not be calculated. Not enough data.")

        if data["SMA10"].iloc[-1] > data["SMA50"].iloc[-1]:
            return "Buy", last_close
        else:
            return "Sell", last_close
    except Exception as e:
        st.error(f"Error generating trade signal: {e}")
        return "No Signal", 0.0

# Plotting function using Matplotlib
def plot_chart(data, signal, entry_price, stop_loss, take_profit):
    try:
        fig, ax = plt.subplots(figsize=(12, 8))
        mpf.plot(data, type="candle", style="charles", ax=ax, mav=(10, 50), volume=False)
        ax.plot(data.index, data["SMA10"], label="SMA10", color="blue")
        ax.plot(data.index, data["SMA50"], label="SMA50", color="orange")
        ax.scatter(data.index[-1], entry_price, color="green", label=f"Entry: {signal}", zorder=5)
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
st.title("Forex Trade Signal Generator")

st.write(f"Fetching data for {selected_pair}...")
data = fetch_data()

if not data.empty:
    try:
        signal, entry_price = generate_signal(data)
        st.write(f"Debug - Entry Price: {entry_price}")
        
        if entry_price == 0 or pd.isna(entry_price):
            st.error("Failed to generate a valid entry price.")
        else:
            stop_loss = float(entry_price * 0.995)
            take_profit = float(entry_price + (entry_price - stop_loss) * risk_reward_ratio)
            lot_size = calculate_lot_size(account_balance, risk_percentage, entry_price, stop_loss)

            st.write(f"### Trade Signal for {selected_pair}")
            st.write(f"- Signal: **{signal}**")
            
            if isinstance(entry_price, (int, float)) and not pd.isna(entry_price):
                st.write(f"- Entry Price: **{entry_price:.2f}**")
            else:
                st.error("Invalid entry price. Unable to format entry price.")
            
            st.write(f"- Stop Loss: **{stop_loss:.2f}**")
            st.write(f"- Take Profit: **{take_profit:.2f}**")
            st.write(f"- Lot Size: **{lot_size:.2f}**")

            plot_chart(data, signal, entry_price, stop_loss, take_profit)
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.error("Failed to fetch data for the selected pair.")

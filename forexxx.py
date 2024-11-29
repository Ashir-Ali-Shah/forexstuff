import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator
from matplotlib.dates import DateFormatter, AutoDateLocator
import yfinance as yf

# App title
st.markdown("# 📈 Forex Trade Signal Generator")

# Currency Pair Selection
currency_pairs = {
    "XAUUSD": "GLD",  # Using SPDR Gold Shares ETF as an alternative
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X",
    "NZDUSD": "NZDUSD=X",
    "USDCAD": "USDCAD=X",
    "USDCHF": "USDCHF=X",
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

# Fetch real-time forex data using yfinance
@st.cache_data
def fetch_forex_data(pair):
    try:
        st.write(f"Fetching data for {pair}...")
        data = yf.download(tickers=pair, period="5d", interval="15m", progress=False)
        if data.empty:
            raise ValueError(f"No data fetched for the selected pair: {pair}")
        data.reset_index(inplace=True)
        if "Close" not in data.columns:
            raise ValueError(f"'Close' column is missing in the data for {pair}")
        data.rename(columns={"Datetime": "Datetime", "Open": "Open", "High": "High", "Low": "Low", "Close": "Close"}, inplace=True)
        data.set_index("Datetime", inplace=True)
        return data
    except Exception as e:
        st.error(f"Error fetching forex data: {e}")
        return pd.DataFrame()

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
        if data.empty or "Close" not in data.columns:
            raise ValueError("Data is empty or invalid for signal generation.")

        # Ensure the 'Close' column is a pandas Series
        close_series = data["Close"]

        if indicator == "SMA (10/50)":
            sma10 = SMAIndicator(close_series, window=10).sma_indicator()
            sma50 = SMAIndicator(close_series, window=50).sma_indicator()
            data["Indicator1"] = pd.Series(sma10.squeeze(), index=data.index)  # Flatten to 1D
            data["Indicator2"] = pd.Series(sma50.squeeze(), index=data.index)  # Flatten to 1D
            signal_condition = data["Indicator1"].iloc[-1] > data["Indicator2"].iloc[-1]
        elif indicator == "EMA (10/50)":
            ema10 = EMAIndicator(close_series, window=10).ema_indicator()
            ema50 = EMAIndicator(close_series, window=50).ema_indicator()
            data["Indicator1"] = pd.Series(ema10.squeeze(), index=data.index)  # Flatten to 1D
            data["Indicator2"] = pd.Series(ema50.squeeze(), index=data.index)  # Flatten to 1D
            signal_condition = data["Indicator1"].iloc[-1] > data["Indicator2"].iloc[-1]
        elif indicator == "RSI (14)":
            rsi = RSIIndicator(close_series, window=14).rsi()
            data["RSI"] = pd.Series(rsi.squeeze(), index=data.index)  # Flatten to 1D
            signal_condition = data["RSI"].iloc[-1] < 30  # Buy signal when RSI is oversold

        last_close = data["Close"].iloc[-1]
        if pd.isna(last_close):
            raise ValueError("Last close price is NaN.")
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
            ax.plot(data.index, data["Indicator1"], label="Indicator 1", color="blue")
            ax.plot(data.index, data["Indicator2"], label="Indicator 2", color="orange")
            ax.scatter(data.index[-1], entry_price, color="green", label=f"Entry: {signal}", zorder=5)
            ax.axhline(stop_loss, color="red", linestyle="--", label="Stop Loss")
            ax.axhline(take_profit, color="green", linestyle="--", label="Take Profit")
        else:
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.plot(data.index, data["Close"], label="Close Price", color="blue")
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
ticker_symbol = currency_pairs[selected_pair]
data = fetch_forex_data(ticker_symbol)

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

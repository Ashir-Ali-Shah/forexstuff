import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, AutoDateLocator
import mplfinance as mpf

# Sidebar for User Inputs
st.sidebar.title("Trade Signal Configurations")

# Currency Pair Selection
currency_pairs = {
    "XAUUSD": "GC=F",  # Gold futures
    "EURUSD": "EURUSD=X",  # Euro to USD Forex
    "GBPUSD": "GBPUSD=X",  # British Pound to USD Forex
    "USDJPY": "USDJPY=X",  # USD to Japanese Yen Forex
}
selected_pair = st.sidebar.selectbox("Select Currency Pair", list(currency_pairs.keys()))
ticker_symbol = currency_pairs[selected_pair]

# Account Balance and Lot Size Inputs
account_balance = st.sidebar.number_input("Account Balance (USD)", min_value=0.0, value=1000.0, step=100.0)
risk_percentage = st.sidebar.slider("Risk Percentage (%)", min_value=0, max_value=10, value=2)
risk_reward_ratio = st.sidebar.slider("Risk/Reward Ratio", min_value=1, max_value=5, value=2)

# Function to calculate lot size
def calculate_lot_size(balance, risk_percent, entry_price, stop_loss):
    risk_amount = balance * (risk_percent / 100)
    pip_risk = abs(entry_price - stop_loss)
    if pip_risk == 0:  # Avoid division by zero
        return 0
    lot_size = risk_amount / pip_risk
    return round(lot_size, 2)

# Fetch historical data using yfinance
def fetch_data(symbol, period="5d", interval="15m"):
    data = yf.download(tickers=symbol, period=period, interval=interval)
    data.reset_index(inplace=True)
    return data

# Generate Trade Signal
def generate_signal(data):
    # Example trading logic: Moving Average Crossover
    short_window = 10  # Short-term moving average
    long_window = 50  # Long-term moving average
    data["SMA10"] = data["Close"].rolling(window=short_window).mean()
    data["SMA50"] = data["Close"].rolling(window=long_window).mean()
    
    if data["SMA10"].iloc[-1] > data["SMA50"].iloc[-1]:  # Buy signal
        return "Buy", data["Close"].iloc[-1]
    else:  # Sell signal
        return "Sell", data["Close"].iloc[-1]

# Backtesting strategy performance
def backtest_strategy(data):
    data["Signal"] = data["SMA10"] > data["SMA50"]  # Buy when SMA10 > SMA50
    data["Daily Return"] = data["Close"].pct_change()
    data["Strategy Return"] = data["Signal"].shift(1) * data["Daily Return"]
    cumulative_strategy_return = (1 + data["Strategy Return"]).cumprod()
    return cumulative_strategy_return, data

# Plotting function using Matplotlib
def plot_chart(data, signal, entry_price):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Candlestick chart
    mpf.plot(data.set_index("Datetime"), type="candle", style="charles", ax=ax, mav=(10, 50), volume=False)

    # Highlight entry point
    ax.plot(data["Datetime"], data["SMA10"], label="SMA10", color="blue")
    ax.plot(data["Datetime"], data["SMA50"], label="SMA50", color="orange")
    ax.scatter(data["Datetime"].iloc[-1], entry_price, color="red", label=f"Entry: {signal}", zorder=5)

    # Annotations and formatting
    ax.set_title(f"{selected_pair} Trade Signal")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid()
    ax.xaxis.set_major_locator(AutoDateLocator())
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d %H:%M"))
    fig.autofmt_xdate()
    st.pyplot(fig)

# Main Interface
st.title("Trade Signal Generator with yfinance")

# Fetch data and generate signal
data = fetch_data(ticker_symbol, period="5d", interval="15m")

if not data.empty:
    signal, entry_price = generate_signal(data)
    stop_loss = entry_price * 0.995  # Example: Stop loss 0.5% below entry
    take_profit = entry_price + (entry_price - stop_loss) * risk_reward_ratio
    lot_size = calculate_lot_size(account_balance, risk_percentage, entry_price, stop_loss)

    # Display Signal
    st.write(f"**{selected_pair} - {signal} Signal**")
    st.write(f"Entry Price: {entry_price}")
    st.write(f"Stop Loss: {stop_loss}")
    st.write(f"Take Profit: {take_profit}")
    st.write(f"Lot Size: {lot_size}")

    # Backtesting
    cumulative_strategy_return, backtest_data = backtest_strategy(data)
    st.write("### Backtest Performance")
    st.line_chart(cumulative_strategy_return)

    # Plot the chart
    plot_chart(data, signal, entry_price)
else:
    st.write("Failed to fetch data for the selected pair.")

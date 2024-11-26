import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from matplotlib.dates import DateFormatter, AutoDateLocator

# Currency Pair Selection
currency_pairs = {
    "XAUUSD": "GC=F",  # Gold futures
    "EURUSD": "EURUSD=X",  # Euro to USD Forex
    "GBPUSD": "GBPUSD=X",  # British Pound to USD Forex
    "USDJPY": "USDJPY=X",  # USD to Japanese Yen Forex
    "AUDUSD": "AUDUSD=X",  # Australian Dollar to USD Forex
    "NZDUSD": "NZDUSD=X",  # New Zealand Dollar to USD Forex
    "USDCAD": "USDCAD=X",  # USD to Canadian Dollar Forex
    "USDCHF": "USDCHF=X",  # USD to Swiss Franc Forex
}

# Streamlit Sidebar
st.sidebar.title("Trade Signal Generator")
selected_pair = st.sidebar.selectbox("Select Currency Pair", options=list(currency_pairs.keys()))
account_balance = st.sidebar.number_input("Account Balance (USD)", min_value=0.0, value=1000.0, step=100.0)
risk_percentage = st.sidebar.slider("Risk Percentage (%)", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
risk_reward_ratio = st.sidebar.slider("Risk/Reward Ratio", min_value=1.0, max_value=5.0, value=2.0, step=0.1)

ticker_symbol = currency_pairs[selected_pair]

# Function to calculate lot size
def calculate_lot_size(balance, risk_percent, entry_price, stop_loss):
    entry_price = float(entry_price)
    stop_loss = float(stop_loss)
    risk_amount = balance * (risk_percent / 100)
    pip_risk = abs(entry_price - stop_loss)
    if pip_risk == 0:
        return 0
    lot_size = risk_amount / pip_risk
    return round(lot_size, 2)

# Fetch historical data using yfinance
@st.cache_data
def fetch_data(symbol, period="5d", interval="15m"):
    try:
        data = yf.download(tickers=symbol, period=period, interval=interval, progress=False)
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# Generate Trade Signal
def generate_signal(data):
    short_window = 10
    long_window = 50
    data["SMA10"] = data["Close"].rolling(window=short_window).mean()
    data["SMA50"] = data["Close"].rolling(window=long_window).mean()

    if data["SMA10"].iloc[-1] > data["SMA50"].iloc[-1]:
        return "Buy", data["Close"].iloc[-1]
    else:
        return "Sell", data["Close"].iloc[-1]

# Backtesting strategy performance
def backtest_strategy(data):
    data["Signal"] = data["SMA10"] > data["SMA50"]
    data["Daily Return"] = data["Close"].pct_change()
    data["Strategy Return"] = data["Signal"].shift(1) * data["Daily Return"]
    cumulative_strategy_return = (1 + data["Strategy Return"]).cumprod()
    return cumulative_strategy_return, data

# Plotting function using Matplotlib
def plot_chart(data, signal, entry_price, stop_loss, take_profit):
    fig, ax = plt.subplots(figsize=(12, 8))
    mpf.plot(data.set_index("Datetime"), type="candle", style="charles", ax=ax, mav=(10, 50), volume=False)
    ax.plot(data["Datetime"], data["SMA10"], label="SMA10", color="blue")
    ax.plot(data["Datetime"], data["SMA50"], label="SMA50", color="orange")
    ax.scatter(data["Datetime"].iloc[-1], entry_price, color="green", label=f"Entry: {signal}", zorder=5)
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

# Main Execution
st.title("Forex Trade Signal Generator")

st.write(f"Fetching data for {selected_pair}...")
data = fetch_data(ticker_symbol, period="5d", interval="15m")

if not data.empty:
    signal, entry_price = generate_signal(data)
    stop_loss = float(entry_price * 0.995)
    take_profit = float(entry_price + (entry_price - stop_loss) * risk_reward_ratio)
    lot_size = calculate_lot_size(account_balance, risk_percentage, entry_price, stop_loss)

    st.write(f"### Trade Signal for {selected_pair}")
    st.write(f"- Signal: **{signal}**")
    st.write(f"- Entry Price: **{entry_price:.2f}**")
    st.write(f"- Stop Loss: **{stop_loss:.2f}**")
    st.write(f"- Take Profit: **{take_profit:.2f}**")
    st.write(f"- Lot Size: **{lot_size:.2f}**")

    cumulative_strategy_return, backtest_data = backtest_strategy(data)
    st.write("### Backtest Performance")
    st.line_chart(cumulative_strategy_return)

    plot_chart(data, signal, entry_price, stop_loss, take_profit)
else:
    st.error("Failed to fetch data for the selected pair.")

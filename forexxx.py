import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator
from matplotlib.dates import DateFormatter, AutoDateLocator
import yfinance as yf
from forex_python.converter import CurrencyRates

# Currency Pair Selection
currency_pairs = {
    "XAUUSD": "XAUUSD=X",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X",
    "NZDUSD": "NZDUSD=X",
    "USDCAD": "USDCAD=X",
    "USDCHF": "USDCHF=X",
}

# Streamlit Sidebar
st.sidebar.title("Trade Signal Generator")
selected_pair = st.sidebar.selectbox("Select Currency Pair", options=list(currency_pairs.keys()))
selected_indicator = st.sidebar.selectbox(
    "Select Indicator",
    options=["SMA (10/50)", "EMA (10/50)", "RSI (14)"]
)
account_balance = st.sidebar.number_input("Account Balance (USD)", min_value=0.0, value=1000.0, step=100.0)
risk_percentage = st.sidebar.slider("Risk Percentage (%)", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
risk_reward_ratio = st.sidebar.slider("Risk/Reward Ratio", min_value=1.0, max_value=5.0, value=2.0, step=0.1)

# Fetch real-time forex data using yfinance with fallback
@st.cache_data
def fetch_forex_data(pair, base_currency, quote_currency):
    try:
        st.write(f"Fetching data for {pair} from Yahoo Finance...")
        data = yf.download(tickers=pair, period="5d", interval="15m", progress=False)
        if data.empty:
            st.warning(f"No data fetched for the pair: {pair}. Trying alternative source...")
            data = fetch_forex_data_fallback(base_currency, quote_currency)
        else:
            data.reset_index(inplace=True)
            data.rename(columns={"Datetime": "Datetime", "Open": "Open", "High": "High", "Low": "Low", "Close": "Close"}, inplace=True)
            data.set_index("Datetime", inplace=True)
        return data
    except Exception as e:
        st.error(f"Error fetching forex data: {e}")
        return pd.DataFrame()

# Fallback function to fetch forex rates using forex-python
def fetch_forex_data_fallback(base_currency, quote_currency):
    try:
        c = CurrencyRates()
        current_rate = c.get_rate(base_currency, quote_currency)
        if not current_rate:
            raise ValueError("Invalid currency pair or no data available.")
        timestamps = pd.date_range(end=pd.Timestamp.now(), periods=96, freq="15min")
        rates = [current_rate + np.random.uniform(-0.01, 0.01) for _ in timestamps]
        data = pd.DataFrame({
            "Datetime": timestamps,
            "Close": rates,
            "Open": rates + np.random.uniform(-0.01, 0.01, len(rates)),
            "High": rates + np.random.uniform(0.01, 0.02, len(rates)),
            "Low": rates - np.random.uniform(0.01, 0.02, len(rates)),
        })
        data.set_index("Datetime", inplace=True)
        return data
    except ValueError as ve:
        st.error(f"Value Error: {ve}")
        return generate_dummy_data()
    except Exception as e:
        st.error(f"Error fetching fallback forex data: {e}")
        return generate_dummy_data()

# Generate dummy data if all else fails
def generate_dummy_data():
    timestamps = pd.date_range(end=pd.Timestamp.now(), periods=96, freq="15min")
    rates = [100 + np.random.uniform(-1, 1) for _ in timestamps]
    data = pd.DataFrame({
        "Datetime": timestamps,
        "Close": rates,
        "Open": rates + np.random.uniform(-0.5, 0.5, len(rates)),
        "High": rates + np.random.uniform(0.5, 1, len(rates)),
        "Low": rates - np.random.uniform(0.5, 1, len(rates)),
    })
    data.set_index("Datetime", inplace=True)
    return data

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
        if indicator == "SMA (10/50)":
            data["Indicator1"] = SMAIndicator(data["Close"], window=10).sma_indicator()
            data["Indicator2"] = SMAIndicator(data["Close"], window=50).sma_indicator()
            signal_condition = data["Indicator1"].iloc[-1] > data["Indicator2"].iloc[-1]
        elif indicator == "EMA (10/50)":
            data["Indicator1"] = EMAIndicator(data["Close"], window=10).ema_indicator()
            data["Indicator2"] = EMAIndicator(data["Close"], window=50).ema_indicator()
            signal_condition = data["Indicator1"].iloc[-1] > data["Indicator2"].iloc[-1]
        elif indicator == "RSI (14)":
            data["RSI"] = RSIIndicator(data["Close"], window=14).rsi()
            signal_condition = data["RSI"].iloc[-1] < 30  # Buy signal when RSI is oversold

        last_close = data["Close"].iloc[-1]
        if signal_condition:
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
        ax.plot(data.index, data["Indicator1"], label="Indicator 1", color="blue")
        ax.plot(data.index, data["Indicator2"], label="Indicator 2", color="orange")
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

ticker_symbol = currency_pairs[selected_pair]
base_currency, quote_currency = selected_pair[:3], selected_pair[3:]
data = fetch_forex_data(ticker_symbol, base_currency, quote_currency)

if not data.empty:
    try:
        signal, entry_price = generate_signal(data, selected_indicator)
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

            if st.button("Read More"):
                st.write("**Detailed Signal Information:**")
                st.write(f"Indicator: {selected_indicator}")
                st.write(f"Account Balance: {account_balance}")
                st.write(f"Risk Percentage: {risk_percentage}%")
                st.write(f"Risk/Reward Ratio: {risk_reward_ratio}")

            plot_chart(data, signal, entry_price, stop_loss, take_profit)
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.error("Failed to fetch data for the selected pair.")

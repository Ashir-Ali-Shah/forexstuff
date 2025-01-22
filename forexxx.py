import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import yfinance as yf
import joblib

# Set page config for a white UI
st.set_page_config(page_title="Forex Trade Signal Generator", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

# App title
st.markdown("# 📈 Forex Trade Signal Generator")

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

# Sidebar layout
st.sidebar.markdown("## ⚙️ Settings")
selected_pair = st.sidebar.selectbox("🌍 Select Currency Pair", options=list(currency_pairs.keys()))
account_balance = st.sidebar.number_input("💰 Account Balance (USD)", min_value=0.0, value=1000.0, step=100.0)
risk_percentage = st.sidebar.slider("📉 Risk Percentage (%)", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
risk_reward_ratio = st.sidebar.slider("🎯 Risk/Reward Ratio", min_value=1.0, max_value=5.0, value=2.0, step=0.1)

# Load the prediction model
@st.cache_resource
def load_model(file_path):
    return joblib.load(file_path)

model = load_model('/mnt/data/forex_model_knn_use_indic.pkl')

# Predict function
def predict_trade(inputs):
    try:
        prediction = model.predict([inputs])
        return "Buy" if prediction[0] == 1 else "Sell"
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return "Unknown"

# Lot Size Calculation based on the risk percentage and stop loss
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

# Fetch forex data from Yahoo Finance
def fetch_forex_data(pair):
    try:
        data = yf.download(tickers=pair, period="5d", interval="15m", progress=False)
        if data.empty:
            raise ValueError(f"No data fetched for the selected pair: {pair}")
        data.reset_index(inplace=True)
        data.rename(columns={"Datetime": "datetime", "Open": "open", "High": "high", "Low": "low", "Close": "close"}, inplace=True)
        data.set_index("datetime", inplace=True)
        return data
    except Exception as e:
        st.error(f"Error fetching forex data: {e}")
        return pd.DataFrame()

# Calculate signal strength based on recent volatility
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

# Plotting function with Plotly
def plot_trade_signal_graph(entry_price, stop_loss, take_profit, lot_size, trade_type, signal_strength):
    try:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[datetime.now()], y=[entry_price], mode='markers', name='Entry Price', marker=dict(color='green', size=12)))
        fig.add_trace(go.Scatter(x=[datetime.now(), datetime.now()], y=[stop_loss, stop_loss], mode='lines', name='Stop Loss', line=dict(color='red', dash='dash')))
        fig.add_trace(go.Scatter(x=[datetime.now(), datetime.now()], y=[take_profit, take_profit], mode='lines', name='Take Profit', line=dict(color='blue', dash='dash')))
        fig.add_annotation(x=datetime.now(), y=(entry_price + stop_loss) / 2, text=f"Lot Size: {lot_size} ({trade_type}, {signal_strength})", showarrow=True, arrowhead=2, ax=-100, ay=-100, font=dict(size=12, color="black"))
        fig.update_layout(title=f"Trade Signal for {selected_pair}", xaxis_title="Date", yaxis_title="Price", showlegend=True, template="plotly_white")
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Error generating graph: {e}")

# Execute Trade
def execute_trade(entry_price, stop_loss, take_profit, lot_size, trade_type, signal_strength):
    trade_data = {
        "Currency Pair": selected_pair,
        "Entry Price": entry_price,
        "Stop Loss": stop_loss,
        "Take Profit": take_profit,
        "Lot Size": lot_size,
        "Risk %": risk_percentage,
        "Reward Ratio": risk_reward_ratio,
        "Trade Type": trade_type,
        "Signal Strength": signal_strength,
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    if 'trade_history' not in st.session_state:
        st.session_state.trade_history = pd.DataFrame(columns=trade_data.keys())
    st.session_state.trade_history = pd.concat([st.session_state.trade_history, pd.DataFrame([trade_data])], ignore_index=True)

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

        plot_trade_signal_graph(entry_price, stop_loss, take_profit, lot_size, trade_type, signal_strength)

        if st.button("Execute Trade"):
            execute_trade(entry_price, stop_loss, take_profit, lot_size, trade_type, signal_strength)
            st.success("Trade executed successfully!")

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.error("Failed to fetch data for the selected pair.")

if 'trade_history' in st.session_state and not st.session_state.trade_history.empty:
    st.markdown("### 📜 Trade History")
    st.dataframe(st.session_state.trade_history)
else:
    st.markdown("### 📜 Trade History is empty. Execute a trade to see the history.")

# Forex Price Prediction Subsection
st.markdown("## 🔮 Forex Price Prediction")
st.markdown("### Enter Features for Prediction")
open_price = st.number_input("Open Price", value=0.0)
high_price = st.number_input("High Price", value=0.0)
low_price = st.number_input("Low Price", value=0.0)
close_price = st.number_input("Close Price", value=0.0)
ma_50 = st.number_input("50-day Moving Average", value=0.0)
ma_200 = st.number_input("200-day Moving Average", value=0.0)
price_diff = st.number_input("Price Difference (Close - Open)", value=0.0)

if st.button("Predict Trade Direction"):
    inputs = [open_price, high_price, low_price, close_price, ma_50, ma_200, price_diff]
    trade_prediction = predict_trade(inputs)
    st.write(f"### 🧾 Predicted Trade Direction: **{trade_prediction}**")

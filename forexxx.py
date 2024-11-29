import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator
from time import sleep

# App title
st.markdown("# 📈 Forex Trade Signal Generator")

# Currency Pair Selection
currency_pairs = {
    "XAUUSD": "GLD",  # Using SPDR Gold Shares ETF as an alternative for XAUUSD
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

# Fetch forex data from Yahoo Finance using yfinance
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

# Plotting function with Plotly for beautiful and interactive charts
def plot_trade_signal_graph(entry_price, stop_loss, take_profit, lot_size, data, signal):
    try:
        fig = go.Figure()

        # Plotting the price data as candlesticks
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name="Price Data"
        ))

        # Add the entry price point
        fig.add_trace(go.Scatter(
            x=[datetime.now()],
            y=[entry_price],
            mode='markers',
            name='Entry Price',
            marker=dict(color='green', size=12)
        ))

        # Add the stop loss line
        fig.add_trace(go.Scatter(
            x=[datetime.now(), datetime.now()],
            y=[stop_loss, stop_loss],
            mode='lines',
            name='Stop Loss',
            line=dict(color='red', dash='dash')
        ))

        # Add the take profit line
        fig.add_trace(go.Scatter(
            x=[datetime.now(), datetime.now()],
            y=[take_profit, take_profit],
            mode='lines',
            name='Take Profit',
            line=dict(color='blue', dash='dash')
        ))

        # Add annotation for Lot Size
        fig.add_annotation(
            x=datetime.now(),
            y=(entry_price + stop_loss) / 2,
            text=f"Lot Size: {lot_size}",
            showarrow=True,
            arrowhead=2,
            ax=-100,
            ay=-100,
            font=dict(size=12, color="black")
        )

        # Update the layout
        fig.update_layout(
            title=f"Trade Signal for {selected_pair}",
            xaxis_title="Date",
            yaxis_title="Price",
            showlegend=True,
            template="plotly_dark"
        )

        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"Error generating graph: {e}")

# Generate signal based on basic indicators (Simple Example)
def generate_signal(data):
    try:
        # Calculate Moving Averages and RSI
        close_series = data["close"]
        sma_10 = SMAIndicator(close_series, window=10).sma_indicator()
        sma_50 = SMAIndicator(close_series, window=50).sma_indicator()
        rsi = RSIIndicator(close_series, window=14).rsi()

        # Simple Signal Strategy: Buy when SMA10 > SMA50 and RSI < 30 (indicating oversold)
        signal = "Sell"  # Default signal
        entry_price = close_series.iloc[-1]

        if sma_10.iloc[-1] > sma_50.iloc[-1] and rsi.iloc[-1] < 30:
            signal = "Buy"
        
        return signal, entry_price

    except Exception as e:
        st.error(f"Error generating trade signal: {e}")
        return "No Signal", 0.0

# Track trade history
def track_trade_history(signal, entry_price, stop_loss, take_profit, lot_size):
    trade_history = st.session_state.get('trade_history', [])
    trade = {
        'Signal': signal,
        'Entry Price': entry_price,
        'Stop Loss': stop_loss,
        'Take Profit': take_profit,
        'Lot Size': lot_size,
        'Date': datetime.now()
    }
    trade_history.append(trade)
    st.session_state['trade_history'] = trade_history

# Main Execution
ticker_symbol = currency_pairs[selected_pair]
data = fetch_forex_data(ticker_symbol)

if not data.empty:
    try:
        # Generate trade signal
        signal, entry_price = generate_signal(data)

        # Define the stop loss and take profit (using an example calculation)
        stop_loss = entry_price - 2  # Example stop loss (adjust as needed)
        take_profit = entry_price + 1.5  # Example take profit (adjust as needed)

        # Calculate lot size
        lot_size = calculate_lot_size(account_balance, risk_percentage, entry_price, stop_loss)

        # Display trade details
        st.markdown("### 📊 Trade Signal")
        st.write(f"**Entry Price**: {entry_price}")
        st.write(f"**Stop Loss**: {stop_loss}")
        st.write(f"**Take Profit**: {take_profit}")
        st.write(f"**Lot Size**: {lot_size}")

        # Plot the trade signal graph
        plot_trade_signal_graph(entry_price, stop_loss, take_profit, lot_size, data, signal)

        # Track trade history
        track_trade_history(signal, entry_price, stop_loss, take_profit, lot_size)

        # Display trade history as a table
        st.markdown("### 📝 Trade History")
        trade_history_df = pd.DataFrame(st.session_state.get('trade_history', []))
        st.dataframe(trade_history_df)

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.error("Failed to fetch data for the selected pair.")

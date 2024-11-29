import streamlit as st
import backtrader as bt
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

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

# Backtrader Strategy: Simple Moving Average Crossover
class SMACrossover(bt.SignalStrategy):
    def __init__(self):
        # Add the indicators to the strategy
        self.sma10 = bt.indicators.SimpleMovingAverage(self.data.close, period=10)
        self.sma50 = bt.indicators.SimpleMovingAverage(self.data.close, period=50)

    def next(self):
        # If SMA10 crosses above SMA50, it's a Buy signal
        if self.sma10 > self.sma50:
            if not self.position:
                self.buy()
        # If SMA10 crosses below SMA50, it's a Sell signal
        elif self.sma10 < self.sma50:
            if self.position:
                self.sell()

# Plotting function
def plot_strategy(data, strategy):
    try:
        # Setup the Backtrader Cerebro engine
        cerebro = bt.Cerebro()
        cerebro.addstrategy(strategy)
        
        # Convert the dataframe to a Backtrader feed
        data_feed = bt.feeds.PandasData(dataname=data)
        cerebro.adddata(data_feed)
        
        # Set initial cash and configure the broker
        cerebro.broker.set_cash(account_balance)
        
        # Set commission (instead of 'set_commission' method, directly assign a commission for the broker)
        cerebro.broker.set_commission(commission=0.001)
        
        # Run the strategy
        cerebro.run()
        
        # Plot the result
        cerebro.plot(style='candlestick')
    except Exception as e:
        st.error(f"Error generating chart: {e}")

# Main Execution
ticker_symbol = currency_pairs[selected_pair]
data = fetch_forex_data(ticker_symbol)

if not data.empty:
    try:
        # Define the signal, entry price, stop loss, take profit, and lot size
        signal, entry_price = "Buy", 2062.5  # Example entry
        stop_loss = entry_price - 2  # Example stop loss (adjust as needed)
        take_profit = entry_price + 1.5  # Example take profit (adjust as needed)
        lot_size = calculate_lot_size(account_balance, risk_percentage, entry_price, stop_loss)

        # Display trade details
        st.markdown("### 📊 Trade Signal")
        st.write(f"**Entry Price**: {entry_price}")
        st.write(f"**Stop Loss**: {stop_loss}")
        st.write(f"**Take Profit**: {take_profit}")
        st.write(f"**Lot Size**: {lot_size}")

        # Plot the strategy result (this is optional, and can be visualized with Backtrader)
        plot_strategy(data, SMACrossover)

        # Display details
        st.markdown("### 📈 Trade Details")
        st.write(f"**Account Balance:** ${account_balance}")
        st.write(f"**Risk Percentage:** {risk_percentage}%")
        st.write(f"**Risk/Reward Ratio:** {risk_reward_ratio}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.error("Failed to fetch data for the selected pair.")

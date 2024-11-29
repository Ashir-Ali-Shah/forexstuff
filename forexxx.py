import streamlit as st
import backtrader as bt
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

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
account_balance = st.sidebar.number_input("💰 Account Balance (USD)", min_value=0.0, value=1000.0, step=100.0)
risk_percentage = st.sidebar.slider("📉 Risk Percentage (%)", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
risk_reward_ratio = st.sidebar.slider("🎯 Risk/Reward Ratio", min_value=1.0, max_value=5.0, value=2.0, step=0.1)

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
        # Run the backtest with Backtrader strategy
        plot_strategy(data, SMACrossover)
        
        # Display details
        st.markdown("### 📊 Trade Details")
        st.write(f"**Account Balance:** ${account_balance}")
        st.write(f"**Risk Percentage:** {risk_percentage}%")
        st.write(f"**Risk/Reward Ratio:** {risk_reward_ratio}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.error("Failed to fetch data for the selected pair.")

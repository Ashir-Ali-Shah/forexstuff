import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
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

# Initialize trade history if not already present in session state
if 'trade_history' not in st.session_state:
    st.session_state.trade_history = pd.DataFrame(columns=["Currency Pair", "Entry Price", "Stop Loss", "Take Profit", "Lot Size", "Risk %", "Reward Ratio", "Date"])

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

# Calculate the Average True Range (ATR) for volatility-based stop loss and take profit
def calculate_atr(data, period=14):
    data['high-low'] = data['high'] - data['low']
    data['high-close'] = abs(data['high'] - data['close'].shift(1))
    data['low-close'] = abs(data['low'] - data['close'].shift(1))
    data['tr'] = data[['high-low', 'high-close', 'low-close']].max(axis=1)
    atr = data['tr'].rolling(window=period).mean().iloc[-1]  # Get the most recent ATR value
    return atr

# Plotting function with Plotly for beautiful and interactive charts
def plot_trade_signal_graph(entry_price, stop_loss, take_profit, lot_size):
    try:
        fig = go.Figure()

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

# Function to execute a trade and store in trade history
def execute_trade(entry_price, stop_loss, take_profit, lot_size):
    trade_data = {
        "Currency Pair": selected_pair,
        "Entry Price": entry_price,
        "Stop Loss": stop_loss,
        "Take Profit": take_profit,
        "Lot Size": lot_size,
        "Risk %": risk_percentage,
        "Reward Ratio": risk_reward_ratio,
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    # Using pd.concat() instead of append
    new_trade = pd.DataFrame([trade_data])
    st.session_state.trade_history = pd.concat([st.session_state.trade_history, new_trade], ignore_index=True)

# Main Execution
ticker_symbol = currency_pairs[selected_pair]
data = fetch_forex_data(ticker_symbol)

if not data.empty:
    try:
        # Calculate the ATR value for dynamic stop loss and take profit
        atr_value = calculate_atr(data)

        # Example entry price (get the last closing price)
        entry_price = data['close'].iloc[-1]

        # Dynamic stop loss and take profit based on ATR
        atr_multiplier = 1.5  # 1.5x ATR for stop loss
        stop_loss = entry_price - atr_value * atr_multiplier  # Stop loss below the entry price
        take_profit = entry_price + atr_value * risk_reward_ratio * atr_multiplier  # Take profit above the entry price

        # Calculate lot size based on user inputs
        lot_size = calculate_lot_size(account_balance, risk_percentage, entry_price, stop_loss)

        # Display trade details
        st.markdown("### 📊 Trade Signal")
        st.write(f"**Entry Price**: {entry_price}")
        st.write(f"**Stop Loss**: {stop_loss}")
        st.write(f"**Take Profit**: {take_profit}")
        st.write(f"**Lot Size**: {lot_size}")

        # Plot the trade signal graph
        plot_trade_signal_graph(entry_price, stop_loss, take_profit, lot_size)

        # Display details
        st.markdown("### 📈 Trade Details")
        st.write(f"**Account Balance:** ${account_balance}")
        st.write(f"**Risk Percentage:** {risk_percentage}%")
        st.write(f"**Risk/Reward Ratio:** {risk_reward_ratio}")

        # Add Execute Trade button
        if st.button("Execute Trade"):
            execute_trade(entry_price, stop_loss, take_profit, lot_size)
            st.success("Trade executed successfully!")

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.error("Failed to fetch data for the selected pair.")

# Display Trade History as a table
if not st.session_state.trade_history.empty:
    st.markdown("### 📜 Trade History")
    st.dataframe(st.session_state.trade_history)
else:
    st.markdown("### 📜 Trade History is empty. Execute a trade to see the history.")

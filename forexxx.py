import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import talib

# Set page configuration
st.set_page_config(page_title="Advanced Forex Trade Signal Generator", layout="wide")

# Custom CSS to enhance UI
st.markdown(
    """
    <style>
        body {
            background-color: #0E1117;
            color: white;
            font-family: Arial, sans-serif;
        }
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stButton > button {
            background: linear-gradient(45deg, #ff4b4b, #ff9966);
            color: white;
            font-size: 16px;
            border-radius: 8px;
            padding: 10px 24px;
            border: none;
            transition: all 0.3s ease-in-out;
        }
        .stButton > button:hover {
            background: linear-gradient(45deg, #ff9966, #ff4b4b);
            transform: scale(1.05);
            box-shadow: 0px 4px 10px rgba(255, 75, 75, 0.5);
        }
        .buy-signal {
            background-color: #00ff00;
            color: black;
            padding: 10px;
            border-radius: 5px;
            font-weight: bold;
            text-align: center;
            margin: 10px 0;
        }
        .sell-signal {
            background-color: #ff0000;
            color: white;
            padding: 10px;
            border-radius: 5px;
            font-weight: bold;
            text-align: center;
            margin: 10px 0;
        }
        .neutral-signal {
            background-color: #ffff00;
            color: black;
            padding: 10px;
            border-radius: 5px;
            font-weight: bold;
            text-align: center;
            margin: 10px 0;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("# 📈 Advanced Forex Trade Signal Generator")

currency_pairs = {
    "XAUUSD": "XAUUSD1.csv",
    "EURUSD": "EURUSD1.csv",
    "GBPUSD": "GBPUSD1.csv",
    "USDCAD": "USDCAD1.csv",
    "USDCHF": "USDCHF1.csv",
    "USDJPY": "USDJPY1.csv",
    "AUDCAD": "AUDCAD1.csv",
    "AUDCHF": "AUDCHF1.csv",
    "AUDJPY": "AUDJPY1.csv",
    "AUDNZD": "AUDNZD1.csv",
}

# Technical Analysis Functions
def calculate_rsi(data, window=14):
    """Calculate RSI"""
    try:
        return talib.RSI(data.values, timeperiod=window)
    except:
        # Fallback manual calculation
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    try:
        macd, macd_signal, macd_hist = talib.MACD(data.values, fastperiod=fast, slowperiod=slow, signalperiod=signal)
        return macd, macd_signal, macd_hist
    except:
        # Fallback manual calculation
        ema_fast = data.ewm(span=fast).mean()
        ema_slow = data.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_hist = macd - macd_signal
        return macd.values, macd_signal.values, macd_hist.values

def calculate_bollinger_bands(data, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    try:
        upper, middle, lower = talib.BBANDS(data.values, timeperiod=window, nbdevup=num_std, nbdevdn=num_std)
        return upper, middle, lower
    except:
        # Fallback manual calculation
        rolling_mean = data.rolling(window=window).mean()
        rolling_std = data.rolling(window=window).std()
        upper = rolling_mean + (rolling_std * num_std)
        lower = rolling_mean - (rolling_std * num_std)
        return upper.values, rolling_mean.values, lower.values

def calculate_moving_averages(data, short_window=10, long_window=30):
    """Calculate moving averages"""
    short_ma = data.rolling(window=short_window).mean()
    long_ma = data.rolling(window=long_window).mean()
    return short_ma, long_ma

def generate_trading_signal(data, current_price):
    """Generate comprehensive trading signal"""
    
    # Calculate technical indicators
    rsi = calculate_rsi(data['Close'])
    macd, macd_signal, macd_hist = calculate_macd(data['Close'])
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(data['Close'])
    short_ma, long_ma = calculate_moving_averages(data['Close'])
    
    # Get latest values
    latest_rsi = rsi[-1] if not np.isnan(rsi[-1]) else 50
    latest_macd = macd[-1] if not np.isnan(macd[-1]) else 0
    latest_macd_signal = macd_signal[-1] if not np.isnan(macd_signal[-1]) else 0
    latest_bb_upper = bb_upper[-1] if not np.isnan(bb_upper[-1]) else current_price * 1.02
    latest_bb_lower = bb_lower[-1] if not np.isnan(bb_lower[-1]) else current_price * 0.98
    latest_short_ma = short_ma.iloc[-1] if not np.isnan(short_ma.iloc[-1]) else current_price
    latest_long_ma = long_ma.iloc[-1] if not np.isnan(long_ma.iloc[-1]) else current_price
    
    # Signal scoring system
    buy_signals = 0
    sell_signals = 0
    
    # RSI signals
    if latest_rsi < 30:  # Oversold
        buy_signals += 2
    elif latest_rsi > 70:  # Overbought
        sell_signals += 2
    elif latest_rsi < 40:
        buy_signals += 1
    elif latest_rsi > 60:
        sell_signals += 1
    
    # MACD signals
    if latest_macd > latest_macd_signal:
        buy_signals += 1
    else:
        sell_signals += 1
    
    # Moving Average signals
    if latest_short_ma > latest_long_ma:
        buy_signals += 1
    else:
        sell_signals += 1
    
    # Bollinger Bands signals
    if current_price <= latest_bb_lower:
        buy_signals += 2
    elif current_price >= latest_bb_upper:
        sell_signals += 2
    
    # Price momentum
    price_change = (current_price - data['Close'].iloc[-5]) / data['Close'].iloc[-5] * 100
    if price_change > 0.5:
        buy_signals += 1
    elif price_change < -0.5:
        sell_signals += 1
    
    # Determine signal
    if buy_signals > sell_signals + 1:
        signal = "BUY"
        confidence = min(90, 50 + (buy_signals - sell_signals) * 10)
    elif sell_signals > buy_signals + 1:
        signal = "SELL"
        confidence = min(90, 50 + (sell_signals - buy_signals) * 10)
    else:
        signal = "HOLD"
        confidence = 30
    
    return {
        'signal': signal,
        'confidence': confidence,
        'rsi': latest_rsi,
        'macd': latest_macd,
        'bb_upper': latest_bb_upper,
        'bb_lower': latest_bb_lower,
        'short_ma': latest_short_ma,
        'long_ma': latest_long_ma,
        'buy_score': buy_signals,
        'sell_score': sell_signals
    }

def calculate_tp_sl(entry_price, signal, risk_reward_ratio, atr_value=None, volatility_multiplier=1.5):
    """Calculate Take Profit and Stop Loss levels"""
    
    # Use ATR for dynamic SL/TP or fallback to percentage
    if atr_value and not np.isnan(atr_value):
        sl_distance = atr_value * volatility_multiplier
    else:
        # Fallback: use percentage based on currency pair
        sl_distance = entry_price * 0.01  # 1% default
    
    if signal == "BUY":
        stop_loss = entry_price - sl_distance
        take_profit = entry_price + (sl_distance * risk_reward_ratio)
    elif signal == "SELL":
        stop_loss = entry_price + sl_distance
        take_profit = entry_price - (sl_distance * risk_reward_ratio)
    else:
        stop_loss = entry_price
        take_profit = entry_price
    
    return round(take_profit, 5), round(stop_loss, 5)

def calculate_atr(data, window=14):
    """Calculate Average True Range"""
    try:
        return talib.ATR(data['High'].values, data['Low'].values, data['Close'].values, timeperiod=window)
    except:
        # Manual ATR calculation
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(window=window).mean().values

# Sidebar
st.sidebar.markdown("## ⚙️ Settings")
st.sidebar.markdown("---")
selected_pair = st.sidebar.selectbox("🌍 Select Currency Pair", options=list(currency_pairs.keys()))
st.sidebar.markdown("---")
account_balance = st.sidebar.number_input("💰 Account Balance (USD)", min_value=0.0, value=1000.0, step=100.0)
risk_percentage = st.sidebar.slider("📉 Risk Percentage (%)", min_value=0.5, max_value=10.0, value=2.0, step=0.1)
risk_reward_ratio = st.sidebar.slider("🎯 Risk/Reward Ratio", min_value=1.0, max_value=5.0, value=2.0, step=0.1)
volatility_multiplier = st.sidebar.slider("📊 Volatility Multiplier", min_value=1.0, max_value=3.0, value=1.5, step=0.1)
st.sidebar.markdown("---")

# Load and process data
try:
    file_path = currency_pairs[selected_pair]
    data = pd.read_csv(file_path, delimiter="\t", header=None)
    data.columns = ["DateTime", "Open", "High", "Low", "Close", "Volume"]
    data["DateTime"] = pd.to_datetime(data["DateTime"])
    data = data.sort_values(by="DateTime").reset_index(drop=True)
    
    # Ensure we have enough data
    if len(data) < 100:
        st.error("Insufficient data for analysis. Need at least 100 data points.")
        st.stop()
    
    current_price = data["Close"].iloc[-1]
    
    # Calculate ATR
    atr_values = calculate_atr(data)
    current_atr = atr_values[-1] if not np.isnan(atr_values[-1]) else None
    
    # Generate trading signal
    signal_data = generate_trading_signal(data, current_price)
    
    # Calculate TP and SL
    take_profit, stop_loss = calculate_tp_sl(
        current_price, 
        signal_data['signal'], 
        risk_reward_ratio,
        current_atr,
        volatility_multiplier
    )
    
    # Display current settings
    st.markdown(f"### ⚙️ Current Settings")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Currency Pair", selected_pair)
    with col2:
        st.metric("Account Balance", f"${account_balance:,.2f}")
    with col3:
        st.metric("Risk %", f"{risk_percentage}%")
    with col4:
        st.metric("R:R Ratio", f"1:{risk_reward_ratio}")
    
    # Display trading signal
    st.markdown("---")
    st.markdown("## 🎯 TRADING SIGNAL")
    
    signal_class = "buy-signal" if signal_data['signal'] == "BUY" else "sell-signal" if signal_data['signal'] == "SELL" else "neutral-signal"
    
    st.markdown(f"""
    <div class="{signal_class}">
        <h2>{signal_data['signal']} SIGNAL</h2>
        <p>Confidence: {signal_data['confidence']}%</p>
        <p>Current Price: {current_price:.5f}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display TP/SL levels
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("💰 Take Profit", f"{take_profit:.5f}", 
                 f"{((take_profit - current_price) / current_price * 100):.2f}%")
    with col2:
        st.metric("🛑 Stop Loss", f"{stop_loss:.5f}", 
                 f"{((stop_loss - current_price) / current_price * 100):.2f}%")
    with col3:
        risk_amount = account_balance * (risk_percentage / 100)
        potential_profit = risk_amount * risk_reward_ratio
        st.metric("💵 Potential Profit", f"${potential_profit:.2f}")
    
    # Technical indicators summary
    st.markdown("---")
    st.markdown("## 📊 Technical Analysis Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        rsi_color = "🟢" if signal_data['rsi'] < 40 else "🔴" if signal_data['rsi'] > 60 else "🟡"
        st.metric(f"{rsi_color} RSI (14)", f"{signal_data['rsi']:.1f}")
    
    with col2:
        macd_color = "🟢" if signal_data['macd'] > 0 else "🔴"
        st.metric(f"{macd_color} MACD", f"{signal_data['macd']:.5f}")
    
    with col3:
        ma_color = "🟢" if signal_data['short_ma'] > signal_data['long_ma'] else "🔴"
        st.metric(f"{ma_color} MA Cross", "Bullish" if signal_data['short_ma'] > signal_data['long_ma'] else "Bearish")
    
    with col4:
        bb_position = "Upper" if current_price >= signal_data['bb_upper'] else "Lower" if current_price <= signal_data['bb_lower'] else "Middle"
        bb_color = "🔴" if bb_position == "Upper" else "🟢" if bb_position == "Lower" else "🟡"
        st.metric(f"{bb_color} BB Position", bb_position)
    
    # Position sizing
    st.markdown("---")
    st.markdown("## 💼 Position Sizing")
    
    risk_amount = account_balance * (risk_percentage / 100)
    pip_value = 0.0001 if selected_pair != "USDJPY" else 0.01
    
    if signal_data['signal'] != "HOLD":
        sl_distance_pips = abs(current_price - stop_loss) / pip_value
        lot_size = risk_amount / (sl_distance_pips * 10)  # Assuming $10 per pip for 1 lot
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("💰 Risk Amount", f"${risk_amount:.2f}")
        with col2:
            st.metric("📏 SL Distance", f"{sl_distance_pips:.1f} pips")
        with col3:
            st.metric("📊 Suggested Lot Size", f"{lot_size:.2f}")
    
    # Execute trade button
    if signal_data['signal'] != "HOLD":
        st.markdown("---")
        if st.button(f"🚀 Generate Trade Setup for {signal_data['signal']} Signal", key="execute_trade"):
            trade_setup = {
                "Currency Pair": selected_pair,
                "Signal": signal_data['signal'],
                "Entry Price": current_price,
                "Take Profit": take_profit,
                "Stop Loss": stop_loss,
                "Confidence": f"{signal_data['confidence']}%",
                "Risk Amount": f"${risk_amount:.2f}",
                "Potential Profit": f"${potential_profit:.2f}",
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            st.markdown("### 📋 Trade Setup Summary")
            trade_df = pd.DataFrame([trade_setup])
            st.dataframe(trade_df, use_container_width=True)
            
            # Copy-paste friendly format
            st.markdown("### 📝 Copy-Paste Format")
            st.code(f"""
FOREX TRADE SETUP - {selected_pair}
================================
Signal: {signal_data['signal']}
Entry: {current_price:.5f}
TP: {take_profit:.5f}
SL: {stop_loss:.5f}
Confidence: {signal_data['confidence']}%
Risk: ${risk_amount:.2f}
Potential Profit: ${potential_profit:.2f}
R:R Ratio: 1:{risk_reward_ratio}
Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            """)
    
    # Price chart
    st.markdown("---")
    st.markdown(f"### 📈 Price Chart - {selected_pair}")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), 
                                   gridspec_kw={'height_ratios': [3, 1]})
    
    # Main price chart
    recent_data = data.tail(200)
    ax1.plot(recent_data["DateTime"], recent_data["Close"], 
             label="Close Price", color="lightblue", linewidth=1.5)
    
    # Add moving averages
    short_ma, long_ma = calculate_moving_averages(recent_data['Close'])
    ax1.plot(recent_data["DateTime"], short_ma, 
             label="MA 10", color="orange", linewidth=1, alpha=0.8)
    ax1.plot(recent_data["DateTime"], long_ma, 
             label="MA 30", color="red", linewidth=1, alpha=0.8)
    
    # Add Bollinger Bands
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(recent_data['Close'])
    ax1.plot(recent_data["DateTime"], bb_upper, 
             label="BB Upper", color="gray", linestyle="--", alpha=0.6)
    ax1.plot(recent_data["DateTime"], bb_lower, 
             label="BB Lower", color="gray", linestyle="--", alpha=0.6)
    ax1.fill_between(recent_data["DateTime"], bb_upper, bb_lower, 
                     alpha=0.1, color="gray")
    
    # Mark current levels
    current_time = recent_data["DateTime"].iloc[-1]
    ax1.axhline(y=current_price, color='white', linestyle='-', 
                linewidth=2, label=f'Current: {current_price:.5f}')
    
    if signal_data['signal'] != "HOLD":
        ax1.axhline(y=take_profit, color='green', linestyle='--', 
                    linewidth=2, label=f'TP: {take_profit:.5f}')
        ax1.axhline(y=stop_loss, color='red', linestyle='--', 
                    linewidth=2, label=f'SL: {stop_loss:.5f}')
    
    ax1.set_title(f"{selected_pair} - {signal_data['signal']} Signal ({signal_data['confidence']}% confidence)")
    ax1.set_ylabel("Price")
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # RSI subplot
    rsi_data = calculate_rsi(recent_data['Close'])
    ax2.plot(recent_data["DateTime"], rsi_data, color='purple', linewidth=1.5)
    ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7)
    ax2.axhline(y=30, color='green', linestyle='--', alpha=0.7)
    ax2.axhline(y=50, color='gray', linestyle='-', alpha=0.5)
    ax2.set_ylabel("RSI")
    ax2.set_xlabel("Date")
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
except FileNotFoundError:
    st.error(f"Data file not found for {selected_pair}. Please ensure the CSV file exists.")
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.info("Please make sure your data files are in the correct format with columns: DateTime, Open, High, Low, Close, Volume")

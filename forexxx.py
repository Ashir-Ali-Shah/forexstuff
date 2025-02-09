import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Forex Trade Signal Generator", layout="wide")
st.markdown("# 📈 Forex Trade Signal Generator")

currency_pairs = {
    "XAUUSD": "XAUUSD1.csv",
    "EURUSD": "EURUSD1.csv",
}

# Sidebar
st.sidebar.markdown("## ⚙️ Settings")
selected_pair = st.sidebar.selectbox("🌍 Select Currency Pair", options=list(currency_pairs.keys()))
account_balance = st.sidebar.number_input("💰 Account Balance (USD)", min_value=0.0, value=1000.0, step=100.0)
risk_percentage = st.sidebar.slider("📉 Risk Percentage (%)", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
risk_reward_ratio = st.sidebar.slider("🎯 Risk/Reward Ratio", min_value=1.0, max_value=5.0, value=2.0, step=0.1)

st.markdown(f"### ⚙️ Selected Settings")
st.markdown(f"**Currency Pair:** {selected_pair}")
st.markdown(f"**Account Balance:** ${account_balance}")
st.markdown(f"**Risk Percentage:** {risk_percentage}%")
st.markdown(f"**Risk/Reward Ratio:** {risk_reward_ratio}")

file_path = currency_pairs[selected_pair]
data = pd.read_csv(file_path, delimiter="\t", header=None)
data.columns = ["DateTime", "Open", "High", "Low", "Close", "Volume"]
data["DateTime"] = pd.to_datetime(data["DateTime"])
data["month"] = data["DateTime"].dt.strftime('%B')
data[["Open", "High", "Low", "Close", "Volume"]] = data[["Open", "High", "Low", "Close", "Volume"]].apply(pd.to_numeric)

# Lot Size Calculation (Adjusted for Risk/Reward Ratio)
def calculate_lot_size(balance, risk_percent, risk_reward_ratio, entry_price, stop_loss, pair):
    try:
        risk_amount = balance * (risk_percent / 100)
        pip_risk = abs(entry_price - stop_loss)
        if pip_risk == 0:
            return 0
        
        if pair == "EURUSD":
            lot_size = (risk_amount / pip_risk) * 100000  # Corrected lot size formula for EURUSD
        elif pair == "XAUUSD":
            pip_value = 10  # Standard pip value for XAUUSD in USD
            lot_size = risk_amount / (pip_risk * pip_value)  # Corrected lot size formula for XAUUSD
        else:
            lot_size = 0
        
        return round(lot_size, 2)
    except Exception as e:
        st.error(f"Error calculating lot size: {e}")
        return 0

# Feature Engineering
def create_lagged_features(df, lags=2):
    for i in range(1, lags + 1):
        df[f'Close_Lag_{i}'] = df['Close'].shift(i)
    df.dropna(inplace=True)
    return df

data = create_lagged_features(data)
features = [col for col in data.columns if 'Close_Lag' in col]
target = 'Close'

scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])

data_sample = data.tail(20000).copy()
train_size = int(len(data_sample) * 0.8)
train, test = data_sample.iloc[:train_size], data_sample.iloc[train_size:]

X_train, y_train = train[features], train[target]
X_test, y_test = test[features], test[target]

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

def predict_closing_price(input_values):
    input_values = np.array(input_values).reshape(1, -1)
    input_values = scaler.transform(input_values)
    prediction = model.predict(input_values)
    return prediction[0]

# Predict Closing Price for Both Currencies
st.markdown(f"## 📈 Predict Closing Price for {selected_pair}")
close_lag_1 = st.number_input(f"Enter Last Closing Price for {selected_pair}", value=0.0, step=0.01, key=f"{selected_pair}_lag_1")
close_lag_2 = st.number_input(f"Enter Second Last Closing Price for {selected_pair}", value=0.0, step=0.01, key=f"{selected_pair}_lag_2")

if st.button(f"Predict {selected_pair} Closing Price"):
    if close_lag_1 and close_lag_2:
        predicted_price = predict_closing_price([close_lag_1, close_lag_2])
        st.success(f"Predicted Closing Price for {selected_pair}: {predicted_price}")
    else:
        st.error("Please enter both closing prices.")

# Calculate and Display Lot Size for Both Currencies
entry_price = data["Close"].iloc[-1]
stop_loss = entry_price - 2
take_profit = entry_price + (2 * risk_reward_ratio)
lot_size = calculate_lot_size(account_balance, risk_percentage, risk_reward_ratio, entry_price, stop_loss, selected_pair)

st.markdown(f"## 💰 Calculated Lot Size for {selected_pair}: {lot_size}")

# Execute Trade
if st.button("Execute Trade"):
    trade_data = {
        "Currency Pair": selected_pair,
        "Entry Price": entry_price,
        "Stop Loss": stop_loss,
        "Take Profit": take_profit,
        "Lot Size": lot_size,
    }
    trade_df = pd.DataFrame([trade_data])
    st.markdown("### 📜 Trade Execution Details")
    st.dataframe(trade_df)

# Dark Themed Graphs
plt.style.use('dark_background')
st.markdown(f"### 📉 Close Price Over Time ({selected_pair})")
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(data["month"], data["Close"], label="Close Price", alpha=0.8, color="lightblue")
ax.set_title(f"Close Price Over Time - {selected_pair}")
ax.set_xlabel("Month")
ax.set_ylabel("Close Price")
ax.legend()
st.pyplot(fig)

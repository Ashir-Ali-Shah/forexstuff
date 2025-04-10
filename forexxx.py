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

# Set page configuration
st.set_page_config(page_title="Forex Trade Signal Generator", layout="wide")

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
        .stSelectbox, .stRadio, .stNumberInput, .stSlider {
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("# 📈 Forex Trade Signal Generator")

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

# Sidebar
st.sidebar.markdown("## ⚙️ Settings")
st.sidebar.markdown("---")
selected_pair = st.sidebar.selectbox("🌍 Select Currency Pair", options=list(currency_pairs.keys()))
st.sidebar.markdown("---")
st.sidebar.markdown("---")
account_balance = st.sidebar.number_input("💰 Account Balance (USD)", min_value=0.0, value=1000.0, step=100.0)
risk_percentage = st.sidebar.slider("📉 Risk Percentage (%)", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
risk_reward_ratio = st.sidebar.slider("🎯 Risk/Reward Ratio", min_value=1.0, max_value=5.0, value=2.0, step=0.1)
st.sidebar.markdown("---")

st.markdown(f"### ⚙️ Selected Settings")
st.markdown(f"**Currency Pair:** {selected_pair}")
st.markdown(f"**Account Balance:** ${account_balance}")
st.markdown(f"**Risk Percentage:** {risk_percentage}%")
st.markdown(f"**Risk/Reward Ratio:** {risk_reward_ratio}")

file_path = currency_pairs[selected_pair]
data = pd.read_csv(file_path, delimiter="\t", header=None)
data.columns = ["DateTime", "Open", "High", "Low", "Close", "Volume"]
data["DateTime"] = pd.to_datetime(data["DateTime"])
data = data.sort_values(by="DateTime").reset_index(drop=True)

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

st.markdown(f"## 📈 Predict Closing Price for {selected_pair}")
close_lag_1 = st.number_input(f"Enter Last Closing Price for {selected_pair}", value=0.0, step=0.01)
close_lag_2 = st.number_input(f"Enter Second Last Closing Price for {selected_pair}", value=0.0, step=0.01)

if st.button(f"Predict {selected_pair} Closing Price"):
    if close_lag_1 and close_lag_2:
        predicted_price = predict_closing_price([close_lag_1, close_lag_2])
        st.success(f"Predicted Closing Price for {selected_pair}: {predicted_price}")
    else:
        st.error("Please enter both closing prices.")

entry_price = data["Close"].iloc[-1]
pip_value = 0.0001 / entry_price
margin = account_balance * (risk_percentage / 100)
maximum_loss = ((pip_value * 30) / entry_price) * margin
maximum_loss_portfolio = 0.01 * margin
lot_size = (0.01 * margin * entry_price) / (30 * pip_value)
contract_size = lot_size * entry_price

st.markdown(f"## 📉 Calculated Lot Size for {selected_pair}: {lot_size}")
st.markdown(f"**Maximum Loss:** ${maximum_loss}")
st.markdown(f"**Maximum Loss for Portfolio:** ${maximum_loss_portfolio}")
st.markdown(f"**Contract Size:** {contract_size}")

if st.button("Execute Trade"):
    trade_data = {
        "Currency Pair": selected_pair,
        "Entry Price": entry_price,
        "Lot Size": lot_size,
        "Maximum Loss Portfolio": maximum_loss_portfolio,
        "Contract Size": contract_size,
    }
    trade_df = pd.DataFrame([trade_data])
    st.markdown("### 🐜 Trade Execution Details")
    st.dataframe(trade_df)

st.markdown(f"### 📉 Close Price Over Time ({selected_pair})")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(data["DateTime"], data["Close"], label="Close Price", alpha=0.8, color="lightblue")
ax.set_title(f"Close Price Over Time - {selected_pair}")
ax.set_xlabel("Date")
ax.set_ylabel("Close Price")
ax.legend()
st.pyplot(fig)

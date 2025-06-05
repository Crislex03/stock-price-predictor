
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

# Load model and scaler
model = joblib.load("predictive_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("📈 Predictive AI Stock Price Model")

st.markdown("""
Enter the latest stock price, volume, and the last 5-day average to get a prediction.
""")

# User inputs
price_lag1 = st.number_input("Previous Day Price", value=150.0)
price_ma5 = st.number_input("5-Day Moving Average", value=148.0)
volume = st.number_input("Current Volume", value=500)

if st.button("Predict Next Price"):
    X_input = scaler.transform([[price_lag1, price_ma5, volume]])
    prediction = model.predict(X_input)[0]
    st.success(f"📊 Predicted Next Price: ${prediction:.2f}")

import streamlit as st
import pandas as pd
import pickle
import os
from datetime import datetime

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="DHT22 Temperature Predictor",
    page_icon="🌡️",
    layout="centered"
)

# ----------------------------
# Safe model loading
# ----------------------------
model_path = os.path.join("model", "model.pkl")
model = pickle.load(open(model_path, "rb"))

# ----------------------------
# Header
# ----------------------------
st.title("🌡️ DHT22 Temperature Prediction System")
st.markdown("XGBoost-based IoT ML model using time-series sensor data")

st.divider()

# ----------------------------
# Input Section
# ----------------------------
st.subheader("📊 Sensor Inputs")

col1, col2 = st.columns(2)

with col1:
    humidity = st.slider("Humidity (%)", 0.0, 100.0, 50.0)

    temp_lag1 = st.number_input("Temp Lag1 (°C)", 10.0, 50.0, 25.0)
    temp_lag2 = st.number_input("Temp Lag2 (°C)", 10.0, 50.0, 25.0)

with col2:
    now = datetime.now()

    hour = st.number_input("Hour", 0, 23, now.hour)
    minute = st.number_input("Minute", 0, 59, now.minute)
    second = st.number_input("Second", 0, 59, now.second)

    hum_lag1 = st.number_input("Humidity Lag1 (%)", 0.0, 100.0, 50.0)

# ----------------------------
# Feature engineering (must match training logic)
# ----------------------------
temp_roll_mean_3 = (temp_lag1 + temp_lag2) / 2
hum_roll_mean_3 = hum_lag1

st.divider()

# ----------------------------
# Prediction
# ----------------------------
if st.button("🔮 Predict Temperature"):

    input_df = pd.DataFrame([[
        humidity,
        hour, minute, second,
        temp_lag1, temp_lag2,
        hum_lag1,
        temp_roll_mean_3, hum_roll_mean_3
    ]], columns=[
        'Humidity (%)',
        'hour', 'minute', 'second',
        'temp_lag1', 'temp_lag2',
        'hum_lag1',
        'temp_roll_mean_3', 'hum_roll_mean_3'
    ])

    prediction = model.predict(input_df)[0]

    st.success(f"🌡️ Predicted Temperature: **{prediction:.2f} °C**")

# ----------------------------
# Dataset preview
# ----------------------------
st.divider()
st.subheader("📁 Dataset Preview")

df = pd.read_csv("data_3hrs.csv", encoding="latin1")
st.dataframe(df.head(10))

# ----------------------------
# Footer
# ----------------------------
st.divider()
st.caption("IoT ML Project | XGBoost | DHT22 Sensor Prediction System")
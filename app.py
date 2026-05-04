import streamlit as st
import pandas as pd
import pickle
from datetime import datetime

# Load model
model = pickle.load(open("model/model.pkl", "rb"))

st.title("🌡️ DHT22 Temperature Prediction App (XGBoost)")

st.write("Predict Temperature (°C) using humidity and time features")

# ----------------------------
# INPUTS
# ----------------------------
humidity = st.slider("Humidity (%)", 0.0, 100.0, 50.0)

# Use current time as default features
now = datetime.now()

hour = st.number_input("Hour", 0, 23, now.hour)
minute = st.number_input("Minute", 0, 59, now.minute)
second = st.number_input("Second", 0, 59, now.second)

# Lag features (simplified demo)
temp_lag1 = st.number_input("Previous Temp (°C)", 20.0, 40.0, 25.0)
temp_lag2 = st.number_input("Temp Lag2 (°C)", 20.0, 40.0, 25.0)
hum_lag1 = st.number_input("Previous Humidity (%)", 0.0, 100.0, 50.0)

temp_roll_mean_3 = temp_lag1
hum_roll_mean_3 = hum_lag1

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict Temperature"):
    input_data = pd.DataFrame([[
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

    prediction = model.predict(input_data)[0]

    st.success(f"🌡️ Predicted Temperature: {prediction:.2f} °C")

# ----------------------------
# DATA PREVIEW
# ----------------------------
st.divider()
st.write("Dataset Preview")

df = pd.read_csv("data_3hrs.csv", encoding="latin1")
st.dataframe(df.head())
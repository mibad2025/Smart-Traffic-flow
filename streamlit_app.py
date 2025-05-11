# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import gdown
from datetime import datetime

# --- Load model from Google Drive ---
file_id = "18iSMMA7dbemFayE3h6Qu63LOgcZ6QlGx" 
url = f"https://drive.google.com/uc?id={file_id}"
model_path = "rf_model.pkl"

if not os.path.exists(model_path):
    gdown.download(url, model_path, quiet=False)

model = joblib.load(model_path)

# --- Streamlit UI ---
st.title("🚦 Smart Traffic Flow Predictor")

# Input fields
road_segment_id = st.number_input("Road Segment ID", min_value=1, max_value=100, value=1)
avg_vehicle_speed = st.slider("Average Vehicle Speed (km/h)", 0, 120, 60)
weather_conditions = st.selectbox("Weather Conditions", ["Clear", "Fog", "Rain", "Snow"])
hour = st.slider("Hour of Day", 0, 23, datetime.now().hour)
day_of_week = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, datetime.now().weekday())

# Weather encoding
weather_map = {"Clear": 0, "Fog": 1, "Rain": 2, "Snow": 3}
weather_encoded = weather_map[weather_conditions]

# Predict
if st.button("Predict Traffic Volume"):
    input_data = np.array([[road_segment_id, avg_vehicle_speed, weather_encoded, hour, day_of_week]])
    prediction = int(model.predict(input_data)[0])
    st.success(f"Predicted Traffic Volume: {prediction} vehicles")

    # Save to history in session
    if "history" not in st.session_state:
        st.session_state.history = []

    st.session_state.history.append({
        "timestamp": datetime.now(),
        "road_segment_id": road_segment_id,
        "avg_vehicle_speed": avg_vehicle_speed,
        "weather_conditions": weather_conditions,
        "hour": hour,
        "day_of_week": day_of_week,
        "predicted_volume": prediction
    })

# View history
if "history" in st.session_state and st.checkbox("Show Prediction History"):
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df)

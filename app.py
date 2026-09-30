import streamlit as st
import pandas as pd
import joblib
import os

# Load the trained pipeline
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "crop_model.pkl")
model = joblib.load(MODEL_PATH)

st.title("🌱 Crop Suitability Predictor")
st.write("Enter soil nutrient values and environmental conditions to get a recommended crop.")

# Input fields
n = st.number_input("Nitrogen (N)", min_value=0.0, step=0.1)
p = st.number_input("Phosphorus (P)", min_value=0.0, step=0.1)
k = st.number_input("Potassium (K)", min_value=0.0, step=0.1)
temp = st.number_input("Temperature (°C)", min_value=-50.0, max_value=60.0, step=0.1)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, step=0.1)
ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, step=0.01)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, step=0.1)

if st.button("Predict Crop"):
    # Create DataFrame for a single sample
    input_df = pd.DataFrame({
        "N": [n],
        "P": [p],
        "K": [k],
        "temperature": [temp],
        "humidity": [humidity],
        "ph": [ph],
        "rainfall": [rainfall],
    })
    # Predict
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df).max() * 100
    st.success(f"**Recommended Crop:** {pred}")
    st.info(f"Prediction confidence: {prob:.2f}%")

import streamlit as st
import joblib
import numpy as np

# Load model, scaler, and label encoder
model = joblib.load("best_flood_risk_model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

# App title
st.title("🌊 Flood Risk Prediction App")

st.write("Enter the values below to predict the flood risk level:")

# Input fields for the 5 features
rainfall = st.number_input("Historical Rainfall Intensity (mm/hr)", min_value=0.0, step=0.1)
elevation = st.number_input("Elevation (m)", step=0.1)
drainage_density = st.number_input("Drainage Density (km/km²)", min_value=0.0, step=0.01)
storm_drain_proximity = st.number_input("Storm Drain Proximity (m)", min_value=0.0, step=0.1)
return_period = st.number_input("Return Period (years)", min_value=1, step=1)

# Prediction button
if st.button("Predict Flood Risk Level"):
    # Arrange features in the same order as training
    features = np.array([[rainfall, elevation, drainage_density, storm_drain_proximity, return_period]])
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict
    prediction = model.predict(features_scaled)
    risk_label = le.inverse_transform(prediction)[0]
    
    # Display result
    st.success(f"Predicted Flood Risk Level: **{risk_label}**")

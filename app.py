import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open("model/model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

# Title
st.title("🏠 House Price Prediction")

st.write("Enter house details below:")

# User Inputs
area = st.number_input("Area (sq ft)", min_value=100)

bedrooms = st.number_input("Bedrooms", min_value=1)

bathrooms = st.number_input("Bathrooms", min_value=1)

parking = st.number_input("Parking Spaces", min_value=0)

# Predict Button
if st.button("Predict Price"):

    # Create input array
    input_data = np.array([[area, bedrooms, bathrooms, parking]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)

    # Display Result
    st.success(f"Estimated House Price: ₹ {prediction[0]:,.2f}")
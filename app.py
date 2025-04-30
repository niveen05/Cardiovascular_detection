import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier

# Load the trained model
model = joblib.load("xgb_heart_model.pkl")


# Streamlit UI
st.title("❤️ Heart Disease Predictor")
st.markdown("Enter patient health information to predict the risk of heart disease.")

# Input fields
age = st.slider("Age (in years)", 30, 100, 50)
gender = st.selectbox("Gender", ["Male", "Female"])
height = st.number_input("Height (cm)", 100, 250, 170)
weight = st.number_input("Weight (kg)", 30, 200, 70)
ap_hi = st.number_input("Systolic Blood Pressure (ap_hi)", 80, 250, 120)
ap_lo = st.number_input("Diastolic Blood Pressure (ap_lo)", 40, 200, 80)
cholesterol = st.selectbox("Cholesterol", ["Normal", "Above Normal", "Well Above Normal"])
gluc = st.selectbox("Glucose", ["Normal", "Above Normal", "Well Above Normal"])
smoke = st.selectbox("Smoking", ["No", "Yes"])
alco = st.selectbox("Alcohol Intake", ["No", "Yes"])
active = st.selectbox("Physically Active", ["Yes", "No"])

# Convert categorical values to numeric
gender = 1 if gender == "Male" else 2
cholesterol = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}[cholesterol]
gluc = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}[gluc]
smoke = 1 if smoke == "Yes" else 0
alco = 1 if alco == "Yes" else 0
active = 1 if active == "Yes" else 0

# Derived features
bmi = weight / ((height / 100) ** 2)
pulse_pressure = ap_hi - ap_lo
age_group = pd.cut([age], bins=[0, 40, 50, 60, 70, 100], labels=[0, 1, 2, 3, 4]).astype(int)[0]

# Prepare input
features = pd.DataFrame([[
    age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc,
    smoke, alco, active, bmi, pulse_pressure, age_group
]], columns=[
    'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc',
    'smoke', 'alco', 'active', 'bmi', 'pulse_pressure', 'age_group'
])

# Predict
if st.button("Predict"):
    prediction = model.predict(features)[0]
    st.subheader("Result:")
    if prediction == 1:
        st.error("⚠️ The patient is likely to have heart disease.")
    else:
        st.success("✅ The patient is unlikely to have heart disease.")

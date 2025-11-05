import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load('knn_model.pkl')
scaler = joblib.load('scaler.pkl')

# App title
st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details to predict the likelihood of heart disease.")

# Collect input data
age = st.number_input("Age", 10, 120)
sex = st.selectbox("Sex", ["Male", "Female"])
chest_pain = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
resting_bp = st.number_input("Resting Blood Pressure", 80, 200)
cholesterol = st.number_input("Cholesterol", 100, 600)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
rest_ecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2])
max_hr = st.number_input("Max Heart Rate Achieved", 60, 220)
exercise_angina = st.selectbox("Exercise Induced Angina (0=No, 1=Yes)", [0, 1])
oldpeak = st.number_input("Oldpeak (ST Depression)", 0.0, 10.0, step=0.1)
st_slope = st.selectbox("ST Slope (0-2)", [0, 1, 2])

# Prepare data
data = {
    "AGE": [age],
    "SEX_MALE": [1 if sex == "Male" else 0],
    "CHESTPAIN": [chest_pain],
    "RESTINGBP": [resting_bp],
    "CHOLESTEROL": [cholesterol],
    "FASTINGBS": [fasting_bs],
    "RESTECG": [rest_ecg],
    "MAXHR": [max_hr],
    "EXERCISEANGINA": [exercise_angina],
    "OLDPEAK": [oldpeak],
    "ST_SLOPE": [st_slope]
}

input_df = pd.DataFrame(data)

# Scale input
scaled_input = scaler.transform(input_df)

# Prediction
if st.button("Predict"):
    pred = model.predict(scaled_input)[0]
    if pred == 1:
        st.error("⚠️ The patient is likely to have heart disease.")
    else:
        st.success("✅ The patient is unlikely to have heart disease.")

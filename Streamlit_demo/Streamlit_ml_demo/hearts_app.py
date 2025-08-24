import streamlit as st
import pickle
import numpy as np

# Load the saved model
filename = 'Streamlit_demo\Streamlit_ml_demo\Heart_anish.pkl'
model = pickle.load(open(filename, 'rb'))

# Define the Streamlit app
st.title('Heart Disease Prediction')

st.write('Enter the following details to predict the presence of heart disease:')

# --- Collect user inputs ---
age = st.number_input("Age", min_value=1, max_value=120, value=30)
sex = st.selectbox("Sex", ("Male", "Female"))
cp_value = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
chol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
fbs_value = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg_value = st.selectbox("Resting ECG", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate", min_value=60, max_value=220, value=150)
exang_value = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, step=0.1)
slope_value = st.selectbox("Slope", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
thal_value = st.selectbox("Thal", [0, 1, 2, 3])

# --- Encode categorical inputs ---
sex = 1 if sex == "Male" else 0

# --- Create feature array ---
features = np.array([[age, sex, cp_value, trestbps, chol,
                      fbs_value, restecg_value, thalach,
                      exang_value, oldpeak, slope_value,
                      ca, thal_value]])
prediction = model.predict(features)
proba = model.predict_proba(features)[:, 1]

if prediction[0] == 1:
    st.error(f"⚠️ High Risk of Heart Disease! (Probability: {proba[0]:.2f})")
else:
    st.success(f"✅ Low Risk of Heart Disease. (Probability: {proba[0]:.2f})")
import streamlit as st

st.write('# BMI Calculator')

# Input fields for weight and height
weight = st.number_input('Enter your weight in kilograms')
height = st.number_input('Enter your height in meters')
# Calculate BMI
if height > 0:
    bmi = weight / (height ** 2)
    st.write('### Your BMI:')
    st.write(f'**BMI:** {bmi:.2f}')
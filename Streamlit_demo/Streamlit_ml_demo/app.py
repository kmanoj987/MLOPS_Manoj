import streamlit as st
import pickle
import numpy as np

# Load the saved model
filename = 'random_forest_model.pkl'
loaded_model = pickle.load(open(filename, 'rb'))

# Define the Streamlit app
st.title('Iris Flower Classification')

st.write('Enter the measurements of the iris flower to predict its species.')

# Get user input for the four features
sepal_length = st.slider('Sepal Length (cm)', 0.0, 10.0, 5.0)
sepal_width = st.slider('Sepal Width (cm)', 0.0, 10.0, 3.5)
petal_length = st.slider('Petal Length (cm)', 0.0, 10.0, 1.5)
petal_width = st.slider('Petal Width (cm)', 0.0, 10.0, 0.2)

# Create a numpy array from the input
features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Make a prediction
prediction = loaded_model.predict(features)

# Map the prediction to the species name
species_mapping = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
predicted_species = species_mapping[prediction[0]]

# Display the prediction
st.write(f'The predicted species is: **{predicted_species}**')

# To run this Streamlit app in Colab, you would typically save this code
# as a Python file (e.g., `app.py`) and then run it from the terminal
# using `streamlit run app.py`. However, running Streamlit directly in Colab
# requires some additional setup (like using `ngrok` or similar).
# For a simple demonstration, you could run it locally after copying the code
# or explore third-party solutions for deploying Streamlit apps from Colab.

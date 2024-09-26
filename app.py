import streamlit as st
import joblib

# Load the model
model = joblib.load('gbc.pkl')

# Function for making prediction
def prediction(input_data):
    predictions = model.predict(input_data)   # Make predictions
    return prediction

# Add a side bar
def add_sidebar():
    st.sidebar.header("User Input")

    # Select gender
    gender = st.sidebar.selectbox("Gender", ['Female', 'Male', 'Other'])

    # Select smoking History
    smoking_history = st.sidebar.selectbox("Smoking history", ['Current', 'Former', 'Never', 'No info'])

    # Select Age
    age = st.sidebar.number_input("Age")

    # Select hypertension
    hypertension = st.sidebar.selectbox("Hypertension", ['Yes', 'No'])

    # Select heartdisease
    heart_disease = st.sidebar.selectbox("Heart Disease", ['yes', 'No'])

    # Select BMI
    bmi = st.sidebar.number_input("BMI", min_value= 0.0, max_value= 100.0, step = 0.1)

    # Select HbA1c_level
    hb = st.sidebar.number_input("HbA1c", min_value= 0.0, max_value= 15.0, step = 0.1)

    # Glucose level
    glucose_level = st.sidebar.number_input("Glucose Level", min_value= 80, max_value= 300, step= 1)

st.set_page_config(
    layout= "wide" ,
    initial_sidebar_state= "expanded"
)

add_sidebar()

# Create a container
with st.container():
    # Set app title
    st.title("Diabetes Diagnosis App")
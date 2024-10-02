import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the saved model
model = joblib.load('best_model.pkl')

# Load the saved scaler
scaler = joblib.load('scaler.pkl')

# import the dataset
data = pd.read_csv('clean_diabetes_data.csv')

# Add a side bar
def add_sidebar():
    st.sidebar.header("User Input")

    # Select gender
    gender = st.sidebar.selectbox("Gender", ['Female', 'Male', 'Other'])

    # Encode gender
    if gender == "Female":
        gender = 0
    elif gender == "Male":
        gender = 1
    else:
        gender = 2

    # Select Age
    age = st.sidebar.slider("Age",
                            min_value= float(data['age'].min()),
                            max_value= float(data['age'].max()),
                            value= float(data['age'].mean()))

    # Select hypertension
    hypertension = st.sidebar.selectbox("Hypertension", ['Yes', 'No'])

    # Encode hypertension
    if hypertension == "Yes":
        hypertension = 1
    else:
        hypertension = 0

    # Select heartdisease
    heart_disease = st.sidebar.selectbox("Heart Disease", ['Yes', 'No'])

    # Encode heart_disease
    if heart_disease == "Yes":
        heart_disease = 1
    else:
        heart_disease = 0


    # Select smoking History
    smoking_history = st.sidebar.selectbox("Smoking history", ['Current', 'Former', 'Never', 'No info'])

    # Encode Smoking history
    if smoking_history == "Current":
        smoking_history = 1
    elif smoking_history == "Former":
        smoking_history = 4
    elif smoking_history == "Never":
        smoking_history = 3
    else:
        smoking_history = 0

    # Select BMI
    bmi = st.sidebar.slider("BMI",
                            min_value= float(data['bmi'].min()),
                            max_value= float(data['bmi'].max()),
                            value= float(data['bmi'].mean()))

    # Select HbA1c_level
    HbA1c_level = st.sidebar.slider("HbA1c",
                            min_value= float(data['HbA1c_level'].min()),
                            max_value= float(data['HbA1c_level'].max()),
                            value= float(data['HbA1c_level'].mean()))

    # Glucose level
    blood_glucose_level = st.sidebar.slider("Glucose Level",
                            min_value= float(data['blood_glucose_level'].min()),
                            max_value= float(data['blood_glucose_level'].max()),
                            value= float(data['blood_glucose_level'].mean()))

    # Create a dictionary to store inputs
    input_dict = {
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease":heart_disease,
        "smoking_history": smoking_history,
        "bmi": bmi,
        "HbA1c_level": HbA1c_level,
        "blood_glucose_level": blood_glucose_level
    }

    return input_dict

st.set_page_config(
    page_title= "Diabetes prediction app",
    layout= "wide" ,
    initial_sidebar_state= "expanded"
)

# Hide overflow in the main container
st.html("<style> .main{overflow: hidden} </style>")

# Create a container
with st.container():
    # Set app title
    st.title("Diabetes Diagnosis App:stethoscope:")
    message1 = """
    Diabetes affects over 422 million people globally and is linked to nearly 1.5 million deaths annually, primarily due to complications like heart disease and kidney failure.</br>
    Many people remain undiagnosed, increasing their risk of serious health issues.
    Early detection and proper management can greatly reduce these risks, making tools like 
    </br> this app valuable in helping users assess their diabetes risk and take proactive steps toward better health.
    """
    st.markdown(message1, unsafe_allow_html= True)

    message2 = """
    This app can assist</br> medical professionals</br> in making a diagnosis,</br> but should not be used </br>as a substitute for a</br> professional diagnosis.
    """
    col1, col2 = st.columns(2)
    with col1:
        st.image("C:/Users/User/Downloads/img4.jpg", use_column_width= True)

    with col2:
        st.subheader("Patient's Predictions:")
        st.write("The predictions is:")

        # Call input data func
        input_data = add_sidebar()

        # Convert input data into a DataFrame
        input_data = pd.DataFrame([input_data])

        # Scale the input data
        input_data_scaled = scaler.transform(input_data)

        #Make predictions
        predictions = model.predict(input_data_scaled)
      
        if predictions[0] == 0:
            st.markdown(
        "<div style='background-color: green; color: white; display: inline-block; border-radius: 5px; padding: 2px 5px;'>Non-Diabetic</div>",
        unsafe_allow_html=True)

        else:
            st.markdown(
        "<div style='background-color: red; color: white; display: inline-block; border-radius: 5px; padding: 2px 5px;'>Diabetic</div>",
        unsafe_allow_html=True)

        # Return probabilities
        st.write("Probability of being Non-Diabetic:", model.predict_proba(input_data)[0][0])
        st.write("Probability of being Diabetic:", model.predict_proba(input_data)[0][1])
        st.markdown(message2, unsafe_allow_html= True)
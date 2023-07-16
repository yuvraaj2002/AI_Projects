import streamlit as st
import webbrowser
from src.components.Prediction_Pipeline import PredictPipeline
import pandas as pd

# Set up sidebar
st.sidebar.title("Project Description")
st.sidebar.write(
    "The Data Science Job Prediction project aims to predict whether an individual will secure a data science job based on various input parameters. These parameters include the city's development index, gender, relevant experience, enrollment type, education level, major discipline, experience, company size, company type, and training hours. By analyzing these factors, the project predicts whether a person will obtain a data science job."
)


# Create button with link to project's GitHub
github_link = "https://github.com/yuvraaj2002/Machine_Learning_Projects/tree/master/DS_Job_Prediction"
button = st.sidebar.button("Project GitHub")
if button:
    webbrowser.open_new_tab(github_link)


# Main app
st.title("Data Science Job Prediction")
st.write("Answer the following 10 questions mentioned below and click predict button")


# Define the options for the radio buttons
gender_options = ["Male", "Female", "Other"]
experience_options = ["Has relevent experience", "No relevent experience"]
enrolled_university_options = ["no_enrollment", "Full time course", "Part time course"]
education_level_options = [
    "Graduate",
    "Masters",
    "High School",
    "Phd",
    "Primary School",
]
major_discipline_options = [
    "STEM",
    "Business Degree",
    "Arts",
    "Humanities",
    "No Major",
    "Other",
]
company_size_options = [
    "50-99",
    "<10",
    "10000+",
    "5000-9999",
    "1000-4999",
    "10/49",
    "100-500",
    "500-999",
]
company_type_options = [
    "Pvt Ltd",
    "Funded Startup",
    "Early Stage Startup",
    "Other",
    "Public Sector",
    "NGO",
]

city_development_index = st.slider("What is your city's development index?")
gender = st.radio("Select your gender", gender_options)
relevent_experience = st.radio("Select your relevant experience", experience_options)
enrolled_university = st.radio(
    "Select your enrollment type", enrolled_university_options
)
education_level = st.radio("What is your education level", education_level_options)
major_discipline = st.radio(
    "What is/was your major discipline?", major_discipline_options
)
experience = st.slider("Select experience", min_value=0, max_value=40, step=1)
company_size = st.radio("Select company size", company_size_options)
company_type = st.radio("Select company type", company_type_options)
training_hours = st.slider("Select experience", min_value=0, max_value=20, step=1)

# Add buttons to confirm user input and reset
confirm_button = st.button("Confirm")

if confirm_button:
    user_input = {
        "city_development_index": city_development_index,  # User input for the city or development location
        "gender": gender,  # User input for gender
        "relevent_experience": relevent_experience,  # User input for relevant experience
        "enrolled_university": enrolled_university,  # User input for enrolled university status
        "education_level": education_level,  # User input for education level
        "major_discipline": major_discipline,  # User input for major discipline
        "experience": experience,  # User input for years of experience
        "company_size": company_size,  # User input for company size
        "company_type": company_type,  # User input for company type
        "training_hours": training_hours,  # User input for training hours
    }
    model_features = pd.DataFrame([user_input])
    predict_obj = PredictPipeline()
    predicted_output = predict_obj.Make_prediction(model_features)
    if predicted_output == 0:
        st.write("No Job")
    else:
        st.write("Get a Job")




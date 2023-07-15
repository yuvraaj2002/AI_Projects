import streamlit as st
import webbrowser

# Set up sidebar
st.sidebar.title("Project Description")
st.sidebar.write("The Data Science Job Prediction project aims to predict whether an individual will secure a data science job based on various input parameters. These parameters include the city's development index, gender, relevant experience, enrollment type, education level, major discipline, experience, company size, company type, and training hours. By analyzing these factors, the project predicts whether a person will obtain a data science job.")


# Create button with link to project's GitHub
github_link = 'https://github.com/yuvraaj2002/Machine_Learning_Projects/tree/master/DS_Job_Prediction'

button = st.sidebar.button('Project GitHub')
if button:
    webbrowser.open_new_tab(github_link)


# Main app
st.title("Data Science Job Prediction")
st.write("Answer the following 10 questions mentioned below and click predict button")


# Define the options for the radio buttons
gender_options = ['Male', 'Female', 'Other']
experience_options = ['Has relevent experience', 'No relevent experience']
enrolled_university_options = ['no_enrollment', 'Full time course', 'Part time course']
education_level_options = ['Graduate', 'Masters', 'High School', 'Phd', 'Primary School']
major_discipline_options = ['STEM', 'Business Degree', 'Arts', 'Humanities', 'No Major', 'Other']
company_size_options = ['50-99', '<10', '10000+', '5000-9999', '1000-4999', '10/49', '100-500', '500-999']
company_type_options = ['Pvt Ltd', 'Funded Startup', 'Early Stage Startup', 'Other', 'Public Sector', 'NGO']

city_dev = st.slider("What is your city's development index?")
gender = st.radio("Select your gender", gender_options)
relevent_experience = st.radio("Select your relevant experience", experience_options)
enrolled_university = st.radio("Select your enrollment type", enrolled_university_options)
education_level = st.radio("What is your education level", education_level_options)
major_discipline = st.radio("What is/was your major discipline?", major_discipline_options)
experience = st.number_input("Enter your experience")
company_size = st.radio("Select company size", company_size_options)
company_type = st.radio("Select company type", company_type_options)
training_hours = st.number_input("Enter training hours")

# Add buttons to confirm user input and reset
confirm_button = st.button("Confirm")

if confirm_button:
    user_input = {
        "city_dev": city_dev,
        "gender": gender,
        "relevent_experience": relevent_experience,
        "enrolled_university": enrolled_university,
        "education_level": education_level,
        "major_discipline": major_discipline,
        "experience": experience,
        "company_size": company_size,
        "company_type": company_type,
        "training_hours": training_hours,
    }

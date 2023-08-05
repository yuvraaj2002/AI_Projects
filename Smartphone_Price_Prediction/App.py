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
st.title("Predict smartphone price")
st.write("Enter the requirements you are looking for and press predict")


# Defining the options
rating_options = ['8+', '7+', '6+']
ram_options = [8.0, 6.0, 4.0, 12.0, 3.0, 2.0, 16.0, 1.0, 18.0]
storage_options = [128.0, 64.0, 256.0, 32.0, 512.0, 16.0, 1.0]
charging_options = ['0-50 Watts','50-100 Watts','More than 100 Watts']
screen_rr_options  =[120.0, 90.0, 144.0, 165.0, 240.0]
rear_cams_options = [3.0, 2.0, 4.0, 1.0]

# Define the range for the slider for PPI
min_value_ppi = 28444.0
max_value_ppi = 971224.0

# Range of values for the front camera pixels
min_value_fmp = 0.3
max_value_fmp = 60.0


rating = st.radio('Choose rating',rating_options)

Has_5g = st.checkbox("Select if you want 5g")
Has_5g = 1 if Has_5g else 0

Add_Features = st.checkbox("Select if you want additional features")
Add_Features = 1 if Add_Features else 0

RAM = st.radio('Select RAM you are looking for',ram_options)
Storage = st.radio('Select Storage you are looking for',storage_options)

Charging = st.radio('Select charging watts you are looking for',charging_options)
if Charging == '0-50 Watts':
    Charging = 1.0
elif Charging == '50-100 Watts':
    Charging = 2.0
elif Charging == 'More than 100 Watts':
    Charging = 3.0

Screen_RR = st.radio('Select Screen Resolution',screen_rr_options)
PPI = st.slider("Select a PPI value:", min_value_ppi, max_value_ppi, step=0.1)
rear_cams = st.radio('Select total rear cams you are looking for',rear_cams_options)
Total_fmp = st.slider('Select value for Front camera Mega Pixels',min_value_fmp, max_value_fmp, step=0.1)


# Add buttons to confirm user input and reset
confirm_button = st.button("Confirm")

if confirm_button: # If predict button is pressed we will create a dictionary of user inputs

    user_input = {
        "rating": rating,
        "Has_5g": Has_5g,
        "Add_Features": Add_Features,
        "RAM": RAM,
        "Storage": Storage,
        "Charging": Charging,
        "Screen_RR":Screen_RR,
        "PPI":PPI,
        "rear_cams":rear_cams,
        "Total_fmp":Total_fmp,
        }
    model_features = pd.DataFrame([user_input])

    # Instantiating a PredictPipeline object
    predict_obj = PredictPipeline()
    prediction = predict_obj.Make_prediction(model_features)
    st.write("You can expect Rs",int(prediction[0]))




import streamlit as st
import pandas as pd
from scipy.stats import yeojohnson

def process_data(Input_df):

    # Encoding the course title
    Input_df['course_title'] = Input_df['course_title'].map({"Yes it does": 1.0, "No it doesn't": 0.0})

    # Transforming the values using Yeojohnson
    lambda_values = [-0.05, -0.19, -0.63]
    Input_df['num_reviews'] = yeojohnson(Input_df['num_reviews'], -0.05)
    Input_df['num_lectures'] = yeojohnson(Input_df['num_lectures'], -0.19)
    Input_df['content_duration'] = yeojohnson(Input_df['content_duration'], -0.63)

    # Mapping the paid feature to 0 or 1
    Input_df['is_paid_False'] = Input_df['is_paid_False'].map({'Yes course will be paid': 0, 'No it will be free': 1})

    # One hot encoding of the subject feature
    columns_to_encode = ['subject']
    subject = Input_df.iloc[0]['subject']
    if subject == 'Business Finance':
        Input_df['subject_Business Finance'] = 1
        Input_df['subject_Web Development'] = 0
        Input_df['subject_creativity'] = 0
    elif subject == 'Web Development':
        Input_df['subject_Business Finance'] = 0
        Input_df['subject_Web Development'] = 1
        Input_df['subject_creativity'] = 0
    else:
        Input_df['subject_Business Finance'] = 0
        Input_df['subject_Web Development'] = 0
        Input_df['subject_creativity'] = 1

    Input_df.drop(['subject'], axis=1, inplace=True)
    return Input_df.values



def prediction_module():
    st.markdown("<h1 style='text-align: center;font-size: 60px;'>Prediction Module</h1>", unsafe_allow_html=True)
    st.markdown("***")

    prediction_col1, prediction_col2 = st.columns([0.8, 2], gap="large")

    with prediction_col1:
        st.markdown("<h3>How to get prediction 🤷‍♂️</h3>", unsafe_allow_html=True)
        st.markdown("<p style='background-color: #C5FFF8; padding: 20px; border-radius: 10px; font-size: 20px;'>Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has ss with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum</p>", unsafe_allow_html=True)

        prediction_bt = st.button("Predict", use_container_width=True)
        if prediction_bt:
            model_output = st.toggle("Raw Model Output")
            if model_output:
                st.write("34")

    with prediction_col2:
        num_lectures = st.number_input("How many number of lectures will be there in your course ?",min_value=1, max_value=400,step=1,value=40)
        content_duration = st.slider("What would be the duration of the content (HH.MM)",min_value=0.1, max_value=50.0,step=0.1,value=3.5)
        is_paid_False = st.selectbox("Will your course be paid?", ("Yes course will be paid", "No it will be free"), index=None, placeholder="Select contact method...")
        num_reviews = st.slider("Based on the quality of you course how many number of people you expect would give reviews? ",min_value=0, max_value=2800,step=1,value=95)
        course_title = st.selectbox("Does your course title contains any of these keywords (Beginner, Learn, Course)?", ("Yes it does", "No it doesn't"), index=None, placeholder="Select contact method...")
        subject = st.selectbox("What is the main theme of your course", ("Business Finance", "Web Development", "creativity"), index=None, placeholder="Select contact method...")

        Input_dict = {
            'course_title': [course_title],
            'num_reviews': [num_reviews],
            'num_lectures': [num_lectures],
            'content_duration': [content_duration],
            'is_paid_False': [is_paid_False],
            'subject': [subject]
        }
        Input_df = pd.DataFrame(Input_dict)

        # Calling the function to get processed data
        Input_arr = process_data(Input_df)

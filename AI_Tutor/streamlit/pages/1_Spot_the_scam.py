import streamlit as st
import pandas as pd
import plotly.express as px
import xgboost as xgb
import tensorflow as tf
import pickle

st.markdown(
    """
        <style>
               .block-container {
                    padding-top: 0.0rem;
                    padding-bottom: 0rem;
                    # padding-left: 2rem;
                    # padding-right:2rem;
                }
        </style>
        """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model_xgb():
    xgb_model = xgb.Booster(model_file="artifacts/xgboost_classifier_model.bin")
    return xgb_model


@st.cache_resource
def load_model_blstm():
    blstm_model = tf.keras.models.load_model("artifacts/model.keras")
    return blstm_model


@st.cache_resource
def load_encodings_pipeline():

    with open('your_model_file.pkl', 'rb') as f:
        oe_edu = pickle.load(f)

    return []


def non_text_feature(
    telecommuting,
    has_company_logo,
    has_questions,
    department_mentioned,
    Salary_range_provided,
    employment_type,
    required_experience,
    required_education,
    industry,
):

    # Loading the encodings and pipeline
    oe_edu, oe_emptype, oe_exp, te_industry, Scaling_pipeline = (
        load_encodings_pipeline()
    )

    # Loading the classifier
    classifier1 = load_model_xgb()
    model1_output = classifier1.predict()
    return model1_output


def text_feature():
    pass


def spot_scam_page():
    st.markdown(
        "<h1 style='text-align: center; font-size: 50px;'>Spot the Scam🕵️</h1>",
        unsafe_allow_html=True,
    )
    # st.write("adsfsa jjajdsjfk sadf jlskdjfklsaj lkfsldkjglsk dg jsj dlkgjsd lgs")
    st.markdown(
        "<p style='font-size: 20px; text-align: center;padding-left: 2rem;padding-right: 2rem;padding-bottom: 1rem;'>In times of tough market situations, fake job postings and scams often spike, posing a significant threat to job seekers. To combat this, I've developed a user-friendly module designed to protect individuals from falling prey to such fraudulent activities. This module requires users to input details about the job posting they're considering. Behind the scenes, two powerful AI models thoroughly analyze the provided information. Once completed, users receive a clear indication of whether the job posting is genuine or potentially deceptive.</p>",
        unsafe_allow_html=True,
    )

    configuration_col, input_col = st.columns(spec=(0.8, 2), gap="large")
    with configuration_col:
        st.markdown(
            "<p class='center' style='font-size: 18px; background-color: #CEFCBA; padding:1rem;'>To obtain predictions regarding the current state of the plant, you need to upload the image below. This image should ideally capture the entire plant, ensuring clar.</p>",
            unsafe_allow_html=True,
        )
        model_output_wt = st.slider(
            label="",
            min_value=1,
            max_value=100,
            value=30,
            step=1,
            key="facilities_recommendation_wt",
            label_visibility="collapsed",
        )
        data = {
            "Categories": ["Model1", "Model2"],
            "Weights": [
                model_output_wt,
                100 - model_output_wt,
            ],
        }

        # Create a DataFrame from the data
        df = pd.DataFrame(data)
        custom_colors = ["#AEF359", "#03C04A"]

        # Create a dynamic pie chart using Plotly Express
        fig = px.pie(
            df,
            names="Categories",
            values="Weights",
            color_discrete_sequence=custom_colors,
            height=380,  # Adjust the height as per your requirement
            width=380,  # Adjust the width as per your requirement
        )
        st.plotly_chart(fig, use_container_width=True)

    with input_col:

        input_col1, input_col2 = st.columns(spec=(1, 1), gap="large")
        with input_col1:
            jd = st.text_input("Enter the job description")
            jad = st.text_input("Enter the job dadfsdescription")
            jgd = st.text_input("Enter the job sadescription")

            telecommuting = st.selectbox(
                "How would you like to be contacted?", ("Yes", "No")
            )
            has_company_logo = st.selectbox(
                "How wouldadsf you like to be contacted?", ("Yes", "No")
            )
            has_questions = st.selectbox(
                "How would youadf like to be contacted?", ("Yes", "No")
            )

        with input_col2:
            department_mentioned = st.selectbox(
                "How wouldadsf you like to be adfsfcontacted?", ("Yes", "No")
            )
            Salary_range_provided = st.selectbox(
                "How would you like to be conassagtacted?", ("Yes", "No")
            )
            employment_type = st.selectbox(
                "How would you like to bes contacted?", ("Yes", "No")
            )
            required_experience = st.selectbox(
                "How wouldadsf you like asdfsto be contacted?", ("Yes", "No")
            )
            required_education = st.selectbox(
                "How would qyouadf like to be contacted?", ("Yes", "No")
            )
            industry = st.selectbox(
                "How wouldadsf you like toga be adfsfcontacted?", ("Yes", "No")
            )

            predict_bt = st.button("Predict")
            if predict_bt:

                model1_output = non_text_feature(
                    telecommuting,
                    has_company_logo,
                    has_questions,
                    department_mentioned,
                    Salary_range_provided,
                    employment_type,
                    required_experience,
                    required_education,
                    industry,
                )
                model2_output = text_feature()


spot_scam_page()

import streamlit as st
import pandas as pd
import plotly.express as px
import pickle


@st.cache_resource
def load_pipeline():
    # Load the pipeline from the pickle file
    with open(
        "/home/yuvraj/Documents/AI/AI_Projects/Find_Home.AI/Artifacts/Loan_Pipeline.pkl",
        "rb",
    ) as file:
        loan_pipeline = pickle.load(file)
        return loan_pipeline


@st.cache_resource
def load_model():
    # Load the pipeline from the pickle file
    with open(
        "/home/yuvraj/Documents/AI/AI_Projects/Find_Home.AI/Artifacts/SVC.pkl", "rb"
    ) as file:
        loan_model = pickle.load(file)
        return loan_model


Credit_History_options = {"No": 0.0, "Yes": 1.0}


def create_dataframe(
    Education="Graduate",
    Self_Employed="Yes",
    Dependents="0",
    Loan_Amount_Term=2,
    Gender="Male",
    Married="Yes",
    Property_Area="Urban",
    CoapplicantIncome=50.0,
    LoanAmount=125.0,
    ApplicantIncome=125.0,
    Credit_History="Yes",
):
    """
    This method will take the input variables and will return a DataFrame with input feature values.
    :return: DataFrame
    """
    # Create a dictionary with your input variables
    data = {
        "Education": [Education],
        "Self_Employed": [Self_Employed],
        "Dependents": [Dependents],
        "Loan_Amount_Term": [float(Loan_Amount_Term) * 12.0] if Loan_Amount_Term is not None else [None],
        "Gender": [Gender],
        "Married": [Married],
        "Property_Area": Property_Area,
        "CoapplicantIncome": [CoapplicantIncome],
        "LoanAmount": [LoanAmount],
        "ApplicantIncome": [ApplicantIncome],
        "Credit_History": [Credit_History_options[Credit_History]] if Credit_History is not None else [None],
    }

    # Create a DataFrame from the dictionary
    load_df = pd.DataFrame(data)
    return load_df


def process_data(Loan_Input_df):
    """
    This method will take the input dataframe and return a 2d numpy array (Processed data) for making prediction
    :return:
    """

    # Loading the pipeline and model
    loan_pipeline = load_pipeline()

    # Processing the data and predicting the output
    Loan_Input = loan_pipeline.transform(Loan_Input_df)
    return Loan_Input


def predict_value(Loan_Input):
    model = load_model()
    return model.predict(Loan_Input)


def load_eligibility_UI():
    st.markdown(
        "<h1 style='text-align: center; font-size: 48px; '>Loan Eligibility Module 🏠</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='font-size: 19px; text-align: center;padding-left: 2rem;padding-right: 2rem;'>Welcome to our HomeLoan Assurance Advisor, a sophisticated module designed to provide you with invaluable insights into your eligibility for a home loan. Deciding to purchase a property is a significant step, and we understand the importance of financial clarity in this decision-making process.This tool is especially beneficial for those who may be uncertain about their eligibility or wish to assess their loan approval chances before committing to a property investment.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("***")

    page_col1, page_col2 = st.columns(spec=(1, 1.4), gap="large")
    with page_col1:
        st.markdown(
            "<p style='background-color: #CEFCBA; padding: 1rem; border-radius: 10px; font-size: 18px;'> 📌 Wondering how each aspect of your profile influences your home loan eligibility? Explore the Feature Contribution Visualization to understand how each aspect of your profile, such as education level, employment status, dependents, and more, influences your loan eligibility predicted by our advanced machine learning model.</p>",
            unsafe_allow_html=True,
        )

        # Create a sample dataframe
        data = {
            "Feature": [
                "Feature1",
                "Feature2",
                "Feature3",
                "Feature4",
                "Feature5",
                "Feature6",
                "Feature7",
                "Feature8",
                "Feature9",
                "Feature10",
            ],
            "Value": [
                0.01818,
                0.04545,
                0.05455,
                0.06364,
                0.02727,
                0.03636,
                0.07273,
                0.1,
                0.10909,
                0.08182,
            ],
        }
        df = pd.DataFrame(data)
        fig = px.bar(
            df, x="Feature", y="Value", labels={"Value": "Y Axis Value"}, height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    with page_col2:
        input_col1, input_col2 = st.columns(spec=(1, 1), gap="large")

        with input_col1:
            Education = st.selectbox(
                "Select Education Level",
                ("Graduate", "Not Graduate"),
                index=None,
                placeholder="Example : Graduate",
            )
            Self_Employed = st.selectbox(
                "Are you self-employed",
                ("Yes", "No"),
                index=None,
                placeholder="Example : Yes",
            )
            Dependents = st.selectbox(
                "Select total number of dependents",
                ("0", "1", "2", "3+"),
                index=None,
                placeholder="Example : 3+",
            )
            Married = st.selectbox(
                "Are you married or not",
                ("Yes", "No"),
                index=None,
                placeholder="Example : Yes",
            )
            Loan_Amount_Term = st.slider(
                "Enter Loan amount term (Years)",
                min_value=1,
                max_value=40,
                value=2,
                step=1,
            )
            LoanAmount = st.slider(
                "Enter Loan amount", min_value=10, max_value=500, value=125
            )

        with input_col2:
            Property_Area = st.selectbox(
                "Select Location Type",
                ("Semiurban", "Urban", "Rural"),
                index=None,
                placeholder="Example : Urban",
            )
            CoapplicantIncome = st.slider(
                "Enter Co-Applicant income", min_value=0.0, max_value=9000.0, value=50.0
            )
            ApplicantIncome = st.slider(
                "Enter Applicant income", min_value=150, max_value=18500, value=125
            )
            Credit_History = st.selectbox(
                "Do you have credit history",
                ("Yes", "No"),
                index=None,
                placeholder="Example : Yes",
            )

            loan_eligibility_bt = st.button(
                "Predict loan eligibility", use_container_width=True
            )
            if loan_eligibility_bt:

                # Check if any field is empty
                if any(
                        [
                            Education == None,
                            Self_Employed == None,
                            Dependents == None,
                            Married == None,
                            Loan_Amount_Term == None,
                            LoanAmount == None,
                            Property_Area == None,
                            CoapplicantIncome == None,
                            ApplicantIncome == None,
                            Credit_History == None,
                        ]
                ):
                    st.error("Please fill in all the values.")
                else:

                    # Calling the function to create dataframe and process it using pre-processing pipeline
                    Loan_Input_df = create_dataframe(
                        Education,
                        Self_Employed,
                        Dependents,
                        Loan_Amount_Term,
                        "Male",
                        Married,
                        Property_Area,
                        CoapplicantIncome,
                        LoanAmount,
                        ApplicantIncome,
                        Credit_History,
                    )
                    Loan_Input = process_data(Loan_Input_df)
                    predicted_value  = predict_value(Loan_Input)
                    if predicted_value == 1.0:
                        st.success("Your loan will be approved!", icon="✅")
                    elif predicted_value == 1.0:
                        st.error("This is a failure message!", icon="❌")

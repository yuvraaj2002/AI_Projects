import streamlit as st
import pandas as pd
import plotly.express as px


def load_eligibility_UI():

    st.markdown("<h1 style='text-align: center; font-size: 50px; '>Loan Eligibility Module 🏠</h1>",
                unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size: 20px; text-align: center;padding-left: 2rem;padding-right: 2rem;'>Welcome to our HomeLoan Assurance Advisor, a sophisticated module designed to provide you with invaluable insights into your eligibility for a home loan. Deciding to purchase a property is a significant step, and we understand the importance of financial clarity in this decision-making process.This tool is especially beneficial for those who may be uncertain about their eligibility or wish to assess their loan approval chances before committing to a property investment.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("***")

    page_col1, page_col2 = st.columns(spec=(1, 1.4), gap="large")
    with page_col1:
        st.markdown(
            "<p style='background-color: #CEFCBA; padding: 1rem; border-radius: 10px; font-size: 18px;'> 📌 Wondering how each aspect of your profile influences your home loan eligibility? Explore the Feature Contribution Visualization to understand how each aspect of your profile, such as education level, employment status, dependents, and more, influences your loan eligibility predicted by our advanced machine learning model.</p>",
            unsafe_allow_html=True)

        # Create a sample dataframe
        data = {
            'Feature': ['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5', 'Feature6', 'Feature7', 'Feature8',
                        'Feature9', 'Feature10', 'Feature11'],
            'Value': [0.01818, 0.04545, 0.05455, 0.06364,0.02727, 0.03636,0.07273, 0.1, 0.10909,0.08182, 0.09091]
        }
        df = pd.DataFrame(data)
        fig = px.bar(df, x='Feature', y='Value',labels={'Value': 'Y Axis Value'},height=400)
        st.plotly_chart(fig, use_container_width=True)


    with page_col2:
        input_col1, input_col2 = st.columns(spec=(1, 1), gap="large")
        with input_col1:
            education_category = st.selectbox("Select Education Level", ('Graduate', 'Not Graduate'), index=None,
                                              placeholder="Example : Graduate")
            self_emp_category = st.selectbox("Are you self employed", ('Yes', 'No'), index=None,
                                             placeholder="Example : Yes")
            dependents = st.selectbox("Select total number of dependents", ('0', '1', '2', '3+'), index=None,
                                      placeholder="Example : 3+")
            gender = st.selectbox("Choose your gender", ('Male', 'Female'), index=None, placeholder="Example : Yes")
            married = st.selectbox("Are you married or not", ('Yes', 'No'), index=None, placeholder="Example : Yes")
            loan_amount_term = st.slider("Enter Loan amount term (Years)", min_value=1, max_value=40, value=2, step=1)

        with input_col2:
            area_type = st.selectbox("Select Location Type", ('Semiurban', 'Urban', 'Rural'), index=None,
                                     placeholder="Example : Urban")
            CoapplicantIncome = st.slider("Enter Co-Applicant income", min_value=0.0, max_value=9000.0, value=50.0)
            LoanAmount = st.slider("Enter Co-Applicant income", min_value=10, max_value=500, value=125)
            ApplicantIncome = st.slider("Enter Co-Applicant income", min_value=150, max_value=18500, value=125)
            Credit_History = st.selectbox("Do you have credit history", ('Yes', 'No'), index=None,
                                          placeholder="Example : Yes")

            loan_eligibility_bt = st.button("Predict",use_container_width=True )

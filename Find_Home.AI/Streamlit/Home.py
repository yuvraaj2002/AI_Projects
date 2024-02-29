import streamlit as st
from Pages.Price_Prediction import Price_Prediction_Page
from Pages.Recommendation_System import Recommendation_System_Page
from Pages.Loan_Eligibility import load_eligibility_Page

# Set page configuration
st.set_page_config(
    page_title="FindHome.AI",
    page_icon="🏠",
    layout="wide",
)

st.markdown(
    """
        <style>
               .block-container {
                    padding-top: 3rem;
                    padding-bottom: 0rem;
                    # padding-left: 2rem;
                    # padding-right:2rem;
                }
                .top-margin{
                    margin-top: 4rem;
                }
        </style>
        """,
    unsafe_allow_html=True,
)


# Main page function
def main_page():
    Overview_col, Img_col = st.columns(spec=(1.2, 1), gap="large")

    with Overview_col:

        # Content for main page
        st.markdown(
            "<h1 style='text-align: left; font-size: 70px; '>Home Finder.AI🏡</h1>",
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown(
            "<p style='font-size: 22px; text-align: left;'>The AI-Powered Home Finder is a cutting-edge solution that harnesses the power of advanced AI algorithms to revolutionize the way people search for their dream homes across Gurgaon. This innovative technology brings a host of benefits to prospective homebuyers by seamlessly streamlining the house-hunting process and delivering personalized recommendations tailored to their unique preferences and requirements.</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='font-size: 22px; text-align: left;'>The AI-Powered Home Finder is a cutting-edge solution that harnesses the power of advanced AI algorithms to revolutionize the way people search for their dream homes across Gurgaon. This innovative technology brings a host of benefits to prospectiv.</p>",
            unsafe_allow_html=True,
        )

        st.write("***")
        st.write("Select the module👇")
        col1, col2, col3 = st.columns(spec=(1, 1, 1), gap="large")
        with col1:
            st.markdown(
                "<button style='padding: 10px; width: 100%;background-color: #c4fcce;'>Price Prediction</button>",
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                "<button style='padding: 10px; width: 100%;background-color: #c4fcce;'>Recommendation Engine</button>",
                unsafe_allow_html=True
            )
        with col3:
            st.markdown(
                "<button style='padding: 10px; width: 100%;background-color: #c4fcce;'>Loan Eligibility</button>",
                unsafe_allow_html=True
            )

    with Img_col:
        st.write("")
        st.markdown("<div class='top-margin'> </div>", unsafe_allow_html=True)
        st.image("Artifacts/Main_Img.png")



page_names_to_funcs = {
    "Project Overview": main_page,
    "Property Price Prediction🗃️": Price_Prediction_Page,
    "Recommendation Engine": Recommendation_System_Page,
    "Loan Eligibility Module": load_eligibility_Page,
}
selected_page = st.sidebar.selectbox("Select Module", list(page_names_to_funcs.keys()))
page_names_to_funcs[selected_page]()

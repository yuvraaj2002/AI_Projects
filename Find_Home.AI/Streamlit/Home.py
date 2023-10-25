import streamlit as st
from Pages.Analytics import Analytics_Page
from Pages.Price_Prediction import Price_Prediction_Page

st.set_page_config(
    page_title = "FindHome.AI",
    page_icon = "🏠",
    layout="wide"
)


def main_page():
    col1, col2 = st.columns(spec=(1.2, 1), gap="small")
    with col1:
        st.markdown("<h1 class='center' style='font-size: 80px;'>Home Finder.AI</h1>", unsafe_allow_html=True)
        st.write("")
        st.markdown(
            "<p class='center' style='font-size: 25px;'>The AI-Powered Home Finder is a cutting-edge solution that harnesses the power of advanced AI algorithms to revolutionize the way people search for their dream homes across Gurgaon. This innovative technology brings a host of benefits to prospective homebuyers by seamlessly streamlining the house-hunting process and delivering personalized recommendations tailored to their unique preferences and requirements.</p>", unsafe_allow_html=True)

        st.markdown("***")
        st.write("Pick the city that'll be home sweet home🏠")
        button_style = """
                <style>
                .stButton > button {
                    color: #31333f;
                    font="sans serif";
                    background: #00d47e;
                    width: 170px;
                    height: 80px;
                }
                </style>
                """
        st.markdown(button_style, unsafe_allow_html=True)
        b1_space, b2_space = st.columns(spec=(1, 1), gap="small")
        with b1_space:
            st.button("**Project Github**")
        with b2_space:
            st.button("**Blog Post about project**")


    with col2:
        st.image('/home/yuvraj/Github/Machine_Learning_Projects/Find_Home.AI/Images/Main_Home.png')






page_names_to_funcs = {
    "Project Overview": main_page,
    "Analytics": Analytics_Page,
    "Price Prediction": Price_Prediction_Page,
}
selected_page = st.sidebar.selectbox("Select Module", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()
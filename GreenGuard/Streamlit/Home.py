import streamlit as st
from Pages.Prediction_Disease import predict

st.set_page_config(
    page_title="GreenGuard.AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
        <style>
               .block-container {
                    padding-top: 2rem;
                    padding-bottom: 0rem;
                }
        </style>
        """,
    unsafe_allow_html=True,
)


def main_page():
    col1, col2 = st.columns(spec=(1, 1), gap="small")
    with col1:
        st.markdown(
            "<h1 class='center' style='font-size: 80px;'>GreenGuard.AI</h1>",
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown(
            "<p class='center' style='font-size: 24px;'>GreenGuard is an innovative image classification project designed to revolutionize agriculture by providing farmers with an automated solution for the early detection of plant diseases. With the aim of mitigating crop losses and optimizing agricultural practices, GreenGuard leverages a convolutional neural network to identify and classify diseased plants into three different categories: healthy, prone to disease, and diseased.</p>",
            unsafe_allow_html=True,
        )

        st.markdown("***")
        button_style = """
                <style>
                .stButton > button {
                    color: #31333f;
                    font="sans serif";
                    width: 150px;
                    height: 50px;
                }
                </style>
                """
        st.markdown(button_style, unsafe_allow_html=True)
        b1_space, b2_space, b3_space = st.columns(spec=(1, 1, 1), gap="small")
        with b1_space:
            st.button("**Project Github**")
            # https://www.kaggle.com/code/yuvikaggle7233831/plant-disease
        with b2_space:
            st.button("**Dataset 📃**")
        with b3_space:
            st.button("**Notebook 👨‍💻**")

        st.markdown("***")
        Intro_text3 = "<p style='font-size: 24px;'>The module for plant disease prediction is available in the drop-down menu at the top left corner of this page. Just select the module, and you will be directed to the prediction model page.</p>"
        st.markdown(Intro_text3, unsafe_allow_html=True)

    with col2:
        st.write("")
        st.image(
            "/home/yuvraj/Github/Deep_Learning_Projects/Plant_Disease/Streamlit/Images/Home.jpg"
        )


page_names_to_funcs = {
    "Project Overview 📑": main_page,
    "Prediction Module 🤔": predict,
}
selected_page = st.sidebar.selectbox("Select Module", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()

##9ef01a;

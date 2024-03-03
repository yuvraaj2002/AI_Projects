import streamlit as st

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
    col1, col2 = st.columns(spec=(1, 1), gap="large")
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
        social_col1, social_col2, social_col3, social_col4 = st.columns(spec=(1, 1, 1, 1), gap="large")
        with social_col1:
            st.link_button("Github👨‍💻", use_container_width=True, url="https://github.com/yuvraaj2002")

        with social_col2:
            st.link_button("Linkedin🧑‍💼", use_container_width=True,
                           url="https://www.linkedin.com/in/yuvraj-singh-a4430a215/")

        with social_col3:
            st.link_button("Twitter🧠", use_container_width=True, url="https://twitter.com/Singh_yuvraaj1")

        with social_col4:
            st.link_button("Blogs✒️", use_container_width=True, url="https://yuvraj01.hashnode.dev/")

        st.markdown("***")
        Intro_text3 = "<p style='font-size: 24px;'>Navigate to the drop-down menu located at the top left corner of this webpage. Within the menu, you will find a module dedicated to plant disease prediction. Simply select this module, and you will seamlessly transition to the prediction model page.</p>"
        st.markdown(Intro_text3, unsafe_allow_html=True)

    with col2:
        st.write("")
        st.image(
            "artifacts/Home.jpg"
        )

main_page()


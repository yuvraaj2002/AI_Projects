import streamlit as st
# from Pages.Price_Prediction import Price_Prediction_Page
# from Pages.Recommendation_System import Recommendation_System_Page
# from Pages.Loan_Eligibility import load_eligibility_Page

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
                    margin-bottom:2rem;
                }
                .block-button{
                    padding: 10px; 
                    width: 100%;
                    background-color: #c4fcce;
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
            "<p style='font-size: 22px; text-align: left;'>The AI-Powered Home Finder is a state-of-the-art tool that transforms the way Gurgaon residents look for their ideal homes by utilizing the strength of sophisticated AI algorithms. Prospective homeowners can take advantage of several advantages provided by this cutting-edge technology, which smoothly streamlines the house-hunting process and provides customized recommendations based on each individual's preferences and needs.There are three separate modules included in this application.</p>",
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown("""
            <div>
                <ul>
                    <li><p style='font-size: 22px; text-align: left;'><em>The Price Prediction Module:</em> Empowered by your input preferences, this module delivers precise estimations of property values, ensuring informed decision-making.</p></li>
                    <li><p style='font-size: 22px; text-align: left;'><em>The Recommendation Engine:</em> Unlocking a realm of personalized suggestions, this engine utilizes advanced algorithms to match your preferences with the perfect property fit.</p></li>
                    <li><p style='font-size: 22px; text-align: left;'><em>The Loan Eligibility Module:</em> Seamlessly integrated, this module provides insights into your eligibility for loans, facilitating smooth financial planning.</p></li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    with Img_col:
        st.write("")
        st.markdown("<div class='top-margin'> </div>", unsafe_allow_html=True)
        st.image("Artifacts/Main_Img.png")
        st.write("")

        social_col1, social_col2, social_col3,social_col4 = st.columns(spec=(1, 1, 1,1), gap="large")
        with social_col1:
            st.link_button("Github👨‍💻", use_container_width=True, url="https://github.com/yuvraaj2002")

        with social_col2:
            st.link_button("Linkedin🧑‍💼",use_container_width=True,url = "https://www.linkedin.com/in/yuvraj-singh-a4430a215/")

        with social_col3:
            st.link_button("Twitter🧠",use_container_width=True,url = "https://twitter.com/Singh_yuvraaj1")

        with social_col4:
            st.link_button("Blogs✒️", use_container_width=True, url="https://yuvraj01.hashnode.dev/")


# page_names_to_funcs = {
#     "Project Overview": main_page,
#     "Property Price Prediction🗃️": Price_Prediction_Page,
#     "Recommendation Engine": Recommendation_System_Page,
#     "Loan Eligibility Module": load_eligibility_Page,
# }
# selected_page = st.sidebar.selectbox("Select Module", list(page_names_to_funcs.keys()))
# page_names_to_funcs[selected_page]()
main_page()
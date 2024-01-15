import streamlit as st
from Pages.Recommendation_Engine import Recommendation_Page
from Pages.Sentiment_Analysis import Sentiment_Analysis_Page

st.set_page_config(
    page_title = "FindHome.AI",
    page_icon = "🏠",
    layout="wide"
)
st.markdown("""
        <style>
               .block-container {
                    padding-top: 3rem;
                    padding-bottom: 0rem;
                    # padding-left: 3rem;
                    # padding-right:3rem;
                }
        </style>
        """, unsafe_allow_html=True)

def main_page():
    col1, col2 = st.columns(spec=(1.7, 1), gap="large")
    with col1:
        st.markdown("<h1 class='center' style='font-size: 75px;'>StyleSphere.AI</h1>", unsafe_allow_html=True)
        st.write("")
        st.markdown(
            "<p class='center' style='font-size: 22px;'>Discover your style effortlessly with our AI-powered fashion app! Upload an image of your desired item, and our advanced recommendation engine crafts a personalized selection of related fashion designs, keeping you on the cutting edge of style. Plus, enjoy an immersive shopping experience with augmented reality features, allowing you to virtually try on selected items before making a purchase.But there's more – we've integrated sentiment analysis! Gauge the crowd's overall sentiment toward your selected products, empowering you to make informed decisions. Elevate your fashion game with ease. Fashion meets technology – discover, decide, and define your style, all in the palm of your hand! Explore the future of fashion today.</p>", unsafe_allow_html=True)

        st.markdown("***")
        st.markdown(
            "<p class='center' style='font-size: 22px;'>I have included the GitHub link for this project along with my social media profiles below. Please feel free to click on the respective buttons, and you will be redirected to the specified pages in a new tab. Additionally, stay connected with the latest project updates and join the community discussions on GitHub. Your feedback is invaluable and contributes to the continuous improvement of this project.</p>",
            unsafe_allow_html=True)

        st.markdown("")
        with st.container():
            b1_space, b2_space,b3_space, b4_space = st.columns(spec=(1, 1,1,1), gap="small")
            with b1_space:
                st.button("**Project Github**")
            with b2_space:
                st.button("**Github Profile**")
            with b3_space:
                st.button("**Linkedin Profile**")
            with b4_space:
                st.button("**My Mail**")


    with col2:
        st.image('/home/yuvraj/Documents/AI/AI_Projects/StyleSphere/Streamlit/Images/Main1.png')






page_names_to_funcs = {
    "Project Overview": main_page,
    "Recommendation Engine": Recommendation_Page,
    "Sentiment analysis": Sentiment_Analysis_Page,
}
selected_page = st.sidebar.selectbox("Select Module", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()
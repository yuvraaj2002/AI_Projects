import streamlit as st


def Sentiment_Analysis_Page():
    st.markdown("<h1 style='text-align: center; font-size: 50px;'>Sentiment Analysis</h1>", unsafe_allow_html=True)

    Analytics_intro = (
        "<p style='font-size: 20px; text-align: center;'>Welcome to a comprehensive exploration of Gurgaon's real estate landscape! Our dedicated module is designed to offer an in-depth understanding of the intricate dynamics that influence property prices in Gurgaon, spanning from independent houses to modern flats. Whether you're a prospective buyer, seller, investor, or industry professional, this module is tailored to provide you with invaluable insights into the factors shaping Gurgaon's property market.epth understanding of the intricate dynamics that influence property prices in Gurgaon, spanning from independent houses to modern flats. Whether you're a prospective buyer, seller, investor, or industry professional, this module is tailored to provide you with invaluable insights into the factors shaping Gurgaon's property </p>")
    st.markdown(Analytics_intro, unsafe_allow_html=True)
    st.markdown("***")

    prediction_col1, prediction_col2 = st.columns([1, 2], gap="large")
    with prediction_col1:
        st.markdown("<h3>How to get prediction 🤷‍♂️</h3>", unsafe_allow_html=True)
        st.markdown(
            "<p style='background-color: #d9b3ff; padding: 20px; border-radius: 10px; font-size: 20px;'>Sentiment analysis offers two methods: 1) Upload a file with plain text reviews, or 2) Copy and paste the Amazon apparel item's URL below. Our system will instantly scrape and analyze all reviews, making sentiment analysis a breeze..</p>",
            unsafe_allow_html=True)

        url_input = st.text_input("Enter the Amazon clothing page URL 🔗", value="Example : https://www.amazon.in/leeve-Printed-Shirts/")
        uploaded_file = st.file_uploader("Upload a plain text file", type=["txt"])

        # prediction_bt = st.button("Predict", use_container_width=True)
        # if prediction_bt:
        #     model_output = st.toggle("Raw Model Output")
        #     if model_output:
        #         st.write("34")

    with prediction_col2:
        pass


#  https://discuss.streamlit.io/t/how-to-give-an-input-option-for-url/7414/7
# https://www.amazon.in/7Shores-Cactus-Sleeve-Printed-Shirts/dp/B0CP7LKNTK/ref=lp_1968093031_1_1_sspa?keywords=Shirts&pf_rd_p=9e034799-55e2-4ab2-b0d0-eb42f95b2d05&pf_rd_r=QZYT7G1PQEF0HF8T5Y7H&sp_csd=d2lkZ2V0TmFtZT1zcF9hcGJfZGVza3RvcF9icm93c2VfaW5saW5lX2F0Zg&th=1&psc=1
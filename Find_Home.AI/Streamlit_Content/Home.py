import streamlit as st
import numpy as np
from PIL import Image
import time

st.set_page_config(layout="wide")

col1,col2 = st.columns(spec=(1.2,1),gap="small")
with col1:
    st.markdown("<h1 class='center' style='font-size: 80px;'>Home Finder.AI</h1>",unsafe_allow_html=True)
    st.write("")
    st.markdown("<p class='center' style='font-size: 25px;'>AI-Powered Home Finder leverages advanced AI algorithms to "
                "swiftly locate ideal homes across India diverse states. It streamlines the house-hunting process, tailoring recommendations to "
                "individual preferences and requirements.</p>",unsafe_allow_html=True)

    st.markdown("***")
    st.write("Pick the city that'll be home sweet home🏠")
    button_style = """
            <style>
            .stButton > button {
                color: #31333f;
                font="sans serif";
                background: #00d47e;
                width: 140px;
                height: 50px;
            }
            </style>
            """
    st.markdown(button_style, unsafe_allow_html=True)
    b1_space, b2_space,b3_space, b4_space = st.columns(spec=(1, 1,1,1), gap="medium")
    with b1_space:
        st.button("**BANGLORE**")
    with b2_space:
        st.button("**DELHI**")
    with b3_space:
        st.button("**GURGAON**")
    with b4_space:
        st.button("**PUNE**")

with col2:
    st.image('Streamlit_Content/Main_Home.png')


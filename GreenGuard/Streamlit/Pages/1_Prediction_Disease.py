import streamlit as st
from PIL import Image
import numpy as np
import plotly.express as px
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import Sequential, layers
from keras.models import load_model
import pickle
import time

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

@st.cache_resource
def load_trained_model():

    # Load the saved model
    model = load_model('artifacts/Trained_Model.h5')
    return model


def make_prediction(img_array, model):

    # Expand the dimensions to make it a batch of size 1
    img_array = tf.expand_dims(img_array, 0)

    # Getting prediction from the model
    predicted_value = model.predict(img_array)
    return predicted_value


def resize_and_rescale_img(image_file):

    # Read the image as a NumPy array from image_file object
    image = Image.open(image_file).convert("RGB")

    # Resize the image to the desired dimensions (256, 256)
    resized_image = image.resize((256, 256))

    # Reading numpy array from image
    raw_img_arr = np.asarray(resized_image)

    # Rescale pixel values to the range [0, 1]
    rescaled_img_arr = raw_img_arr / 255.0

    return raw_img_arr



def predict_page():
    col1, col2, col3 = st.columns(spec=(0.9, 2, 0.5), gap="large")
    with col1:
        pass
        image = Image.open(
            "Artifacts/Model.png"
        )
        image = image.resize((360, 750))
        st.image(image)

    with col2:

        #st.title("Visualizations")
        st.markdown(
            "<p class='center' style='font-size: 22px; background-color: #CEFCBA; padding:1rem;'>To obtain predictions regarding the current state of the plant, you need to upload the image below. This image should ideally capture the entire plant, ensuring clarity and focus. Once you've uploaded the image, our advanced AI algorithms will analyze it meticulously. These algorithms are trained to detect various indicators such as leaf color, texture, size, and overall plant health. Once the analysis is complete, simply click the prediction button, and you'll receive a detailed report outlining the plant's current condition.</p>",
            unsafe_allow_html=True
        )

        #st.markdown("***")

        image_file = st.file_uploader(
            label="Upload the image", type=["jpg", "jpeg", "png"]
        )

        # Creating column for the displaying the image and for showing its properties
        img_col, properties_col = st.columns(spec=(1, 1.5), gap="large")

        with img_col:
            if image_file is not None:
                # Reading and displaying the image
                image = Image.open(image_file).convert("RGB")
                data = np.asarray(image)
                st.image(data, caption="Uploaded Image", use_column_width=True)

        with properties_col:
            if image_file is not None:
                st.markdown(
                    "<p class='center' style='font-size: 30px;'><strong>File Details 📂</strong></p>",
                    unsafe_allow_html=True,
                )
                uploaded_image_arr = Image.open(image_file)
                st.markdown(f"<p style='font-size:18px'><b>Name of File:</b> {image_file.name}</p>",
                            unsafe_allow_html=True)
                st.markdown(f"<p style='font-size:18px'><b>File type:</b> {image_file.type}</p>",
                            unsafe_allow_html=True)
                st.markdown(f"<p style='font-size:18px'><b>File Size:</b> {image_file.size} Bytes</p>",
                            unsafe_allow_html=True)
                st.markdown(f"<p style='font-size:18px'><b>Type of Image:</b> {uploaded_image_arr.mode}</p>",
                            unsafe_allow_html=True)
                st.markdown(f"<p style='font-size:18px'><b>Shape of Image:</b> {uploaded_image_arr.size}</p>",
                            unsafe_allow_html=True)

                button_style = """
                                <style>
                                .stButton > button {
                                    color: #31333f;
                                    font="sans serif";
                                    width: 150px;
                                    height: 35px;
                                }
                                </style>
                                """
                st.markdown(button_style, unsafe_allow_html=True)
                prediction_bt = st.button("Predict👨‍⚕️")
                if prediction_bt:
                    with st.spinner('Analyzing the plant🔎'):
                        img_array = resize_and_rescale_img(image_file)
                        model = load_trained_model()
                        value = make_prediction(img_array, model)
                        idx_value = np.argmax(value)
                        time.sleep(5)  # Simulating the model prediction time
                        if idx_value == 0:
                            st.write(
                                "<p style='font-size: 24px;'>Plant is detected to be <strong>Early Blight😷</strong></p>",
                                unsafe_allow_html=True,
                            )
                        elif idx_value == 1:
                            st.write(
                                "<p style='font-size: 24px;'>Plant is detected to be <strong>Late Blight😷</strong></p>",
                                unsafe_allow_html=True,
                            )
                        else:
                            st.write(
                                "<p style='font-size: 24px;'>Plant is detected to be <strong>Healthy💚</strong></p>",
                                unsafe_allow_html=True,
                            )

    with col3:
        st.title("Model Comparision")
        st.markdown("***")
        st.metric(label="Resnet50",value="96.8%",delta="5.1%")
        st.progress(0.968, text=None)
        #st.markdown("***")
        st.metric(label="VGG16", value="95.2%",delta="3.5%")
        st.progress(0.952, text=None)
        #st.markdown("***")
        st.metric(label="Alexnet", value="92.6%",delta="0.9%")
        st.progress(0.926, text=None)
        #st.markdown("***")
        st.metric(label="Custom CNN", value="91.7%",delta="Baseline")
        st.progress(0.917, text=None)

predict_page()
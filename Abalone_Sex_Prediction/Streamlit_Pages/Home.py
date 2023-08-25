import streamlit as st
import pandas as pd
from Src.Utils import load_object

st.set_page_config(
    page_title="AI APP",
    page_icon="👋",
)

def main():

    st.title("Abalone Sex prediction🐌")
    st.write("The abalone sex prediction app uses machine learning to predict the sex of an abalone based on its physical "
             "measurements. The app is a valuable tool for abalone researchers and aquaculturists, as it can be used to sex "
             "abalone quickly and accurately, without the need for dissection.A short description of the dataset used for training"
             "this project is availabe in the sidebar.")


    Length = st.number_input('Length', 0.0, 1.0, key='length_key')
    Diameter = st.number_input('Diameter', 0.0, 1.0, key='diameter_key')
    Height = st.number_input('Height', 0.0, 1.5, key='height_key')
    Whole_weight = st.number_input('Whole weight', 0.0, 3.0, key='whole_weight_key')
    Shucked_weight = st.number_input('Shucked weight', 0.0, 1.5, key='shucked_weight_key')
    Viscera_weight = st.number_input('Viscera weight', 0.0, 1.0, key='viscera_weight_key')
    Shell_weight = st.number_input('Shell weight', 0.0, 1.0, key='shell_weight_key')
    Rings = st.number_input('Rings: ',1,30,key='rings_key')

    # Creating a dictionary
    abalone_data = {
        "Length": Length,
        "Diameter": Diameter,
        "Height": Height,
        "Whole weight": Whole_weight,
        "Shucked weight": Shucked_weight,
        "Viscera weight": Viscera_weight,
        "Shell weight": Shell_weight,
        "Rings": Rings,
    }

    # Create a DataFrame with the dictionary and specify the index
    df = pd.DataFrame([abalone_data], index=["Abalone Data"])
    st.dataframe(df)

    # Loading the model and processing pipeline
    model = load_object("../Artifacts/Model.pkl")
    pipeline = load_object("../Artifacts/Processing_pipeline.pkl")

    # Creating a button for making prediction
    button = st.button("Make Prediction")
    if button:
        output = model.predict(pipeline.transform(df))
        if output == 2:
            st.write("Male")
        elif output == 1:
            st.write("Infant")
        else:
            st.write("Female")

if __name__ == "__main__":
    main()

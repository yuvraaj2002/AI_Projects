import streamlit as st
import pandas as pd
from Src.Utils import load_object

def main():
    st.sidebar.title("Dataset Description")
    st.sidebar.write("""
    * **Sex:** The sex of the abalone, either male, female, or infant.
    * **Length:** The longest shell measurement in millimeters.
    * **Diameter:** The perpendicular shell measurement in millimeters.
    * **Height:** The height of the abalone with meat in the shell in millimeters.
    * **Whole weight:** The weight of the whole abalone in grams.
    * **Shucked weight:** The weight of the abalone meat in grams.
    * **Viscera weight:** The weight of the abalone viscera (internal organs) in grams.
    * **Shell weight:** The weight of the abalone shell in grams.
    * **Rings:** The number of rings on the abalone shell, which is used to determine the abalone's age.
    """)

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
    model = load_object("Artifacts/Model.pkl")
    pipeline = load_object("Artifacts/Processing_pipeline.pkl")

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

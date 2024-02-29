import streamlit as st
import plotly.express as px
import pandas as pd
import pickle
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import (
    OrdinalEncoder,
    OneHotEncoder,
    MinMaxScaler,
    FunctionTransformer,
)

st.markdown(
    """
        <style>
               .block-container {
                    padding-top: 0.5rem;
                    padding-bottom: 0rem;
                    # padding-left: 2rem;
                    # padding-right:2rem;
                }
        </style>
        """,
    unsafe_allow_html=True,
)

lat_long_df = pd.read_csv("Artifacts/latlong.csv")

group_df = lat_long_df.groupby("sector").mean(numeric_only=True)[
    ["price", "built_up_area", "latitude", "longitude"]
]


property_type_dict = {"Flat": "flat", "Independent House": "Independent_house"}
agePossession_options = (
    "Relatively New",
    "Moderately Old",
    "New Property",
    "Old Property",
    "Under Construction",
)
luxury_category_options = ("Medium", "Low", "High")
floor_category_options = ("Mid Floor", "Low Floor", "High Floor")
servant_room_options = {"No": 0.0, "Yes": 1.0}
balcony_options = ("0", "1", "2", "3", "3+")
furnishing_type_options = {
    "Unfurnished": 0.0,
    "Semi Furnished": 1.0,
    "Fully furnished": 2.0,
}
sector_options = (
    "sector 1",
    "sector 33",
    "sector 102",
    "sector 85",
    "sector 70",
    "sector 92",
    "sector 69",
    "sector 90",
    "sector 81",
    "sector 65",
    "sector 109",
    "sector 79",
    "sector 3",
    "sector 25",
    "sector 104",
    "sector 67",
    "sector 83",
    "sector 37",
    "sector 86",
    "sector 89",
    "sector 50",
    "sector 82",
    "sector 107",
    "sector 108",
    "sector 95",
    "sector 56",
    "sector 99",
    "sector 48",
    "sector 84",
    "sector 49",
    "sector 66",
    "sector 24",
    "sector 103",
    "sector 113",
    "sector 43",
    "sector 61",
    "sector 106",
    "sector 68",
    "sector 63",
    "sector 7",
    "sector 12",
    "sector 77",
    "sector 72",
    "sector 53",
    "sector 54",
    "sector 71",
    "sector 88",
    "sector 55",
    "sector 112",
    "sector 36",
    "sector 9",
    "sector 57",
    "sector 111",
    "sector 105",
    "sector 22",
    "sector 47",
    "sector 110",
    "sector 78",
    "sector 74",
    "sector 62",
    "sector 11",
    "sector 28",
    "sector 91",
    "sector 6",
    "sector 93",
    "sector 60",
    "sector 14",
    "sector 76",
    "sector 52",
    "sector 46",
    "sector 39",
    "sector 31",
    "sector 58",
    "sector 38",
    "sector 40",
    "sector 26",
    "sector 4",
    "sector 45",
    "sector 23",
    "sector 59",
    "sector 10",
    "sector 17",
    "sector 51",
    "sector 13",
    "sector 8",
    "sector 5",
    "sector 15",
    "sector 2",
    "sector 80",
    "sector 18",
    "sector 30",
    "sector 21",
    "sector 73",
    "sector 27",
)


@st.cache_resource
def load_reg_pipeline():

    # Load the pipeline from the pickle file
    with open(
        "Artifacts/Regression_Pipeline.pkl",
        "rb",
    ) as file:
        reg_pipeline = pickle.load(file)
        return reg_pipeline


def create_input_features(
    Property_Type,
    bathroom,
    agePossession,
    furnishing_type,
    built_up_area,
    sector,
    bedRoom,
    luxury_category,
    floor_category,
    servant_room,
    balcony,
):
    """
    This method will take the input variables and will return a DataFrame with input feature values.
    :return: DataFrame
    """
    # Create a dictionary with your input variables
    data = {
        "balcony": [balcony],
        "agePossession": [agePossession],
        "luxury_category": [luxury_category],
        "floor_category": [floor_category],
        "Property_Type": [property_type_dict[Property_Type]],
        "sector": [float(sector.split()[1])],
        "built_up_area": [built_up_area],
        "bedRoom": [float(bedRoom)],
        "bathroom": [float(bathroom)],
        "servant room": [servant_room_options[servant_room]],
        "furnishing_type": [furnishing_type_options[furnishing_type]],
    }

    # Create a DataFrame from the dictionary
    df = pd.DataFrame(data)
    return df


# Defining the function for doing the cube root transformation
def cube_root_transform(column):
    return np.cbrt(column)


def process_input(Input_df):
    """
    This method will take the input dataframe and return a 2d numpy array (Processed data) for making prediction
    :return:
    """
    reg_pipeline = load_reg_pipeline()
    Input = reg_pipeline.transform(Input_df)
    return Input


def predict(Input):
    """
    This method will take 2d Numpy array, load the model, make prediction and return the result
    :return:
    """

    # Load the model from the pickle file
    model = xgb.Booster(model_file="Artifacts/xgboost_regressor_model.bin")

    # Assuming 'Input' is your numpy.ndarray
    data_matrix = xgb.DMatrix(Input)
    predicted_value = model.predict(data_matrix)
    return predicted_value


def Price_Prediction_Page():

    page_col1, page_col2 = st.columns(spec=(2, 1.5), gap="large")
    with page_col1:
        st.markdown(
            "<h1 class='center' style='font-size: 45px;'>Share Your Details for Price Prediction📝</h1>",
            unsafe_allow_html=True,
        )
        Guideline_text = "<p style='font-size: 18px;'>In order to receive the most accurate price prediction for your property, it's crucial to provide precise and comprehensive details.The more specific and accurate your details about your property's location, size, condition, and any unique features, the more refined and trustworthy the price estimate will be.</p>"
        st.markdown(Guideline_text, unsafe_allow_html=True)
        input_col1, input_col2 = st.columns(spec=(1, 1), gap="large")
        with input_col1:
            Property_Type = st.selectbox(
                "Select Property Type",
                ("Flat", "Independent House"),
                index=None,
                placeholder="Choose Property type you are looking for",
                key="property_type_input",
            )
            bathroom = st.slider(
                "Select Number of Bathroom you want",
                min_value=1,
                max_value=14,
                value=5,
                step=1,
                key="bathroom_input",
            )
            agePossession = st.selectbox(
                "Choose age possession",
                agePossession_options,
                index=None,
                placeholder="Select age possession type",
                key="agePoss_input",
            )
            furnishing_type = st.selectbox(
                "Select Furnishing category",
                ("Unfurnished", "Semi Furnished", "Fully furnished"),
                index=None,
                placeholder="How much furnishing do you want for your property?",
                key="furnishing_input",
            )
            built_up_area = st.slider(
                "Enter Built up Area you want (sq.ft)",
                min_value=1000,
                max_value=10000,
                value=1600,
                step=1,
            )
            sector = st.selectbox(
                "Select Sector",
                sector_options,
                index=None,
                placeholder="Which sector are you searching for a house in?",
                key="sector_input",
            )

        with input_col2:
            bedRoom = st.slider(
                "Select Number of Bedrooms you want",
                min_value=1,
                max_value=14,
                value=5,
                step=1,
                key="bedroom_input",
            )
            luxury_category = st.selectbox(
                "Select Luxury Category",
                luxury_category_options,
                index=None,
                placeholder="How much luxurious house you want?",
                key="luxury_category_input",
            )
            floor_category = st.selectbox(
                "Select Floor Category",
                floor_category_options,
                index=None,
                placeholder="Select Floor Category (Flats)",
                key="floor_category_input",
            )
            servant_room = st.selectbox(
                "Servant room",
                ("Yes", "No"),
                index=None,
                placeholder="Are you looking for servant room?",
                key="servent_input",
            )
            balcony = st.selectbox(
                "Balcony in house",
                balcony_options,
                index=None,
                placeholder="How many balcony are you looking for",
                key="balcony_input",
            )

            predict_button = st.button(
                label="What would be the estimated Price?",
                key="price_prediction_button",
                use_container_width=True,
            )
            if predict_button:
                Input_df = create_input_features(
                    Property_Type,
                    bathroom,
                    agePossession,
                    furnishing_type,
                    built_up_area,
                    sector,
                    bedRoom,
                    luxury_category,
                    floor_category,
                    servant_room,
                    balcony,
                )
                Input = process_input(Input_df)
                Predicted_value = predict(Input)
                st.write(
                    "<p style='font-size: 20px;'>The estimated Price of the property will be <strong>"
                    + str(round(Predicted_value[0], 2))
                    + "Cr</strong></p>",
                    unsafe_allow_html=True,
                )

    with page_col2:
        # Create your plotly map
        fig = px.scatter_mapbox(
            group_df,
            lat="latitude",
            lon="longitude",
            color="price",
            size="built_up_area",
            color_continuous_scale=px.colors.cyclical.IceFire,
            zoom=10,
            mapbox_style="open-street-map",
            text=group_df.index,
        )

        # Adjust the height of the map
        fig.update_layout(height=800)

        # Use st.plotly_chart to display the plotly figure within the column
        st.write("")
        st.plotly_chart(fig)


Price_Prediction_Page()

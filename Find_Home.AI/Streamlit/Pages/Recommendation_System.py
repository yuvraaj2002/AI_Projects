import streamlit as st
import plotly.express as px
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import random
import time


# Loading the facilities dataframe
with open(
    "/home/yuvraj/Documents/AI/AI_Projects/Find_Home.AI/Notebook_And_Dataset/Trained_Model/Facilities_RE.pkl",
    "rb",
) as file:
    Facilities_Recomm_df = pickle.load(file)

# Loading the cosine similarities
with open(
    "/home/yuvraj/Documents/AI/AI_Projects/Find_Home.AI/Notebook_And_Dataset/Trained_Model/CosineSim_Prices.pkl",
    "rb",
) as file:
    Cosine_Similarity_Prices = pickle.load(file)

with open(
    "/home/yuvraj/Documents/AI/AI_Projects/Find_Home.AI/Notebook_And_Dataset/Trained_Model/CosineSim_facilities.pkl",
    "rb",
) as file:
    Cosine_Similarity_Facilities = pickle.load(file)



def recommend_properties_with_scores(property_name, top_n=5):
    """
    This method will take the property name as an input and will return 5
    most similar properties
    """
    facilities_wt = random.uniform(0.0, 1.0)
    price_wt = 1-facilities_wt
    cosine_sim_matrix = facilities_wt * Cosine_Similarity_Facilities + price_wt * Cosine_Similarity_Prices

    # Get the similarity scores for the property using its name as the index
    sim_scores = list(enumerate(cosine_sim_matrix[Facilities_Recomm_df.index.get_loc(property_name)]))

    # Sort properties based on the similarity scores
    sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the indices and scores of the top_n most similar properties
    top_indices = [i[0] for i in sorted_scores[1:top_n + 1]]
    top_scores = [i[1] for i in sorted_scores[1:top_n + 1]]

    # Retrieve the names of the top properties using the indices
    top_properties = Facilities_Recomm_df.index[top_indices].tolist()

    # Create a dataframe with the results
    recommendations_df = pd.DataFrame({
        'PropertyName': top_properties,
        'SimilarityScore': top_scores
    })

    return recommendations_df




def Recommendation_System_Page():
    page_col1, page_col2 = st.columns(spec=(2, 1), gap="large")

    with page_col1:
        st.title("Recommendation Engine 👨‍💼")
        Input_value_text = '<p style="font-size: 20px;padding-bottom:1rem">To receive tailored recommendations from each recommendation system, please ensure to input the details accurately. Specifically, provide the expected price range you are considering, allowing the algorithms to refine their suggestions based on your budget. Additionally, specify the names of the apartments you are interested in. This focused input on expected price and apartment names will significantly improve the relevance and suitability of the recommendations provided by the systems, streamlining your search for the perfect apartment that aligns with your preferences and financial considerations..</p>'
        st.markdown(Input_value_text, unsafe_allow_html=True)

        Input_price_col, Input_apartment_col = st.columns(spec=(1, 1), gap="large")
        user_input_price = 2.5
        user_input_apartment = "M3M Crown"

        user_input_apartment = st.selectbox(
            "Select any Apartment",
            Facilities_Recomm_df.index.values,
            index=None,
            placeholder="Select Apartment for which you want to get recommendations",
            key="user_input_apartment",
        )

        # Checking if the user have provided input or not for the recommendation engine
        if any(
            [
                user_input_apartment == None,
            ]
        ):
            st.error("Please select some appartment for getting recommendations")
        else:
            progress_text = "Finding the best place for you🔎."
            my_bar = st.progress(0, text=progress_text)
            for percent_complete in range(100):
                time.sleep(0.05)
                my_bar.progress(percent_complete + 1, text=progress_text)
            time.sleep(1)
            my_bar.empty()

            st.markdown("***")

            facilities_results = recommend_properties_with_scores(user_input_apartment)
            st.write(facilities_results)


    with page_col2:
        st.markdown(
            "<p style='background-color: #CEFCBA; padding: 2rem; border-radius: 10px; font-size: 18px;'>This recommendation system comprises a fusion of 2 distinct recommendation engines: Facilities-based recommendations, Price-based recommendations. The ultimate recommendation is derived from the collective outcomes of these 2 recommendation systems. To assign greater significance to a specific recommendation system, you have the flexibility to adjust the weighting percentage below.</p>",
            unsafe_allow_html=True,
        )

        # Input for the Facilities based recommendation system weight
        facilities_recommendation_wt = st.slider(
            "Select the Weightage of Facilities based recommendation system (%)",
            min_value=1,
            max_value=100,
            value=30,
            step=1,
            key="facilities_recommendation_wt",
        )

        # Create data for the pie chart
        data = {
            "Categories": ["Facilities", "Price"],
            "Weights": [
                facilities_recommendation_wt,
                100 - facilities_recommendation_wt,
            ],
        }

        # Create a DataFrame from the data
        df = pd.DataFrame(data)
        custom_colors = ["#AEF359", "#03C04A"]

        # Create a dynamic pie chart using Plotly Express
        fig = px.pie(
            df,
            names="Categories",
            values="Weights",
            title="Recommendation System Weights",
            color_discrete_sequence=custom_colors,
        )
        st.plotly_chart(fig, use_container_width=True)

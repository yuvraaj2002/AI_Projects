import streamlit as st
import plotly.express as px
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import random
import time

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


# Loading the facilities dataframe
with open(
    "Artifacts/Facilities_RE.pkl",
    "rb",
) as file:
    Facilities_Recomm_df = pickle.load(file)

# Loading the cosine similarities
with open(
    "Artifacts/CosineSim_Prices.pkl",
    "rb",
) as file:
    Cosine_Similarity_Prices = pickle.load(file)

with open(
    "Artifacts/CosineSim_facilities.pkl",
    "rb",
) as file:
    Cosine_Similarity_Facilities = pickle.load(file)


def recommend_properties_with_scores(
    property_name, facilities_recommendation_wt, top_n=5
):
    """
    This method will take the property name as an input and will return 5
    most similar properties
    """
    facilities_wt_normalized = facilities_recommendation_wt / 100
    price_wt_normalized = 1 - facilities_wt_normalized
    cosine_sim_matrix = (
        facilities_wt_normalized * Cosine_Similarity_Facilities
        + price_wt_normalized * Cosine_Similarity_Prices
    )

    # Get the similarity scores for the property using its name as the index
    sim_scores = list(
        enumerate(cosine_sim_matrix[Facilities_Recomm_df.index.get_loc(property_name)])
    )

    # Sort properties based on the similarity scores
    sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the indices and scores of the top_n most similar properties
    top_indices = [i[0] for i in sorted_scores[1 : top_n + 1]]
    top_scores = [i[1] for i in sorted_scores[1 : top_n + 1]]

    # Retrieve the names of the top properties using the indices
    top_properties = Facilities_Recomm_df.index[top_indices].tolist()

    # Create a dataframe with the results
    recommendations_df = pd.DataFrame(
        {"PropertyName": top_properties, "SimilarityScore": top_scores}
    )

    return recommendations_df


def Recommendation_System_Page():
    st.markdown(
        "<h1 style='text-align: center; font-size: 50px; '>Recommendation Engine 👨‍💼</h1>",
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown(
            "<p style='font-size: 20px; text-align: center;padding-left: 2rem;padding-right: 2rem;padding-bottom: 1rem;'>To enhance the precision of tailored recommendations from our recommendation system, accurately input details such as the expected price range you're considering and the names of the apartments you're interested in. This focused input enables our fusion of two distinct recommendation engines—Facilities-based and Price-based recommendations—to refine suggestions based on your budget and preferences. The ultimate recommendation derives from the collective outcomes of both systems, and you can further customize the significance of each by adjusting the weighting percentage below. This streamlined approach ensures that our recommendations align with your preferences and financial considerations, optimizing your search for the perfect apartment.</p>",
            unsafe_allow_html=True,
        )

        apartment_input_col, weight_input_col = st.columns(spec=(1, 1), gap="large")
        with apartment_input_col:
            user_input_apartment = st.selectbox(
                "Select any Apartment",
                Facilities_Recomm_df.index.values,
                index=None,
                placeholder="Select Apartment for which you want to get recommendations",
                key="user_input_apartment",
            )
        with weight_input_col:
            # Input for the Facilities based recommendation system weight
            facilities_recommendation_wt = st.slider(
                "Select the Weightage of Facilities based recommendation system (%)",
                min_value=1,
                max_value=100,
                value=30,
                step=1,
                key="facilities_recommendation_wt",
            )

    st.markdown("***")

    recommendation_col, weight_plot_col = st.columns(spec=(2.2, 1), gap="large")
    with recommendation_col:
        # Checking if the user has provided input or not for the recommendation engine
        if any(
            [
                user_input_apartment == None,
            ]
        ):
            st.error("Please select some apartment for getting recommendations")
        else:
            progress_text = "Finding the best place for you🔎."
            my_bar = st.progress(0, text=progress_text)
            for percent_complete in range(100):
                time.sleep(0.05)
                my_bar.progress(percent_complete + 1, text=progress_text)
            time.sleep(1)
            my_bar.empty()

            facilities_results = recommend_properties_with_scores(
                user_input_apartment, facilities_recommendation_wt
            )
            baseline_similarity_score = facilities_results["SimilarityScore"].iloc[4]

            st.markdown(
                "<p style='font-size: 18px; padding-bottom: 1rem;'>Presenting the top five apartments meticulously curated for your consideration, derived from your selected apartment and the thoughtful configurations of our recommendation engine weights. We trust these recommendations will add value to your search and enhance your experience in finding the ideal residence.</p>",
                unsafe_allow_html=True,
            )
            row = st.columns(5)
            index = 0
            for col in row:
                tile = col.container(height=200)  # Adjust the height as needed
                tile.markdown(
                    "<p style='text-align: left; font-size: 18px; '>"
                    + str(facilities_results["PropertyName"][index])
                    + "</p>",
                    unsafe_allow_html=True,
                )
                if index == 4:
                    tile.metric(
                        label="Similarity Score",
                        value=round(facilities_results["SimilarityScore"][index], 3),
                        delta="Base line score",
                    )
                else:
                    tile.metric(
                        label="Similarity Score",
                        value=round(facilities_results["SimilarityScore"][index], 3),
                        delta=round(
                            facilities_results["SimilarityScore"][index]
                            - baseline_similarity_score,
                            5,
                        ),
                    )
                index = index + 1

            # st.write(facilities_results)

    with weight_plot_col:
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


Recommendation_System_Page()

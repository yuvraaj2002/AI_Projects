import streamlit as st
import plotly.express as px
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from Src.Facility_RE import facilities_recommend_properties
from Src.Price_RE import recommend_properties_price_details
from Src.Location_RE import recommend_properties_locations


Facilities_df = pd.read_csv(
    "/home/yuvraj/Github/Machine_Learning_Projects/Find_Home.AI/Notebook_And_Dataset/Cleaned_datasets/Facilities_RE.csv"
)
Price_Recomm_df = pd.read_csv(
    "/home/yuvraj/Github/Machine_Learning_Projects/Find_Home.AI/Notebook_And_Dataset/Cleaned_datasets/Price_RE.csv",index_col= "PropertyName"
)

def recommend_properties_price(property_name, top_n=5):
    
    # Compute the cosine similarity matrix
    cosine_sim_price_details = cosine_similarity(Price_Recomm_df)

    # Get the similarity scores for the property using its name as the index
    sim_scores = list(enumerate(cosine_sim_price_details[Price_Recomm_df.index.get_loc(property_name)]))

    # Sort properties based on the similarity scores
    sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the indices and scores of the top_n most similar properties
    top_indices = [i[0] for i in sorted_scores[1:top_n + 1]]
    top_scores = [i[1] for i in sorted_scores[1:top_n + 1]]

    # Retrieve the names of the top properties using the indices
    top_properties = Price_Recomm_df.index[top_indices].tolist()

    # Create a dataframe with the results
    price_recommendations_df = pd.DataFrame({
        'PropertyName': top_properties,
        'SimilarityScore': top_scores
    })

    return price_recommendations_df


def recommend_properties_facilities(property_name):
    """
    This method will take the property name as an input and will return 5
    most similar properties
    """

    # Creating word embedding using tf-idf and Calculating the cosine similarity between the vectors
    tfidf_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = tfidf_vectorizer.fit_transform(Facilities_df["Facilities_Str"])
    cosine_sim_facilities = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Getting the index of the property that matches the name
    idx = Facilities_df[Facilities_df["PropertyName"] == property_name].index[0]

    # Calculating the similarity scores
    sim_scores = list(enumerate(cosine_sim_facilities[idx]))

    # Sort the properties based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 5 most similar properties
    sim_scores = sim_scores[1:6]

    # Get the property indices
    property_indices = [i[0] for i in sim_scores]

    facilities_recommendations_df = pd.DataFrame(
        {
            "PropertyName": Facilities_df["PropertyName"].iloc[property_indices],
            "SimilarityScore": sim_scores,
        }
    )

    # Return the top 5 most similar properties
    return facilities_recommendations_df


def Recommendation_System_Page():
    page_col1, page_col2 = st.columns(spec=(1.4, 2.0), gap="large")
    with page_col1:
        st.title("Recommendation Configuration 🔩")
        html_text_intro = '<p style="font-size: 20px;">This recommendation system comprises a fusion of 2 distinct recommendation engines: Facilities-based recommendations, Price-based recommendations. The ultimate recommendation is derived from the collective outcomes of these 2 recommendation systems. To assign greater significance to a specific recommendation system, you have the flexibility to adjust the weighting percentage below.</p>'
        st.markdown(html_text_intro, unsafe_allow_html=True)
        st.write("")

        configuation_col, input_col = st.columns(spec=(1, 1), gap="large")
        with configuation_col:
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
            price_recommendation_wt = st.slider(
                "Select the Weightage of Price based recommendation system (%)",
                min_value=1,
                max_value=100,
                value=30,
                step=1,
                key="price_recommendation_wt",
            )

        with input_col:
            # Create data for the pie chart
            data = {
                "Categories": ["Facilities", "Price"],
                "Weights": [facilities_recommendation_wt, price_recommendation_wt],
            }

            # Create a DataFrame from the data
            df = pd.DataFrame(data)
            custom_colors = ["#AEF359", "#03C04A"]
            #  "#0B6623"

            # Create a dynamic pie chart using Plotly Express
            fig = px.pie(
                df,
                names="Categories",
                values="Weights",
                title="Recommendation System Weights",
                color_discrete_sequence=custom_colors,
            )
            st.plotly_chart(fig, use_container_width=True)

    with page_col2:
        st.title("Enter Coverage and Apartment details 🏡")
        Input_value_text = '<p style="font-size: 20px;">To receive recommendations from each of the recommendation systems, you need to provide input specifying the location you are interested in, the desired coverage in kilometers, and the name of the apartment.</p>'
        st.markdown(Input_value_text, unsafe_allow_html=True)

        Input_price_col, Input_apartment_col = st.columns(spec=(1, 1), gap="large")
        user_input_price = 2.5
        user_input_apartment = "M3M Crown"

        with Input_price_col:
            user_input_price = st.number_input(
                "Enter the price",
                value=None,
                placeholder="Enter the price(Cr), Example: 2.5",
                step=0.1,
                min_value=0.07,
                max_value=20.0,
            )

        with Input_apartment_col:
            user_input_apartment = st.selectbox(
                "Select any Apartment",
                Facilities_df["PropertyName"].value_counts().index,
                index=None,
                placeholder="Select Apartment for which you want to get recommendations",
                key="user_input_apartment",
            )

        # st.markdown("***")
        st.markdown("***")
        facilities_recommendations = st.toggle(
            "Show Facilities based recommendations", key="facilities_toggle"
        )
        if facilities_recommendations:
            facilities_results = recommend_properties_facilities(user_input_apartment)
            facilities_results = facilities_results["PropertyName"].values
            st.write(facilities_results)

        price_recommendations = st.toggle(
            "Show Price based recommendations", key="price_toggle"
        )
        if price_recommendations:
            price_results = recommend_properties_price(user_input_apartment)
            price_results = price_results["PropertyName"].values
            st.write(price_results)

        st.title("Combined recommendations")
        st.dataframe(Price_Recomm_df.head(5))

import numpy as np
import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import ast

df = pd.read_csv('/home/yuvraj/Github/Machine_Learning_Projects/Find_Home.AI/Notebook_And_Dataset/Cleaned_datasets/Locations_RE.csv')
df = df.set_index("PropertyName")

# Calculating the cosine similarity
cosine_sim_location = cosine_similarity(df)


def recommend_properties_locations(property_name, top_n=5):
    # Get the similarity scores for the property using its name as the index
    sim_scores = list(enumerate(cosine_sim_location[df.index.get_loc(property_name)]))

    # Sort properties based on the similarity scores
    sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the indices and scores of the top_n most similar properties
    top_indices = [i[0] for i in sorted_scores[1:top_n + 1]]
    top_scores = [i[1] for i in sorted_scores[1:top_n + 1]]

    # Retrieve the names of the top properties using the indices
    top_properties = df.index[top_indices].tolist()

    # Create a dataframe with the results
    recommendations_df = pd.DataFrame({
        'PropertyName': top_properties,
        'SimilarityScore': top_scores
    })

    return recommendations_df
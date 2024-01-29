import numpy as np
import pandas as pd
import re
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(
    "/home/yuvraj/Github/Machine_Learning_Projects/Find_Home.AI/Notebook_And_Dataset/Cleaned_datasets/Price_RE.csv"
)
df = df.set_index("PropertyName")

# Compute the cosine similarity matrix
cosine_sim_price_details = cosine_similarity(df)


def recommend_properties_price_details(property_name, top_n=5):
    # Get the similarity scores for the property using its name as the index
    sim_scores = list(
        enumerate(cosine_sim_price_details[df.index.get_loc(property_name)])
    )

    # Sort properties based on the similarity scores
    sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the indices and scores of the top_n most similar properties
    top_indices = [i[0] for i in sorted_scores[1 : top_n + 1]]
    top_scores = [i[1] for i in sorted_scores[1 : top_n + 1]]

    # Retrieve the names of the top properties using the indices
    top_properties = df.index[top_indices].tolist()

    # Create a dataframe with the results
    recommendations_df = pd.DataFrame(
        {"PropertyName": top_properties, "SimilarityScore": top_scores}
    )

    return recommendations_df

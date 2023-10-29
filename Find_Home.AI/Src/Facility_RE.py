import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('/home/yuvraj/Github/Machine_Learning_Projects/Find_Home.AI/Notebook_And_Dataset/Cleaned_datasets/Facilities_RE.csv')
tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))

# Creating word embedding using tf-idf and caluclating the cosine similarity
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Facilities_Str'])
cosine_sim_facilities = cosine_similarity(tfidf_matrix, tfidf_matrix)


def facilities_recommend_properties(property_name):
    """
    This method will take the property name as an input and will return 5
    most similar properties
    """

    # Getting the index of the property that matches the name
    idx = df[df['PropertyName'] == property_name].index[0]

    # Calculating the similarity scores
    sim_scores = list(enumerate(cosine_sim_facilities[idx]))

    # Sort the properties based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 5 most similar properties
    sim_scores = sim_scores[1:6]

    # Get the property indices
    property_indices = [i[0] for i in sim_scores]

    recommendations_df = pd.DataFrame({
        'PropertyName': df['PropertyName'].iloc[property_indices],
        'SimilarityScore': sim_scores
    })

    # Return the top 10 most similar properties
    return recommendations_df



# AI-Powered Movie Recommendation System

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import streamlit as st

# Step 1: Load the data (MovieLens small dataset used as an example)
movies = pd.read_csv("https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/books.csv")

# For this example, we simulate genres and directors columns
def mock_genres(title):
    if "Harry" in title:
        return ["Fantasy", "Adventure"]
    elif "Twilight" in title:
        return ["Romance", "Drama"]
    else:
        return ["Fiction", "Mystery"]

movies['genres'] = movies['title'].apply(mock_genres)

movies['director'] = movies['authors']  # Using author as a proxy for director

# Step 2: Preprocessing
# Combine features
movies['combined_features'] = movies.apply(lambda x: ' '.join(x['genres']) + ' ' + x['director'], axis=1)

# Step 3: Vectorize the combined features
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['combined_features'])

# Step 4: Compute similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Step 5: Build a recommendation function
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = movies[movies['title'].str.contains(title, case=False, na=False)].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Top 5
    movie_indices = [i[0] for i in sim_scores]
    recommendations = movies.iloc[movie_indices][['title', 'authors']]
    explanations = [
        f"Because it's also by {movies.iloc[i]['director']} and shares genres like {' & '.join(movies.iloc[i]['genres'])}."
        for i in movie_indices
    ]
    recommendations['Why Recommended'] = explanations
    return recommendations

# Step 6: Streamlit UI
st.title("🎬 AI Movie Recommendation System")

user_input = st.text_input("Enter a movie title you like:", "Harry Potter")

if user_input:
    try:
        recs = get_recommendations(user_input)
        st.write("### Recommended Movies:")
        st.dataframe(recs)
    except:
        st.write("Movie not found. Try another title.")
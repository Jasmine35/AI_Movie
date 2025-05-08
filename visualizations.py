# visualizations.py
# Used to plot cosine similarity matrix and TF-IDF bar chart for each searched word in the movie recommender

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st

# Plot the top TF-IDF terms
def plot_top_tfidf_words(movie_title, movies_df):
    movie = movies_df[movies_df['title'] == movie_title]
    if movie.empty:
        st.warning("Movie not found.")
        return

    text = movie.iloc[0]['combined_features']
    if not isinstance(text, str):
        st.warning("Text features unavailable.")
        return

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform([text])
    words = tfidf.get_feature_names_out()
    scores = tfidf_matrix.toarray().flatten()

    tfidf_df = pd.DataFrame({'word': words, 'score': scores})
    top_words = tfidf_df.sort_values(by='score', ascending=False).head(10)

    fig, ax = plt.subplots()
    sns.barplot(x='score', y='word', data=top_words, ax=ax)
    ax.set_title(f"Top TF-IDF Words: {movie_title}")
    st.pyplot(fig)

# Plot a similarity heatmap for top N similar movies
def plot_similarity_heatmap(movie_title, movies_df, similarity_matrix, top_n=10):
    try:
        idx = movies_df[movies_df['title'] == movie_title].index[0]
    except IndexError:
        st.warning("Movie not found in the dataset.")
        return

    similarity_scores = list(enumerate(similarity_matrix[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_indices = [i for i, _ in similarity_scores]

    titles = movies_df.iloc[movie_indices]['title'].values
    sub_matrix = similarity_matrix[movie_indices][:, movie_indices]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(sub_matrix, xticklabels=titles, yticklabels=titles, cmap="YlGnBu", annot=True, fmt=".2f")
    ax.set_title(f"Similarity Heatmap for '{movie_title}' and Top {top_n} Similar Movies")
    st.pyplot(fig)

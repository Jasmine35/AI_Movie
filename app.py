# This is the program where the main UI is ran
# it uses precompute_similarity.py to compute the similarity matrix with a fraction of the databases 
# and use that matrix to select the recommended movies and build the explanations

import pandas as pd
import streamlit as st
import ast  # Safe alternative to eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from visualizations import plot_top_tfidf_words, plot_similarity_heatmap

# Load preprocessed movie data
@st.cache_data
def load_data():
    movies = pd.read_csv("top_movies.csv") # using top because the time to run the app is exponential in time and memory
    
    # Safely parse stringified lists
    for col in ['genres', 'directors', 'actors']:
        def safe_parse(x):
            try:
                return ast.literal_eval(x) if isinstance(x, str) else []
            except (ValueError, SyntaxError):
                return []
        movies[col] = movies[col].apply(safe_parse)

    return movies

movies = load_data()

# Create similarity matrix from combined text features
@st.cache_resource
def get_similarity_matrix(movies_df):
    # Combine relevant text features into a single string per movie
    def combine_features(x):
        title = x['title'] if isinstance(x['title'], str) else ''
        genres = ' '.join(x['genres']) if isinstance(x['genres'], list) else ''
        directors = ' '.join(x['directors']) if isinstance(x['directors'], list) else ''
        actors = ' '.join(x['actors']) if isinstance(x['actors'], list) else ''
        return f"{title} {genres} {directors} {actors}"

    movies_df['combined_features'] = movies_df.apply(combine_features, axis=1)

    # Vectorize the text using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(movies_df['combined_features'])

    # Compute cosine similarity
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

similarity_matrix = get_similarity_matrix(movies)

# --- Streamlit UI ---
st.title("🎬 AI Movie Recommender")

# Movie selection dropdown
movie_titles = movies['title'].tolist()
selected_title = st.selectbox("Choose a movie you like:", movie_titles)

# Find index of selected movie
if selected_title:
    idx = movies[movies['title'] == selected_title].index[0]

    # Get pairwise similarity scores for that movie
    similarity_scores = list(enumerate(similarity_matrix[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Get top 5 most similar movies (excluding the movie itself)
    top_indices = [i for i, score in similarity_scores[1:6]]

    st.subheader("📽️ Recommended Movies")
    selected = movies.iloc[idx]

    for i in top_indices:
        movie = movies.iloc[i]
        title = movie['title']
        year = int(movie['year']) if not pd.isna(movie['year']) else 'Unknown'
        rating = round(movie['averageRating'], 1) if not pd.isna(movie['averageRating']) else 'N/A'

        # Identify shared features
        shared_genres = set(selected['genres']).intersection(set(movie['genres']))
        shared_directors = set(selected.get('directors', [])).intersection(set(movie.get('directors', [])))
        shared_actors = set(selected.get('actors', [])).intersection(set(movie.get('actors', [])))

        # Build explanation
        reasons = []
        if shared_genres:
            reasons.append(f"Genres: {', '.join(shared_genres)}")
        if shared_directors:
            reasons.append(f"Director(s): {', '.join(shared_directors)}")
        if shared_actors:
            reasons.append(f"Actor(s): {', '.join(list(shared_actors)[:3])}")  # limit to 3 actors

        reason_text = "; ".join(reasons) if reasons else "Similar content features"

        # Display result
        st.markdown(f"**{title}** ({year}) – ⭐ {rating}")
        st.caption(f"🧠 Recommended because of: {reason_text}")

# plot the heat map and the bar graph
plot_top_tfidf_words(selected_title, movies)
plot_similarity_heatmap(selected_title, movies, similarity_matrix)


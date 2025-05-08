import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load cleaned movie data
movies = pd.read_csv("cleaned_data.csv")

# Use top 20,0000 movies by rating
movies = movies.sort_values(by='averageRating', ascending=False).head(15000).reset_index(drop=True)

# Prepare combined text features
def combine_features(row):
    try:
        title = row['title'] if isinstance(row['title'], str) else ''
        genres = ' '.join(eval(row['genres'])) if isinstance(row['genres'], str) else ''
        directors = ' '.join(eval(row['directors'])) if isinstance(row['directors'], str) else ''
        actors = ' '.join(eval(row['actors'])) if isinstance(row['actors'], str) else ''
        return f"{title} {genres} {directors} {actors}"
    except:
        return ''

movies['combined_features'] = movies.apply(combine_features, axis=1)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(movies['combined_features'])

# Compute cosine similarity
similarity_matrix = cosine_similarity(tfidf_matrix)

# Save the matrix and the reduced movie list
np.save("similarity_matrix.npy", similarity_matrix)
movies.to_csv("top_movies.csv", index=False)
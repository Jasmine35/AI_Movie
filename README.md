AI Movie Recommender

By: Jasmine Yee


This project is an AI-powered movie recommendation system that suggests similar movies based on natural language features like title, genre, and actors. It uses **Natural Language Processing (NLP)** techniques and **cosine similarity** to identify relationships between films and provide recommendations, along with an explanation for each suggestion.

Features:

- TF-IDF Vectorization of movie metadata (title, genres, actors)

- Cosine Similarity for comparing movie vectors

- HeatMap and Bar Graphs for visualizing similarity and word importance

- Explainable Recommendations that shows why each movie was suggested

- Uses StreamLit app for an interactive user interface

The dataset contains movie metadata including:
  - Title
  - Genre
  - Actors
  - Description
- Subset of the top 15000 movies used for faster performance (scalable to full dataset).

Technologies:
- Python
- Pandas
- Scikit-learn
- Streamlit
- Matplotlib / Seaborn

# Data preprocessing for the IMDb database

import pandas as pd

# Step 1: Filter Titles: From title.basics.tsv, select movies (titleType == 'movie') 

# load/filter title.basics​ 
basics_cols = ['tconst', 'titleType', 'primaryTitle', 'startYear', 'runtimeMinutes', 'genres']
basics = pd.read_csv("/Users/jasmine/Downloads/AI_Movie/title.basics.tsv.gz", sep='\t', usecols = basics_cols, dtype = str, na_values = '\\N')

# filter movies between 1972 and 2016 (certiain boundary for the movies included in this database)
movies = basics[
    (basics['titleType'] == 'movie') &
    (basics['startYear'].astype(float) >= 1972) &
    (basics['startYear'].astype(float) <= 2016)
].copy()

# convert the runtime and year to numeric values
movies['startYear'] = pd.to_numeric(movies['startYear'], errors = 'coerce')
movies['runtimeMinutes'] = pd.to_numeric(movies['runtimeMinutes'], errors = 'coerce')

# Step 2: extract genres as a list for each movie to handle multiple genres
movies['genres'] = movies['genres'].fillna('').apply(lambda x: x.split(',') if x else [])


# Step 3: Merge Datasets: Combine title.basics.tsv with title.ratings.tsv on the tconst field to associate ratings with each movie
ratings = pd.read_csv("/Users/jasmine/Downloads/AI_Movie/title.ratings.tsv.gz", sep = '\t', usecols=['tconst', 'averageRating'], dtype=str, na_values = '\\N')
ratings['averageRating'] = pd.to_numeric(ratings['averageRating'], errors='coerce')

# merge ratings into movies
movies = movies.merge(ratings, on = 'tconst', how = 'left')


# Step 4: Associate Directors and Actors:

# load title.principals and filter for directors and actors
principals = pd.read_csv("/Users/jasmine/Downloads/AI_Movie/title.principals.tsv.gz", sep = '\t', usecols=['tconst', 'nconst', 'category'], dtype=str, na_values = '\\N')
principals = principals[principals['category'].isin(['director', 'actor'])]

# load name.basics to get human-readable name
names = pd.read_csv("/Users/jasmine/Downloads/AI_Movie/name.basics.tsv.gz", sep = '\t', usecols=['nconst', 'primaryName'], dtype=str, na_values = '\\N')
names = names[['nconst', 'primaryName']]

# merge principal people with names
principals = principals.merge(names, on='nconst', how='left')

# Group by tconst to aggregate multiple directors or actors per movie

# group by movie and role
crew = principals.groupby(['tconst', 'category'])['primaryName'].apply(list).unstack(fill_value=[]).reset_index()

# final merge 
movies = movies.merge(crew, on='tconst', how='left')

# rename/clean the columns
movies.rename(columns={
    'primaryTitle': 'title',
    'startYear': 'year',
    'director': 'directors',
    'actor': 'actors',
}, inplace = True)

# preview the cleaned data - set the clean data as a file in AI project
print(movies[['title', 'year', 'genres', 'averageRating', 'directors', 'actors']].head())

movies.to_csv("cleaned_data.csv", index=False)
print("✅ Data preprocessing complete. Saved to cleaned_data.csv.") 
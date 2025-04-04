import pandas as pd
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
from fuzzywuzzy import process
import streamlit as st

# Importing album data
raw_data = pd.read_csv('rym_top_5000_all_time.csv')

# Remove missing values 
cleaned_data = raw_data.dropna().copy()

# Create 'Release Year' variable 
cleaned_data['Release Year'] = cleaned_data['Release Date'].astype(str).str[-4:].astype(int)

# Remove irrelevant variables 
cleaned_data.drop(['Release Date', 'Number of Ratings', 'Number of Reviews'], axis=1, inplace=True)

# Create feature vectors from categorical features 'Genres' and 'Descriptors'
initialise_tfidf_genres = TfidfVectorizer()
genres_tfidf = initialise_tfidf_genres.fit_transform(cleaned_data['Genres'])

initialise_tfidf_descriptors = TfidfVectorizer()
descriptors_tfidf = initialise_tfidf_descriptors.fit_transform(cleaned_data['Descriptors'])

categorical_features = hstack((genres_tfidf, descriptors_tfidf))

# Filter and scale numerical features between 0 and 1 
numerical_features = cleaned_data[['Average Rating', 'Release Year']]
scaler = MinMaxScaler()
scaled_numerical_features = scaler.fit_transform(numerical_features)

# Horizontally stack (add columns) the categorical and numerical features 
combined_features = hstack((categorical_features, scaled_numerical_features))

# Compute the cosine similarity matrix 
# Cosine similarity is used instead of Euclidean distance (typical for numerical features) and Jaccard index (typical for categorical features)
# due to its suitability for sparse, TF-IDF comparisons
similarity_matrix = cosine_similarity(combined_features, combined_features)

# ML-based album recommender function based on genre, descriptors, average rating, and release year 
def recommend_albums(album_name, no_recs=5):    
        # Gets the row index of the album that matches the entered album name 
        album_idx = cleaned_data[cleaned_data['Album'] == album_name].index[0]

        # Lists the similarity scores between that index and every other album in tuples & in order of albums 
        # E.g. (0, 0.9) means the user specified album has a cosine similarity 0.9 with album 1
        similarity_scores = list(enumerate(similarity_matrix[album_idx]))

        # Sorts the list of tuples based on their second item in descending order, skipping the user album itself 
        sorted_similar_albums = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:no_recs+1]

        # Loop through i tuples with the highest similarity and retrieve the album name, genre, and descriptors from 'cleaned_data' 
        recommended_albums = [
        {
            'Album': cleaned_data.iloc[i[0]]['Album'],
            'Genre': cleaned_data.iloc[i[0]]['Genres'],
            'Descriptors': cleaned_data.iloc[i[0]]['Descriptors']
        }
        for i in sorted_similar_albums]
        
        return recommended_albums

# Simpler album recommender function based on genres
def recommend_albums_genre_version(genre, no_recs=5):
    # Filter for albums that contain the input genre
    filtered_albums = cleaned_data[cleaned_data['Genres'].str.contains(genre, case=False, na=False)].copy()

    # Sort by rating and take top 'no_recs' albums
    top_albums_by_genre = filtered_albums.sort_values(by="Average Rating", ascending=False).head(no_recs)

    recommended_albums_by_genre = [
        {
            'Album': row['Album'],
            'Genre': row['Genres'],
            'Descriptors': row['Descriptors']
        }
        for _, row in top_albums_by_genre.iterrows()
    ]

    return recommended_albums_by_genre

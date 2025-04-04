import pandas as pd
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
from fuzzywuzzy import process
import streamlit as st

raw_data = pd.read_csv('rym_top_5000_all_time.csv')
cleaned_data = raw_data.dropna().copy()
cleaned_data['Release Year'] = cleaned_data['Release Date'].astype(str).str[-4:].astype(int)
cleaned_data.drop(['Release Date', 'Number of Ratings', 'Number of Reviews'], axis=1, inplace=True)

initialise_tfidf_genres = TfidfVectorizer()
genres_tfidf = initialise_tfidf_genres.fit_transform(cleaned_data['Genres'])

initialise_tfidf_descriptors = TfidfVectorizer()
descriptors_tfidf = initialise_tfidf_descriptors.fit_transform(cleaned_data['Descriptors'])

categorical_features = hstack((genres_tfidf, descriptors_tfidf))

numerical_features = cleaned_data[['Average Rating', 'Release Year']]
scaler = MinMaxScaler()
scaled_numerical_features = scaler.fit_transform(numerical_features)

combined_features = hstack((categorical_features, scaled_numerical_features))

similarity_matrix = cosine_similarity(combined_features, combined_features)

def recommend_albums(album_name, no_recs=5):    
        album_idx = cleaned_data[cleaned_data['Album'] == album_name].index[0]

        similarity_scores = list(enumerate(similarity_matrix[album_idx]))

        sorted_similar_albums = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:no_recs+1]

        recommended_albums = [
        {
            'Album': cleaned_data.iloc[i[0]]['Album'],
            'Genre': cleaned_data.iloc[i[0]]['Genres'],
            'Descriptors': cleaned_data.iloc[i[0]]['Descriptors']
        }
        for i in sorted_similar_albums]
        
        return recommended_albums

# def recommend_albums_genre_version(genre, no_recs=5):

#     filtered_albums = cleaned_data[cleaned_data['Genres'].str.contains(genre, case=False, na=False)].copy()

#     top_albums_by_genre = filtered_albums.sort_values(by="Average Rating", ascending=False).head(no_recs)

#     recommended_albums_by_genre = [
#         {
#             'Album': cleaned_data.iloc[i[0]]['Album'],
#             'Genre': cleaned_data.iloc[i[0]]['Genres'],
#             'Descriptors': cleaned_data.iloc[i[0]]['Descriptors']
#         }
#         for i in top_albums_by_genre]
    
#     return recommended_albums_by_genre

def recommend_albums_genre_version(genre, no_recs=5):
    # Filter for albums that contain the input genre
    filtered_albums = cleaned_data[cleaned_data['Genres'].str.contains(genre, case=False, na=False)].copy()

    # Sort by rating and take top N
    top_albums_by_genre = filtered_albums.sort_values(by="Average Rating", ascending=False).head(no_recs)

    # Build the recommendations list using the rows directly
    recommended_albums_by_genre = [
        {
            'Album': row['Album'],
            'Genre': row['Genres'],
            'Descriptors': row['Descriptors']
        }
        for _, row in top_albums_by_genre.iterrows()
    ]

    return recommended_albums_by_genre

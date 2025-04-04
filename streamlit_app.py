import pandas as pd
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
from fuzzywuzzy import process
import streamlit as st
from album_recommender import recommend_albums, recommend_albums_genre_version


cleaned_data = pd.read_csv('cleaned_data.csv')

st.title('Album Recommender :headphones:')
st.title('Want to find a new album that matches your favorite album\'s vibe?')
st.write('Enter an album name to get recommendations:')
st.write('Don\'t know where to start? Try OK Computer.')

album_input = st.text_input('Album Name')

# Checks if album_input is None or an empty string
if album_input:
    if album_input in cleaned_data['Album'].values:
        recs_ml = recommend_albums(album_input)
        st.write("Here are some recommendations based on your input:")
        st.dataframe(pd.DataFrame(recs_ml))
    else:
        closest_match, score = process.extractOne(album_input, cleaned_data['Album'].values)
        closest_match_input = st.selectbox(f'Did you mean {closest_match}?', ('Yes', 'No'))
        if closest_match_input == 'Yes': 
            album_input = closest_match
            recs_ml = recommend_albums(album_input)
            st.write("Here are some recommendations based on your input:")
            st.dataframe(pd.DataFrame(recs_ml))
        else:
            st.write('This exact album was not found in my dataset. However, I can still provide you with some recommendations if you enter an album genre.')
            genre_input = st.text_input('Genre')

            if genre_input: 
                if cleaned_data['Genres'].str.contains(genre_input, case=False, na=False).any():
                    recs_by_genre = recommend_albums_genre_version(genre_input)
                    st.write("Here are some recommendations based on your genre input:")
                    st.dataframe(pd.DataFrame(recs_by_genre))
                else:
                    closest_match, score = process.extractOne(genre_input, cleaned_data['Genres'].values)
                    closest_match_input = st.selectbox(
                        f'Did you mean {closest_match}?',
                        ('Yes', 'No'))
                    if closest_match_input == 'Yes': 
                        genre_input = closest_match
                        recs_by_genre = recommend_albums_genre_version(genre_input)
                        st.write("Here are some recommendations based on your genre input:")
                        st.dataframe(pd.DataFrame(recs_by_genre))





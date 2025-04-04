import pandas as pd
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
from fuzzywuzzy import process
import streamlit as st
from album_recommender import recommend_albums, recommend_albums_genre_version

# Import data
cleaned_data = pd.read_csv('cleaned_data.csv')

st.title('Album Recommender :headphones:')
st.title('Want to find a new album that matches your favorite album\'s vibe?')
st.write('Enter an album name to get recommendations:')
st.write('Don\'t know where to start? Try OK Computer.')

# Receives album name as input from the user
album_input = st.text_input('Album Name')

# Checks if album_input is None or an empty string
if album_input:
    
    # If the album exists in the dataset then provide recommendations using ML-based 'recommend_albums' function
    if album_input in cleaned_data['Album'].values:
        recs_ml = recommend_albums(album_input)
        st.write("Here are some recommendations based on your input:")
        st.dataframe(pd.DataFrame(recs_ml))
    
    # Perform fuzzy string matching in case of typos
    else:
        closest_match, score = process.extractOne(album_input, cleaned_data['Album'].values)
        closest_match_input = st.selectbox(f'Did you mean {closest_match}?', ('Yes', 'No'))
        if closest_match_input == 'Yes': 
            album_input = closest_match
            recs_ml = recommend_albums(album_input)
            st.write("Here are some recommendations based on your input:")
            st.dataframe(pd.DataFrame(recs_ml))
        
        # If no typo occured, then recommends albums based on genre using simpler 'recommend_albums_genre_version' function
        else:
            st.write('This exact album was not found in my dataset. However, I can still provide you with some recommendations if you enter an album genre.')
            genre_input = st.text_input('Genre')

            # Prevents dataframe from being automatically displayed until user provides input
            if genre_input: 
                
                # Checks for match between genre in 'Genres' and user input
                if cleaned_data['Genres'].str.contains(genre_input, case=False, na=False).any():
                    recs_by_genre = recommend_albums_genre_version(genre_input)
                    st.write('Here are some recommendations based on your genre input:')
                    st.dataframe(pd.DataFrame(recs_by_genre))

                # If no match exists, performs fuzzy string matching
                else:
                    closest_match, score = process.extractOne(genre_input, cleaned_data['Genres'].values)
                    closest_match_input = st.selectbox(
                        f'Did you mean {closest_match}?',
                        ('Yes', 'No'))
                    if closest_match_input == 'Yes': 
                        genre_input = closest_match
                        recs_by_genre = recommend_albums_genre_version(genre_input)
                        st.write('Here are some recommendations based on your genre input:')
                        st.dataframe(pd.DataFrame(recs_by_genre))
                    else: 
                        st.write('There are no recommendations for this genre at this time. Please try another input.')





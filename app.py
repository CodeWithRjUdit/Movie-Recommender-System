import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

# Load movie data
movies = pd.read_csv('movies.csv')

# Create a CountVectorizer to convert tags to vectors
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()

# Compute cosine similarity between movies
similarity = cosine_similarity(vectors)

# Fetch poster from TMDb API
def fetch_posters(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
        data = requests.get(url, timeout=5)
        data.raise_for_status()  # Raise an error for bad responses
        data = data.json()
        poster_path = data.get('poster_path')
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500/{poster_path}"
    except requests.exceptions.RequestException as e:
        print(f"Error fetching poster: {e}")
    return "https://via.placeholder.com/150"  # Placeholder image URL

# Recommend movies based on title
def recommend(movie):
    try:
        movie_index = movies[movies['title'] == movie].index[0]
        distances = similarity[movie_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

        recommended_movie_names = []
        recommended_movie_posters = []
        for i in movies_list:
            recommended_movie_id = movies.iloc[i[0]].id
            recommended_movie_names.append(movies.iloc[i[0]].title)
            recommended_movie_posters.append(fetch_posters(recommended_movie_id))

        return recommended_movie_names, recommended_movie_posters
    except IndexError:
        st.error("Movie not found in the database.")
        return [], []

# Streamlit app
st.title('Movie Recommender System')

# Dropdown: User selects a movie title from the dropdown menu
movie_title = st.selectbox("Choose a movie:", movies['title'].values)

# Recommend button
if st.button("Recommend"):
    names, posters = recommend(movie_title)

    if names:
        # Display recommendations in columns
        cols = st.columns(len(names))  # Adjust columns dynamically
        for idx, col in enumerate(cols):
            col.text(names[idx])
            col.image(posters[idx])
    else:
        st.write("No recommendations available.")

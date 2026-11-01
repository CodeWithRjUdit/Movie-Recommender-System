import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
)

DATA_PATH = Path(__file__).parent / "movies.csv"
TMDB_API_URL = "https://api.themoviedb.org/3/movie/{}"
TMDB_IMAGE_URL = "https://image.tmdb.org/t/p/w500{}"


@st.cache_resource(show_spinner="Loading the movie catalogue...")
def load_recommender():
    """Load the catalogue and cache its sparse text-vector representation."""
    if not DATA_PATH.exists():
        raise FileNotFoundError("The movies.csv file is missing from the repository.")

    movies = pd.read_csv(DATA_PATH)
    required_columns = {"id", "title", "tags"}
    missing_columns = required_columns.difference(movies.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"movies.csv is missing required columns: {missing}")

    movies = movies.copy()
    movies["title"] = movies["title"].fillna("").astype(str).str.strip()
    movies["tags"] = movies["tags"].fillna("").astype(str)
    movies = movies.loc[movies["title"] != ""].reset_index(drop=True)

    if movies.empty:
        raise ValueError("No valid movie titles were found in movies.csv.")

    vectorizer = CountVectorizer(
        max_features=5000,
        stop_words="english",
        dtype=np.float32,
    )
    vectors = vectorizer.fit_transform(movies["tags"])

    return movies, vectors


def get_recommendations(movie_index, movies, vectors, number_of_recommendations=5):
    """Find the most similar movies without creating a dense N x N matrix."""
    selected_vector = vectors[movie_index]

    if selected_vector.nnz == 0:
        return []

    similarity_scores = cosine_similarity(selected_vector, vectors).ravel()
    similarity_scores[movie_index] = -1

    recommended_indices = np.argsort(similarity_scores)[::-1]
    recommended_indices = recommended_indices[:number_of_recommendations]

    return recommended_indices.tolist()


def get_tmdb_api_key():
    """Read the optional TMDb key from Streamlit Secrets or an environment variable."""
    key = os.getenv("TMDB_API_KEY", "")
    try:
        key = st.secrets.get("TMDB_API_KEY", key)
    except Exception:
        pass
    return str(key).strip()


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def fetch_poster(movie_id, api_key):
    """Retrieve and cache a movie poster from TMDb."""
    if not api_key:
        return None

    try:
        response = requests.get(
            TMDB_API_URL.format(movie_id),
            params={"api_key": api_key, "language": "en-US"},
            timeout=8,
        )
        response.raise_for_status()
        poster_path = response.json().get("poster_path")

        if poster_path:
            return TMDB_IMAGE_URL.format(poster_path)
    except (requests.RequestException, ValueError):
        return None

    return None


st.title("🎬 Movie Recommender System")
st.write(
    "Select a movie you enjoy and discover five similar movies using "
    "content-based filtering."
)

try:
    movies, movie_vectors = load_recommender()
except Exception as error:
    st.error(f"Unable to load the movie catalogue: {error}")
    st.stop()

movie_titles = movies["title"].tolist()
selected_title = st.selectbox(
    "Choose a movie",
    options=movie_titles,
    help="Start typing to search the movie catalogue.",
)

if st.button("Recommend Movies", type="primary"):
    selected_index = movies.index[movies["title"] == selected_title][0]
    recommendation_indices = get_recommendations(
        selected_index,
        movies,
        movie_vectors,
    )

    if not recommendation_indices:
        st.warning(
            "This movie does not contain enough descriptive information "
            "to generate recommendations."
        )
    else:
        st.subheader(f"Movies similar to **{selected_title}**")

        api_key = get_tmdb_api_key()
        if not api_key:
            st.caption(
                "Poster images are optional. Add a TMDB_API_KEY in the app's "
                "Streamlit Secrets to display them."
            )

        columns = st.columns(len(recommendation_indices))

        for column, recommendation_index in zip(columns, recommendation_indices):
            recommended_movie = movies.iloc[recommendation_index]

            with column:
                poster_url = fetch_poster(recommended_movie["id"], api_key)
                if poster_url:
                    st.image(poster_url, use_container_width=True)
                else:
                    st.markdown(
                        "<div style='height:225px;display:flex;"
                        "align-items:center;justify-content:center;"
                        "background:#20242b;border-radius:8px;"
                        "font-size:48px;'>🎞️</div>",
                        unsafe_allow_html=True,
                    )
                st.write(str(recommended_movie["title"]))

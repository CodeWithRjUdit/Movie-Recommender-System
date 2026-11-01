import re
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import quote

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
CINEMETA_SEARCH_URL = (
    "https://v3-cinemeta.strem.io/catalog/movie/top/search={}.json"
)
WIKIPEDIA_SEARCH_URL = "https://en.wikipedia.org/w/rest.php/v1/search/page"


def normalize_title(title):
    """Normalize titles to make punctuation and subtitle differences less significant."""
    title = str(title).lower().strip()
    title = re.sub(r"\s*\([^)]*\)\s*$", "", title)
    return re.sub(r"[^a-z0-9]+", "", title)


def title_match_score(first, second):
    first_normalized = normalize_title(first)
    second_normalized = normalize_title(second)

    if not first_normalized or not second_normalized:
        return 0.0
    if first_normalized == second_normalized:
        return 1.0

    return SequenceMatcher(None, first_normalized, second_normalized).ratio()


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
    """Find similar movies without creating a dense all-movies similarity matrix."""
    selected_vector = vectors[movie_index]

    if selected_vector.nnz == 0:
        return []

    similarity_scores = cosine_similarity(selected_vector, vectors).ravel()
    similarity_scores[movie_index] = -1

    recommended_indices = np.argsort(similarity_scores)[::-1]
    recommended_indices = recommended_indices[:number_of_recommendations]

    return recommended_indices.tolist()


@st.cache_data(ttl=7 * 24 * 60 * 60, show_spinner=False)
def fetch_cinemeta_poster(movie_title):
    """Find a matching movie poster using the public, key-free Cinemeta catalogue."""
    try:
        search_url = CINEMETA_SEARCH_URL.format(quote(str(movie_title), safe=""))
        response = requests.get(
            search_url,
            headers={"User-Agent": "MovieRecommender/1.0"},
            timeout=10,
        )
        response.raise_for_status()
        metadata = response.json()

        candidates = metadata.get("metas", [])
        if not isinstance(candidates, list):
            return None

        best_match = None
        best_score = 0.0

        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue

            candidate_title = candidate.get("name", "")
            score = title_match_score(movie_title, candidate_title)

            if score > best_score and candidate.get("poster"):
                best_match = candidate
                best_score = score

        # Avoid showing artwork for an unrelated title.
        if best_match and best_score >= 0.82:
            poster_url = best_match.get("poster")
            if isinstance(poster_url, str) and poster_url.startswith("https://"):
                return poster_url

    except (requests.RequestException, ValueError, TypeError):
        pass

    return None


@st.cache_data(ttl=7 * 24 * 60 * 60, show_spinner=False)
def fetch_wikipedia_poster(movie_title):
    """Fallback: use the matching film article's lead image from Wikipedia."""
    try:
        response = requests.get(
            WIKIPEDIA_SEARCH_URL,
            params={"q": f"{movie_title} film", "limit": 8},
            headers={"User-Agent": "MovieRecommender/1.0 (poster lookup)"},
            timeout=10,
        )
        response.raise_for_status()
        pages = response.json().get("pages", [])

        best_page = None
        best_score = 0.0

        for page in pages:
            if not isinstance(page, dict):
                continue

            page_title = page.get("title", "")
            description = str(page.get("description") or "").lower()
            title_score = title_match_score(movie_title, page_title)

            is_film_article = (
                "film" in description
                or "movie" in description
                or "motion picture" in description
                or bool(re.search(r"\(.*film.*\)$", page_title, re.IGNORECASE))
            )

            if is_film_article and title_score > best_score:
                best_page = page
                best_score = title_score

        if best_page and best_score >= 0.82:
            thumbnail = best_page.get("thumbnail") or {}
            image_url = thumbnail.get("url")
            if isinstance(image_url, str):
                if image_url.startswith("//"):
                    image_url = "https:" + image_url
                if image_url.startswith("https://"):
                    return image_url

    except (requests.RequestException, ValueError, TypeError):
        pass

    return None


def fetch_poster(movie_title):
    """Try Cinemeta first, then Wikipedia. Neither source requires an API key."""
    poster_url = fetch_cinemeta_poster(movie_title)
    if poster_url:
        return poster_url

    return fetch_wikipedia_poster(movie_title)


st.title("🎬 Movie Recommender System")
st.write(
    "Choose a movie you enjoy and discover five similar movies using "
    "content-based filtering."
)
st.caption(
    "Movie posters are retrieved automatically from public metadata sources. "
    "No account or API key is required."
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

        columns = st.columns(len(recommendation_indices))

        for column, recommendation_index in zip(columns, recommendation_indices):
            recommended_movie = movies.iloc[recommendation_index]
            movie_title = str(recommended_movie["title"])

            with column:
                with st.spinner("Loading poster..."):
                    poster_url = fetch_poster(movie_title)

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
                    st.caption("Poster unavailable")

                st.write(movie_title)

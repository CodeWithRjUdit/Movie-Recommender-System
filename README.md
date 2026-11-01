# Movie Recommender System

A content-based movie recommendation app built with Python, Streamlit, and scikit-learn.

Choose a movie from the catalogue and the application recommends five similar titles by comparing the textual tags associated with each movie.

## How it works

1. The application loads the movie catalogue from `movies.csv`.
2. `CountVectorizer` converts the movie tags into sparse numerical feature vectors.
3. Cosine similarity measures how closely the selected movie matches the other movies.
4. The five highest-scoring movies (excluding the selected title) are displayed as recommendations.
5. Optional poster artwork is retrieved from The Movie Database (TMDb).

The app computes similarities only for the selected movie. It does not create and retain a dense all-pairs similarity matrix, which reduces memory use during deployment.

## Run locally

Python 3.10 or later is recommended.

```bash
git clone https://github.com/CodeWithRjUdit/Movie-Recommender-System.git
cd Movie-Recommender-System
python -m venv .venv
```

Activate the virtual environment, then install the dependencies:

**Windows**
```powershell
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

**macOS / Linux**
```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with the GitHub account that can access this repository.
2. Select **Create app**.
3. Choose repository `CodeWithRjUdit/Movie-Recommender-System`.
4. Select branch `main` and main file path `app.py`.
5. Deploy the app. Streamlit will provide a shareable URL for the project.

## Optional: enable movie posters

Recommendations work without a TMDb key. To display poster images, create or use a TMDb API key and add it in the app's **Settings → Secrets** in Streamlit Community Cloud:

```toml
TMDB_API_KEY = "your_tmdb_api_key"
```

For local development, you can put the same setting in `.streamlit/secrets.toml` (this file is excluded by `.gitignore`) or set the `TMDB_API_KEY` environment variable.

**Security:** Never commit API keys to GitHub. If a key was previously committed to a public repository, revoke or rotate it with the provider before using a replacement in Streamlit Secrets.

## Project files

- `app.py` — Streamlit interface and recommendation logic.
- `movies.csv` — movie catalogue and text tags used by the recommender.
- `requirements.txt` — Python dependencies.

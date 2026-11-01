# Movie Recommender System

A content-based movie recommendation app built with Python, Streamlit, and scikit-learn.

Choose a movie from the catalogue and the application recommends five similar titles by comparing the textual tags associated with each movie. Movie poster artwork is retrieved automatically from public metadata sources. No user account, API key, or paid subscription is required.

## How it works

1. The application loads the movie catalogue from `movies.csv`.
2. `CountVectorizer` converts the movie tags into sparse numerical feature vectors.
3. Cosine similarity measures how closely the selected movie matches the other movies.
4. The five highest-scoring movies (excluding the selected title) are displayed as recommendations.
5. Poster artwork is looked up using the public Cinemeta catalogue. If no sufficiently close title match is available there, the app tries the matching Wikipedia film article. Poster lookups are cached for seven days to reduce repeat requests.

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

No Streamlit Secrets are needed.

## Poster data sources

- [Cinemeta](https://v3-cinemeta.strem.io) — public movie catalogue and poster URLs, accessed without an API key.
- [Wikipedia REST API](https://www.mediawiki.org/wiki/API:REST_API/Reference) — fallback film-article search and lead-image thumbnails.

These are third-party services. Poster availability depends on whether the selected movie has a matching entry and image at one of these sources. A placeholder is shown when neither source returns a confident match.

## Project files

- `app.py` — Streamlit interface, recommendation logic, and poster lookup.
- `movies.csv` — movie catalogue and text tags used by the recommender.
- `requirements.txt` — Python dependencies.

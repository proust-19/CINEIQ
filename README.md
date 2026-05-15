# CineIQ 🎬
> Hybrid Movie Recommendation Engine combining Collaborative Filtering, Content-Based Filtering, and Sentiment Re-ranking.

## Problem
Content discovery on streaming platforms is opaque and biased toward promoted titles. CineIQ builds an open, explainable recommendation engine that combines multiple ML strategies.

## Architecture
Content-Based (TF-IDF) ──┐
                          ├──► Weighted Ensemble ──► Sentiment Re-ranker ──► Results
SVD Collaborative ────────┘

## Tech Stack
- ML: scikit-learn, Surprise (SVD), pandas, numpy
- NLP: VADER sentiment analysis
- API: FastAPI
- Dashboard: Streamlit + Plotly
- Dataset: MovieLens 1M

## How to Run
```bash
pip install -r requirements.txt

# Terminal 1 — API
cd api && uvicorn main:app --reload --port 8000

# Terminal 2 — Dashboard
cd dashboard && streamlit run app.py
```

## Models
Run the notebooks in order (01 → 05) to regenerate all model artifacts in `/models/`.

## Results
- SVD RMSE: 0.8861 (3-fold CV)
- Ensemble: Content (30%) + SVD (50%) + Popularity (20%)
- Sentiment re-ranking improves result quality using audience reception signals

## Dataset
MovieLens 1M — grouplens.org/datasets/movielens/1m/

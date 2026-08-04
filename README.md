# CineIQ 🎬

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Hybrid Movie Recommendation Engine combining Collaborative Filtering (SVD), Content-Based Filtering (TF-IDF), and Sentiment Re-ranking.

---

## Demo

📺 [Watch the demo video](https://drive.google.com/file/d/1pG4cxj4uCaGx4k4y1ur_9LR1Tl5vv7fd/view?usp=sharing)

---

## Problem

Content discovery on modern streaming platforms is opaque, biased toward promoted titles, and traps users in recommendation loops. CineIQ builds an open, explainable recommendation engine that combines multiple ML strategies to deliver personalized, interpretable suggestions.

## Deliverables

- **Hybrid Recommendation Engine**: Content-Based (TF-IDF + cosine similarity) + SVD Matrix Factorization + Popularity via weighted ensemble
- **Sentiment-Aware Re-Ranker**: Derives audience sentiment from ratings to re-rank recommendations
- **Interactive Dashboard**: Streamlit UI with real-time recommendations and genre visualization
- **REST API**: FastAPI serving `/recommend` and `/similar` endpoints

## System Architecture

```
User Input (liked movie + user ID)
           │
           ▼
┌──────────────────────┐
│  Content-Based       │  TF-IDF + Cosine Similarity (30% weight)
│  (genre features)    │
└──────────┬───────────┘
           │
┌──────────────────────┐
│  SVD Collaborative   │  Matrix Factorization (50% weight)
│  (user-item matrix)  │
└──────────┬───────────┘
           │
┌──────────────────────┐
│  Popularity          │  Global avg rating (20% weight)
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Weighted Ensemble   │  Normalized scores combined
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Sentiment Re-Ranker │  Audience reception signal (α = 0.3)
└──────────┬───────────┘
           │
           ▼
     Top-N Recommendations
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Framework | scikit-learn, Surprise (SVD), pandas, numpy |
| NLP | VADER-based sentiment proxy from ratings |
| API Server | FastAPI + uvicorn |
| Dashboard | Streamlit + Plotly |
| Dataset | MovieLens 1M (1M ratings, 3,883 movies, 6,040 users) |

## Project Structure

```
CINEIQ/
├── api/                    # FastAPI server
│   └── main.py             # /recommend and /similar endpoints
├── dashboard/              # Streamlit web app
│   └── app.py              # Interactive recommendation UI
├── notebooks/              # Jupyter notebooks (run in order)
│   ├── 01_eda.ipynb        # Exploratory Data Analysis
│   ├── 02_content_based.ipynb    # TF-IDF + Cosine Similarity
│   ├── 03_collaborative_svd.ipynb # SVD Matrix Factorization
│   ├── 04_ensemble.ipynb   # Weighted Ensemble
│   └── 05_sentiment.ipynb  # Sentiment Re-ranking
├── data/
│   ├── raw/                # MovieLens 1M original files (not tracked)
│   └── processed/          # Cleaned CSV files
├── models/                 # Serialized model pickles (not tracked)
├── requirements.txt
└── README.md
```

## How to Run

### 1. Setup

```bash
git clone https://github.com/proust-19/CINEIQ.git
cd CINEIQ
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download the Dataset

```bash
# Download MovieLens 1M
wget https://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-1m.zip -d data/raw/
```

### 3. Run Notebooks (Optional — for retraining)

Run notebooks in order from `notebooks/`:
```bash
jupyter notebook notebooks/01_eda.ipynb
# ... repeat for 02 through 05
```

### 4. Start the API Server

```bash
cd api && uvicorn main:app --reload --port 8000
```

### 5. Launch the Dashboard

```bash
cd dashboard && streamlit run app.py
```

The dashboard opens at `http://localhost:8501` and the API at `http://localhost:8000`.

## Results

| Metric | Value |
|--------|-------|
| SVD RMSE (3-fold CV) | 0.8861 |
| Ensemble weights | Content: 30%, SVD: 50%, Popularity: 20% |
| Sentiment α | 0.3 (30% sentiment influence in final score) |

## Future Work / Upgradation

1. **Scale to MovieLens 25M** — Retrain all models on the full 25M rating dataset to improve coverage and accuracy for niche and long-tail movies
2. **Deep Learning Models** — Replace SVD with neural collaborative filtering (NCF) or two-tower models for better feature interactions
3. **Review Text Sentiment** — Integrate actual review text from IMDB/TMDB using DistilBERT or RoBERTa for genuine sentiment signals instead of rating proxies
4. **Real-Time Updates** — Implement incremental model updates (online learning) so recommendations evolve immediately with new ratings
5. **Cold-Start Handling** — Use side information (director, cast, release year) for content features to recommend new/unrated movies effectively
6. **A/B Testing Framework** — Add an experimentation layer to compare ensemble variants and measure online engagement metrics

## Dataset

[MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) — Stable benchmark dataset with 1M ratings from 6,040 users on 3,883 movies.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/recommend` | Hybrid recommendations for a user + movie |
| POST | `/similar` | Content-based similar movies |

### Example Request

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "movie_title": "Toy Story (1995)", "n": 5}'
```

## License
MIT
## Authors
* **Purshotam Kumar** - [proust-19](https://github.com/proust-19)
* **Uday Kumar** - [udaykumar-01](https://github.com/udaykumar-01)

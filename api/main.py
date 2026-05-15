from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import json
import sys
sys.path.append('..')

app = FastAPI(title="CineIQ", description="Hybrid Movie Recommendation Engine")

# Load all artifacts
movies = pd.read_csv('../data/processed/movies.csv')
ratings = pd.read_csv('../data/processed/merged.csv')
cosine_sim = pickle.load(open('../models/cosine_sim.pkl', 'rb'))
indices = pickle.load(open('../models/indices.pkl', 'rb'))
svd = pickle.load(open('../models/svd_model.pkl', 'rb'))
movie_sentiment = pd.read_csv('../data/processed/movie_sentiment.csv')
weights = json.load(open('../models/ensemble_weights.json'))

# Popularity scores
popularity = ratings.groupby('movieId')['rating'].mean().to_dict()

# --- Core functions ---
def content_scores(title, n=50):
    if title not in indices:
        return {}
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
    return {movies.iloc[i[0]]['movieId']: i[1] for i in sim_scores}

def svd_scores(user_id, movie_ids):
    return {mid: svd.predict(user_id, mid).est for mid in movie_ids}

def popularity_scores(movie_ids):
    max_r = max(popularity.values())
    return {mid: popularity.get(mid, 0) / max_r for mid in movie_ids}

def ensemble_recommend(user_id, liked_movie_title, n=10):
    candidates = content_scores(liked_movie_title, n=50)
    if not candidates:
        return None
    movie_ids = list(candidates.keys())
    max_c = max(candidates.values()) or 1
    c_scores = {mid: v/max_c for mid, v in candidates.items()}
    s_raw = svd_scores(user_id, movie_ids)
    max_s = max(s_raw.values()) or 1
    s_scores = {mid: v/max_s for mid, v in s_raw.items()}
    p_scores = popularity_scores(movie_ids)
    final = {}
    for mid in movie_ids:
        final[mid] = (weights['w_content'] * c_scores.get(mid, 0) +
                      weights['w_svd']     * s_scores.get(mid, 0) +
                      weights['w_pop']     * p_scores.get(mid, 0))
    top_ids = sorted(final, key=final.get, reverse=True)[:n]
    result = movies[movies['movieId'].isin(top_ids)][['movieId','title','genres']].copy()
    result['score'] = result['movieId'].map(final)
    return result.sort_values('score', ascending=False)

def sentiment_rerank(df, alpha=0.3):
    merged = df.merge(
        movie_sentiment[['movieId','sentiment_score']], on='movieId', how='left'
    )
    merged['sentiment_score'] = merged['sentiment_score'].fillna(0.5)
    merged['final_score'] = (1 - alpha) * merged['score'] + alpha * merged['sentiment_score']
    return merged.sort_values('final_score', ascending=False)

# --- Request schemas ---
class RecommendRequest(BaseModel):
    user_id: int
    movie_title: str
    n: int = 10

class SimilarRequest(BaseModel):
    movie_title: str
    n: int = 10

# --- Endpoints ---
@app.get("/")
def root():
    return {"message": "CineIQ API is running"}

@app.post("/recommend")
def recommend(req: RecommendRequest):
    result = ensemble_recommend(req.user_id, req.movie_title, req.n)
    if result is None:
        return {"error": f"Movie '{req.movie_title}' not found"}
    reranked = sentiment_rerank(result)
    return reranked[['title','genres','final_score']].to_dict(orient='records')

@app.post("/similar")
def similar(req: SimilarRequest):
    scores = content_scores(req.movie_title, n=req.n)
    if not scores:
        return {"error": f"Movie '{req.movie_title}' not found"}
    top_ids = sorted(scores, key=scores.get, reverse=True)[:req.n]
    result = movies[movies['movieId'].isin(top_ids)][['title','genres']].copy()
    return result.to_dict(orient='records')
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import pickle
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.recommender import (
    load_movies,
    load_ratings,
    load_content_index,
    load_svd_model,
    load_weights,
    load_quality_signal,
    load_popularity,
    content_scores,
    svd_scores,
    popularity_scores,
    ensemble_recommend,
    quality_signal_rerank,
)

app = FastAPI(title="CineIQ", description="Hybrid Movie Recommendation Engine")

# Load all artifacts at startup
movies = load_movies()
ratings = load_ratings()
content_index = load_content_index()
svd_model = load_svd_model()
weights = load_weights()
quality_signal = load_quality_signal()
popularity = load_popularity(ratings)

title_to_id = dict(zip(movies["title"], movies["movieId"]))
max_user_id = int(ratings["userId"].max())


# --- Explainability ---


def add_explanations(
    result: pd.DataFrame, user_id: int, liked_movie_title: str
) -> pd.DataFrame:
    """Add human-readable explanations to each recommendation."""
    user_ratings = ratings[ratings["userId"] == user_id]
    liked_movie_id = title_to_id.get(liked_movie_title)

    explanations = []
    for _, movie in result.iterrows():
        movie_id = movie["movieId"]
        movie_info = movies[movies["movieId"] == movie_id]
        if movie_info.empty:
            explanations.append(["Recommended by hybrid model"])
            continue
        movie_info = movie_info.iloc[0]

        reasons = []

        # 1. Genre similarity to liked movie
        if liked_movie_id is not None:
            liked_info = movies[movies["movieId"] == liked_movie_id]
            if not liked_info.empty:
                liked_info = liked_info.iloc[0]
                movie_genres = set(str(movie_info["genres"]).split("|"))
                liked_genres = set(str(liked_info["genres"]).split("|"))
                overlap = movie_genres & liked_genres
                if overlap:
                    reasons.append(f"Similar genres: {', '.join(list(overlap)[:3])}")

        # 2. User genre preference
        if len(user_ratings) > 0 and "genres" in user_ratings.columns:
            user_genres = set(
                g
                for genres in user_ratings["genres"].dropna()
                for g in str(genres).split("|")
            )
            movie_genres = set(str(movie_info["genres"]).split("|"))
            match = movie_genres & user_genres
            if match:
                reasons.append(f"Matches your taste: {', '.join(list(match)[:3])}")

        # 3. Release era
        title_str = str(movie_info["title"])
        if "(" in title_str and ")" in title_str:
            year = title_str.split("(")[-1].split(")")[0]
            if len(year) == 4 and year.isdigit():
                reasons.append(f"Released in {year}")

        # 4. Quality signal
        if movie_id in quality_signal.index:
            qs = quality_signal.loc[movie_id, "quality_score"]
            if qs > 0.7:
                reasons.append(f"Well-received (quality: {qs:.2f})")
            elif qs < 0.3:
                reasons.append(f"Hidden gem (quality: {qs:.2f})")

        explanations.append(reasons if reasons else ["Recommended by hybrid model"])

    result = result.copy()
    result["explainability"] = explanations
    return result


# --- Request schemas ---


class RecommendRequest(BaseModel):
    user_id: int = Field(
        ..., ge=1, le=max_user_id, description=f"User ID (1-{max_user_id})"
    )
    movie_title: str = Field(
        ..., min_length=1, description="Movie title (e.g., 'Toy Story (1995)')"
    )
    n: int = Field(10, ge=1, le=50, description="Number of recommendations (1-50)")
    include_explanations: bool = Field(
        False, description="Include human-readable explanations"
    )


class SimilarRequest(BaseModel):
    movie_title: str = Field(..., min_length=1, description="Movie title")
    n: int = Field(10, ge=1, le=50, description="Number of similar movies (1-50)")


# --- Endpoints ---


@app.get("/")
def root():
    return {
        "message": "CineIQ API is running",
        "dataset": "MovieLens 20M",
        "movies": len(movies),
        "ratings": len(ratings),
        "users": max_user_id,
    }


@app.post("/recommend")
def recommend(req: RecommendRequest):
    result = ensemble_recommend(
        user_id=req.user_id,
        liked_movie_title=req.movie_title,
        content_index=content_index,
        movies=movies,
        ratings=ratings,
        svd_model=svd_model,
        popularity=popularity,
        weights=weights,
        n=req.n,
    )
    if result is None or result.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Movie '{req.movie_title}' not found or no recommendations available",
        )

    reranked = quality_signal_rerank(result, quality_signal)
    response_data = reranked[["movieId", "title", "genres", "final_score"]].to_dict(
        orient="records"
    )

    if req.include_explanations:
        result_with_exp = add_explanations(result, req.user_id, req.movie_title)
        for i, record in enumerate(response_data):
            mid = record["movieId"]
            exp_row = result_with_exp[result_with_exp["movieId"] == mid]
            if not exp_row.empty and "explainability" in exp_row.columns:
                response_data[i]["explainability"] = exp_row.iloc[0]["explainability"]

    return response_data


@app.post("/similar")
def similar(req: SimilarRequest):
    scores = content_scores(req.movie_title, content_index, movies, n=req.n)
    if not scores:
        raise HTTPException(
            status_code=404,
            detail=f"Movie '{req.movie_title}' not found",
        )
    top_ids = sorted(scores, key=scores.get, reverse=True)[: req.n]
    result = movies[movies["movieId"].isin(top_ids)][
        ["movieId", "title", "genres"]
    ].copy()
    result["similarity_score"] = result["movieId"].map(scores)
    return result.sort_values("similarity_score", ascending=False).to_dict(
        orient="records"
    )

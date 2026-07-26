"""
CineIQ — Shared recommendation functions.

Used by notebooks, API, and tests to eliminate code duplication.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def load_movies() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "movies.csv")


def load_ratings() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "merged.csv")


def load_content_index() -> dict:
    """Load precomputed top-K content similarity index.

    Returns: {movieId: [(similar_movieId, score), ...]}
    """
    with open(MODELS_DIR / "content_index.json") as f:
        return json.load(f)


def load_svd_model():
    import pickle

    with open(MODELS_DIR / "svd_model.pkl", "rb") as f:
        return pickle.load(f)


def load_weights() -> dict:
    with open(MODELS_DIR / "ensemble_weights.json") as f:
        return json.load(f)


def load_quality_signal() -> pd.DataFrame:
    path = DATA_DIR / "movie_quality_signal.csv"
    if not path.exists():
        # Fallback to old sentiment file if quality signal not yet generated
        path = DATA_DIR / "movie_sentiment.csv"
    df = pd.read_csv(path)
    # Handle both old (sentiment_score) and new (quality_score) column names
    if "quality_score" not in df.columns and "sentiment_score" in df.columns:
        df = df.rename(columns={"sentiment_score": "quality_score"})
    return df.set_index("movieId")


def load_popularity(ratings: pd.DataFrame) -> dict:
    """Bayesian average popularity score per movie."""
    C = ratings["rating"].mean()
    stats = ratings.groupby("movieId")["rating"].agg(["mean", "count"])
    stats.columns = ["avg_rating", "num_ratings"]
    bayesian = (stats["avg_rating"] * stats["num_ratings"] + C * 50) / (
        stats["num_ratings"] + 50
    )
    return bayesian.to_dict()


def normalize_title(title: str) -> str:
    """Normalize title to handle 'The Matrix (1999)' vs 'Matrix, The (1999)'.

    MovieLens 20M uses 'Title, The (Year)' format.
    Users often type 'The Title (Year)'.
    """
    # Already in correct format
    return title


def find_movie_id(title: str, movies: pd.DataFrame) -> int | None:
    """Find movieId by title, handling both 'The Matrix (1999)' and 'Matrix, The (1999)'."""
    title_to_id = dict(zip(movies["title"], movies["movieId"]))

    # Direct match
    if title in title_to_id:
        return title_to_id[title]

    # Try swapping article: "The Matrix (1999)" → "Matrix, The (1999)"
    for prefix in ["The ", "A ", "An "]:
        if title.startswith(prefix):
            core = title[len(prefix) :]
            swapped = (
                f"{core[:-1]}, {prefix.strip()} {core[-1]}"
                if core.endswith(")")
                else title
            )
            # More robust: "The Matrix (1999)" → "Matrix, The (1999)"
            parts = title.rsplit(" (", 1)
            if len(parts) == 2:
                movie_name = parts[0]
                year = parts[1]
                if movie_name.startswith(prefix):
                    core_name = movie_name[len(prefix) :]
                    swapped = f"{core_name}, {prefix.strip()} ({year}"
                    if swapped in title_to_id:
                        return title_to_id[swapped]

    # Case-insensitive fallback
    title_lower = title.lower()
    for t, mid in title_to_id.items():
        if t.lower() == title_lower:
            return mid

    return None


def content_scores(
    title: str, content_index: dict, movies: pd.DataFrame, n: int = 50
) -> dict:
    """Get top-N content-based scores for a movie title using precomputed index.

    Returns: {movieId: score}
    """
    movie_id = find_movie_id(title, movies)
    if movie_id is None:
        return {}
    movie_id_str = str(movie_id)
    if movie_id_str not in content_index:
        return {}
    neighbors = content_index[movie_id_str][:n]
    return {int(mid): score for mid, score in neighbors}


def svd_scores(user_id: int, movie_ids: list, svd_model) -> dict:
    """Get SVD predicted ratings for a user and list of movie IDs."""
    return {mid: svd_model.predict(user_id, mid).est for mid in movie_ids}


def popularity_scores(movie_ids: list, popularity: dict) -> dict:
    """Normalized popularity scores for given movie IDs."""
    max_r = max(popularity.values()) if popularity else 1
    return {mid: popularity.get(mid, 0) / max_r for mid in movie_ids}


def ensemble_recommend(
    user_id: int,
    liked_movie_title: str,
    content_index: dict,
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    svd_model,
    popularity: dict,
    weights: dict,
    n: int = 10,
) -> pd.DataFrame | None:
    """Hybrid ensemble recommendation.

    Combines content-based, SVD, and popularity via weighted sum.
    """
    candidates = content_scores(liked_movie_title, content_index, movies, n=50)
    if not candidates:
        return None

    movie_ids = list(candidates.keys())

    # Normalize content scores
    max_c = max(candidates.values()) or 1
    c_scores = {mid: v / max_c for mid, v in candidates.items()}

    # SVD scores
    s_raw = svd_scores(user_id, movie_ids, svd_model)
    max_s = max(s_raw.values()) or 1
    s_scores = {mid: v / max_s for mid, v in s_raw.items()}

    # Popularity scores
    p_scores = popularity_scores(movie_ids, popularity)

    # Weighted combination
    final = {}
    for mid in movie_ids:
        final[mid] = (
            weights["w_content"] * c_scores.get(mid, 0)
            + weights["w_svd"] * s_scores.get(mid, 0)
            + weights["w_pop"] * p_scores.get(mid, 0)
        )

    top_ids = sorted(final, key=final.get, reverse=True)[:n]
    result = movies[movies["movieId"].isin(top_ids)][
        ["movieId", "title", "genres"]
    ].copy()
    result["score"] = result["movieId"].map(final)
    return result.sort_values("score", ascending=False)


def quality_signal_rerank(
    df: pd.DataFrame, quality_signal: pd.DataFrame, alpha: float = 0.3
) -> pd.DataFrame:
    """Rerank recommendations using rating-based quality signal."""
    merged = df.merge(
        quality_signal[["quality_score"]],
        left_on="movieId",
        right_index=True,
        how="left",
    )
    merged["quality_score"] = merged["quality_score"].fillna(0.5)
    merged["final_score"] = (1 - alpha) * merged["score"] + alpha * merged[
        "quality_score"
    ]
    return merged.sort_values("final_score", ascending=False)

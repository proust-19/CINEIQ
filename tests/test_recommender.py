"""Tests for CineIQ recommendation system."""

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


class TestDataLoading:
    def test_load_movies(self):
        movies = load_movies()
        assert len(movies) > 0
        assert "movieId" in movies.columns
        assert "title" in movies.columns
        assert "genres" in movies.columns

    def test_load_ratings(self):
        ratings = load_ratings()
        assert len(ratings) > 0
        assert "userId" in ratings.columns
        assert "movieId" in ratings.columns
        assert "rating" in ratings.columns

    def test_load_content_index(self):
        index = load_content_index()
        assert len(index) > 0
        # Each entry should be a list of [movieId, score] pairs
        first_key = list(index.keys())[0]
        assert isinstance(index[first_key], list)
        assert len(index[first_key]) > 0
        assert len(index[first_key][0]) == 2

    def test_load_svd_model(self):
        svd = load_svd_model()
        assert hasattr(svd, "predict")

    def test_load_weights(self):
        weights = load_weights()
        assert "w_content" in weights
        assert "w_svd" in weights
        assert "w_pop" in weights
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_load_quality_signal(self):
        qs = load_quality_signal()
        assert len(qs) > 0
        assert "quality_score" in qs.columns


class TestContentScores:
    def test_valid_movie(self):
        movies = load_movies()
        index = load_content_index()
        scores = content_scores("Toy Story (1995)", index, movies, n=5)
        assert len(scores) > 0
        assert all(isinstance(mid, int) for mid in scores.keys())
        assert all(0 <= s <= 1 for s in scores.values())

    def test_invalid_movie(self):
        movies = load_movies()
        index = load_content_index()
        scores = content_scores("Nonexistent Movie (9999)", index, movies, n=5)
        assert scores == {}


class TestEnsembleRecommend:
    def test_returns_dataframe(self):
        movies = load_movies()
        ratings = load_ratings()
        index = load_content_index()
        svd = load_svd_model()
        weights = load_weights()
        pop = load_popularity(ratings)

        result = ensemble_recommend(
            user_id=1,
            liked_movie_title="Toy Story (1995)",
            content_index=index,
            movies=movies,
            ratings=ratings,
            svd_model=svd,
            popularity=pop,
            weights=weights,
            n=5,
        )
        assert result is not None
        assert len(result) <= 5
        assert "score" in result.columns
        assert "title" in result.columns

    def test_invalid_movie_returns_none(self):
        movies = load_movies()
        ratings = load_ratings()
        index = load_content_index()
        svd = load_svd_model()
        weights = load_weights()
        pop = load_popularity(ratings)

        result = ensemble_recommend(
            user_id=1,
            liked_movie_title="Nonexistent Movie (9999)",
            content_index=index,
            movies=movies,
            ratings=ratings,
            svd_model=svd,
            popularity=pop,
            weights=weights,
            n=5,
        )
        assert result is None


class TestQualitySignalRerank:
    def test_rerank_adds_final_score(self):
        import pandas as pd

        movies = load_movies()
        ratings = load_ratings()
        index = load_content_index()
        svd = load_svd_model()
        weights = load_weights()
        pop = load_popularity(ratings)
        qs = load_quality_signal()

        result = ensemble_recommend(
            user_id=1,
            liked_movie_title="Toy Story (1995)",
            content_index=index,
            movies=movies,
            ratings=ratings,
            svd_model=svd,
            popularity=pop,
            weights=weights,
            n=5,
        )
        reranked = quality_signal_rerank(result, qs)
        assert "final_score" in reranked.columns
        assert "quality_score" in reranked.columns
        assert len(reranked) == len(result)

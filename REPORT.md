# CineIQ: Hybrid Movie Recommendation Engine

**Author:** Purshotam Kumar  
**Date:** May 2026  
**Code:** [github.com/proust-19/CINEIQ](https://github.com/proust-19/CINEIQ)

---

## 1. Abstract

Content discovery on streaming platforms is opaque, biased toward promoted titles, and traps users in recommendation loops. CineIQ addresses this with an open, explainable hybrid movie recommendation engine combining content-based filtering (TF-IDF + cosine similarity), collaborative filtering (SVD matrix factorization), and sentiment-aware re-ranking -- served via a FastAPI REST API and an interactive Streamlit dashboard.

## 2. Dataset

**MovieLens 1M** (grouplens.org/datasets/movielens/1m/): 1,000,209 ratings from 6,040 users on 3,883 movies across 18 genres, rated 1-5. Sentiment is proxied from rating values since review text is not available.

## 3. System Architecture

**Content-Based Filtering:** Genre labels are TF-IDF vectorized (English stop-word removal), producing a 3883 x 3883 cosine similarity matrix. Given a seed movie, the top-50 most similar candidates are retrieved.

**Collaborative Filtering (SVD):** An SVD model with 100 latent factors is trained for 20 epochs on the user-item matrix using the Surprise library. 3-fold cross-validation yields an RMSE of **0.8861**.

**Weighted Ensemble:** Scores from SVD (50%), Content (30%), and Popularity (20%) are min-max normalized and combined via weighted sum.

**Sentiment Re-Ranking:** Since ML-1M lacks review text, a proxy is derived (rating >= 4 = +1, <= 2 = -1, 3 = 0), aggregated per movie with log(review_count) confidence weighting, normalized to [0,1]. Final score = 0.7 x ensemble + 0.3 x sentiment.

**API & Dashboard:** FastAPI serves `/recommend` and `/similar` endpoints. Streamlit dashboard visualizes results with Plotly bar charts and genre distribution pie charts.

## 4. Implementation Pipeline

| Step | Notebook | Output |
|------|----------|--------|
| EDA | `01_eda.ipynb` | Statistics, visualizations, merged CSV |
| Content Model | `02_content_based.ipynb` | `cosine_sim.pkl`, `indices.pkl` |
| SVD Model | `03_collaborative_svd.ipynb` | `svd_model.pkl`, RMSE = 0.8861 |
| Ensemble | `04_ensemble.ipynb` | `ensemble_weights.json` |
| Sentiment | `05_sentiment.ipynb` | `sentiment.pkl`, `movie_sentiment.csv` |

Notebooks run sequentially; each saves artifacts consumed by the next stage and the API.

## 5. Results

**SVD 3-fold CV:** Mean RMSE = 0.8861 (folds: 0.8862, 0.8859, 0.8861).

**Qualitative (user=1, seed="Toy Story (1995)"):** Top results include Toy Story 2 (0.936), Wrong Trousers (0.922), Bug's Life (0.918), Chicken Run (0.912), Iron Giant (0.897). Sentiment re-ranking further adjusts rankings by audience reception.

## 6. Future Work / Upgradation

- **Scale to MovieLens 25M** -- retrain all models on 25x larger data for improved niche coverage
- **Deep Learning** -- replace SVD with Neural Collaborative Filtering or two-tower models
- **Review Text Sentiment** -- integrate actual IMDB reviews with DistilBERT/RoBERTa instead of rating proxies
- **Online Learning** -- incremental FTRL or streaming SVD for real-time updates
- **Cold-Start** -- use director, cast, plot embeddings (Sentence-BERT) for richer content features
- **A/B Testing** -- experimentation layer to compare ensemble variants on engagement metrics

## 7. Conclusion

CineIQ demonstrates a modular, production-ready hybrid recommendation system achieving SVD RMSE of 0.8861 on MovieLens 1M. The weighted ensemble balances collaborative, content, and popularity signals, while sentiment re-ranking incorporates audience reception. The API-first design enables easy integration, and the proposed upgradation roadmap provides clear pathways for scaling and enhancement.

---

*Report generated for CINEIQ project submission -- May 2026*

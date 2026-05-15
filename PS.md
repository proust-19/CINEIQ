# CINEIQ



#### PROBLEM STATEMENT



Content discovery on modern streaming platforms is opaque, biased toward

promoted titles, and traps users in recommendation loops. There is a need

for an open, explainable movie recommendation engine that combines

multiple ML strategies to deliver personalized, interpretable suggestions that

evolve with user taste over time.



#### DELIVERABLES


. Hybrid Recommendation Engine: Combines collaborative filtering, content-

based filtering (TF-IDF + cosine similarity), and SVD-based matrix factorization

via a weighted ensemble

. Sentiment-Aware Re-Ranker: Uses VADER/DistilBERT on user reviews to re-rank

recommendations based on real audience reception signals

. User Taste Dashboard: Streamlit interface visualizing genre radar charts,

decade preferences, and director/actor affinities from rating history

. Explainability Layer: Every recommendation surfaces a human-readable reason

using LIME or rule-based templates



#### DATASETS:

. MovieLens 25M - grouplens.org/datasets/movielens/25m

. TMDB Metadata (Kaggle) - cast, genres, keywords for 45K movies

. IMDB 50K Reviews (Kaggle) - for sentiment model training



#### TECH STACK:

. ML: Python, scikit-learn, Surprise (SVD), Pandas, NumPy

. NLP: VADER / HuggingFace DistilBERT

. Serving: FastAPI (/recommend and /similar endpoints)

. Dashboard: Streamlit + Plotly

. Tracking: MLflow for experiment logging






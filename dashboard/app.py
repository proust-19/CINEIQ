import streamlit as st
import requests
import pandas as pd
import plotly.express as px

API_URL = "http://localhost:8000"

st.set_page_config(page_title="CineIQ", page_icon="🎬", layout="wide")

st.title("🎬 CineIQ")
st.caption(
    "Hybrid Movie Recommendation Engine — Content (Genome) + Collaborative (SVD) + Popularity"
)

st.sidebar.header("Your Preferences")
user_id = st.sidebar.number_input("User ID", min_value=1, max_value=6040, value=1)
movie_title = st.sidebar.text_input("Movie you liked", value="Toy Story (1995)")
n = st.sidebar.slider("Number of recommendations", 5, 20, 10)

if st.sidebar.button("Get Recommendations"):
    with st.spinner("Finding your movies..."):
        res = requests.post(
            f"{API_URL}/recommend",
            json={
                "user_id": int(user_id),
                "movie_title": movie_title,
                "n": n,
            },
        )

    if res.status_code == 200:
        data = res.json()

        if isinstance(data, dict) and "error" in data:
            st.error(data["error"])
        elif isinstance(data, dict) and "detail" in data:
            st.error(data["detail"])
        else:
            df = pd.DataFrame(data)

            st.subheader(f"Top {n} Recommendations for User {user_id}")

            # Main table
            st.dataframe(
                df[["title", "genres", "final_score"]].rename(
                    columns={
                        "title": "Movie",
                        "genres": "Genres",
                        "final_score": "Score",
                    }
                ),
                use_container_width=True,
            )

            # Bar chart
            fig = px.bar(
                df,
                x="final_score",
                y="title",
                orientation="h",
                color="final_score",
                color_continuous_scale="Viridis",
                labels={"final_score": "Score", "title": "Movie"},
                title="Recommendation Scores",
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

            # Genre breakdown
            st.subheader("Genre Breakdown")
            all_genres = []
            for g in df["genres"]:
                all_genres.extend(str(g).split("|"))
            genre_df = pd.Series(all_genres).value_counts().reset_index()
            genre_df.columns = ["Genre", "Count"]
            fig2 = px.pie(
                genre_df, values="Count", names="Genre", title="Genre Distribution"
            )
            st.plotly_chart(fig2, use_container_width=True)

            # Explanations
            if "explainability" in df.columns:
                st.subheader("Why These Movies?")
                for _, row in df.iterrows():
                    with st.expander(f"**{row['title']}**"):
                        for reason in row["explainability"]:
                            st.write(f"• {reason}")
    else:
        st.error(
            f"API error (status {res.status_code}) — make sure FastAPI is running on port 8000"
        )

st.divider()
st.subheader("Find Similar Movies")
similar_title = st.text_input("Movie title", value="Pulp Fiction (1994)")
if st.button("Find Similar"):
    res = requests.post(
        f"{API_URL}/similar",
        json={"movie_title": similar_title, "n": 8},
    )
    if res.status_code == 200:
        data = res.json()
        if isinstance(data, dict) and "detail" in data:
            st.error(data["detail"])
        else:
            st.dataframe(pd.DataFrame(data), use_container_width=True)
    else:
        st.error(f"API error (status {res.status_code})")

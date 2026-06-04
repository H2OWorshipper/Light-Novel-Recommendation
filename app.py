import streamlit as st
import pandas as pd

from main import (
    compute_metrics,
    generate_explanations,
    get_decision_path_text,
    load_and_clean_data,
    create_features,
    recommend,
    rerank_results,
    send_to_google_sheets,
    get_explainable_feature_importance
)

st.title("📚 Light Novel Recommendation System")

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    df = load_and_clean_data("novels.csv")
    df.columns = df.columns.str.lower().str.strip()
    X, titles, title_to_idx, feature_objs = create_features(df)
    return df, X, titles, title_to_idx, feature_objs

df, X, titles, title_to_idx, feature_objs = load_data()

# -----------------------------
# User Input
# -----------------------------
st.write("### Select novels you like:")
liked_titles = st.multiselect("Choose novels you like", titles)

st.write("### Select novels you dislike (optional):")
disliked_titles = st.multiselect("Choose novels you dislike", titles)

# -----------------------------
# Generate Recommendations
# -----------------------------
if st.button("Get Recommendations"):
    if len(liked_titles) < 1:
        st.warning("Please select at least one novel.")
    else:
        recommendations, model, _, _ = recommend(
            liked_titles, X, titles, title_to_idx, df, disliked_titles=disliked_titles
        )

        # Store in session_state
        st.session_state["recommendations"] = recommendations
        st.session_state["model"] = model
        st.session_state["liked_titles"] = liked_titles

# -----------------------------
# If recommendations exist → display everything
# -----------------------------
if "recommendations" in st.session_state:

    recommendations = st.session_state["recommendations"]
    model = st.session_state["model"]
    liked_titles = st.session_state["liked_titles"]

    # Re-rank + explanations
    recommendations = rerank_results(recommendations, df, title_to_idx)
    explanations = generate_explanations(df, recommendations, liked_titles, title_to_idx)

    # -----------------------------
    # Top 10
    # -----------------------------
    st.write("## 📌 Top Recommendations")

    for i, (title, score) in enumerate(recommendations[:3], start=1):
        row = df.iloc[title_to_idx[title]]
        img = row.get("image", "")

        st.markdown(f"""
        <div style="padding:20px; border-radius:15px; background:#1f2937; margin-bottom:20px;">
            <h2>#{i} - {title}</h2>
            <h1>{score}</h1>
            {"<img src='" + img + "' width='200'>" if img else ""}
            <p><b>Author:</b> {row.get('authors', 'Unknown')}</p>
            <p>{' | '.join(explanations.get(title, ["No explanation"]))}</p>
        </div>
        """, unsafe_allow_html=True)

    cols = st.columns(2)
    for idx, (title, score) in enumerate(recommendations[3:10]):
        rank = idx + 4

        row = df.iloc[title_to_idx[title]]
        img = row.get("image", "")

        with cols[idx % 2]:
            st.markdown(f"""
            <div style="padding:15px; border-radius:10px; background:#111827; margin-bottom:15px;">
                <h4>#{rank} - {title}</h4>
                <h3>{score}</h3>
                {"<img src='" + img + "' width='150'>" if img else ""}
                <p>{row.get('authors','')}</p>
                <p>{' | '.join(explanations.get(title, ["No explanation"]))}</p>
            </div>
            """, unsafe_allow_html=True)

    # -----------------------------
    # Table (11–50)
    # -----------------------------
    table_data = []
    for i, (title, score) in enumerate(recommendations[10:50], start=11):
        row = df.iloc[title_to_idx[title]]
        table_data.append({
            "Rank": i,
            "Title": title,
            "English Title": row.get("title_eng", ""),
            "Author": row.get("authors", ""),
            "Relevance Score": score
        })

    st.write("## 📊 More Recommendations")
    st.dataframe(pd.DataFrame(table_data), hide_index=True)

    # -----------------------------
    # Visualization and Result Interpretation
    # -----------------------------
    st.write("## 📊 Feature Importance")
    # feature_importances = model.feature_importances_

    # feature_names = feature_objs["feature_names"]

    # fi_df = pd.DataFrame({
    #     "feature": feature_names,
    #     "importance": feature_importances
    # }).sort_values(by="importance", ascending=False).head(20)

    # st.bar_chart(fi_df.set_index("feature"))

    # st.write("## 🌳 Decision Path (Sample Tree)")

    # tree_text = get_decision_path_text(
    #     model,
    #     feature_objs["feature_names"]
    # )
    # st.text(tree_text)

    top_features_df, grouped_importance_df = (
        get_explainable_feature_importance(
            model,
            feature_objs["feature_names"]
        )
    )

    for _, row in top_features_df.iterrows():
        importance_pct = row["importance"] * 100

        st.markdown(
            f"""
            ✅ **{row['display_name']}**
            - Influence Score: {importance_pct:.2f}%
            """
        )
    
    st.bar_chart(
        grouped_importance_df.set_index("Factor")[
            "Percentage"
        ]
    )

    dominant_factor = grouped_importance_df.loc[
        grouped_importance_df["Percentage"].idxmax()
    ]

    st.info(
        f"""
        Most recommendations were influenced by
        **{dominant_factor['Factor']}**
        ({dominant_factor['Percentage']:.1f}%).
        """
    )

    # -----------------------------
    # User selects liked recommendations
    # -----------------------------
    st.write("## ✅ Select the recommendations you actually like")

    recommended_titles = [title for title, _ in recommendations[:50]]

    liked_from_recs = st.multiselect(
        "Pick any recommendations you like:",
        recommended_titles,
        key="liked_from_recs"
    )

    # -----------------------------
    # Metrics display
    # -----------------------------
    if liked_from_recs:
        metrics = compute_metrics(recommendations, liked_from_recs)
        st.write("## 📊 Evaluation Results (Sementara saja)")
        st.write(metrics)
    else:
        metrics = {}

    # -----------------------------
    # Feedback form
    # -----------------------------
    st.write("## 📝 Feedback")

    accuracy = st.slider("Accuracy", 1, 5, 3)
    diversity = st.slider("Diversity", 1, 5, 3)
    serendipity = st.slider("Serendipity", 1, 5, 3)

    # -----------------------------
    # Submit feedback
    # -----------------------------
    if st.button("Submit Feedback", key="submit_feedback"):

        feedback = {
            "accuracy": accuracy,
            "diversity": diversity,
            "serendipity": serendipity
        }

        send_to_google_sheets(
            feedback,
            liked_titles,
            metrics,
            liked_from_recs
        )

        st.success("✅ Feedback and metrics sent successfully!")
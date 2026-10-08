import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

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
    # df.columns = df.columns.str.lower().str.strip()
    X, titles, title_to_idx, feature_objs = create_features(df)
    return df, X, titles, title_to_idx, feature_objs

df, X, titles, title_to_idx, feature_objs = load_data()

# -----------------------------
# User Input
# -----------------------------

all_titles = titles

if "liked_titles" not in st.session_state:
    st.session_state["liked_titles"] = []

if "disliked_titles" not in st.session_state:
    st.session_state["disliked_titles"] = []

def update_liked():
    liked = st.session_state["liked_titles"]

    # Remove newly liked titles from disliked
    st.session_state["disliked_titles"] = [
        title
        for title in st.session_state["disliked_titles"]
        if title not in liked
    ]


def update_disliked():
    disliked = st.session_state["disliked_titles"]

    # Remove newly disliked titles from liked
    st.session_state["liked_titles"] = [
        title
        for title in st.session_state["liked_titles"]
        if title not in disliked
    ]

liked_options = [
    title
    for title in all_titles
    if title not in st.session_state["disliked_titles"]
]

disliked_options = [
    title
    for title in all_titles
    if title not in st.session_state["liked_titles"]
]

st.write("### Select novels you like:")

st.multiselect(
    "Choose novels you like",
    options=liked_options,
    key="liked_titles",
    on_change=update_liked
)

st.write("### Select novels you dislike (optional):")

st.multiselect(
    "Choose novels you dislike",
    options=disliked_options,
    key="disliked_titles",
    on_change=update_disliked
)

liked_titles = st.session_state["liked_titles"]
disliked_titles = st.session_state["disliked_titles"]

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

        # -----------------------------
    # Decision Path Visualization
    # -----------------------------
    st.write("## 🌳 Decision Path (Random Forest)")

    def pretty_feature(name):
        if name.startswith("genre_"):
            return f"Genre: {name[6:].title()}"
        if name.startswith("synopsis_"):
            return f"Synopsis term: {name[9:]}"
        if name.startswith("status_"):
            cleaned = name[7:].replace("status_", "").replace("_", " ")
            return f"Status: {cleaned.title()}"
        labels = {
            "score": "Novel Rating",
            "favorites": "Reader Favorites",
            "scored_by": "Number of Ratings",
            "popularty": "Popularity",
            "year_start": "Publication Year",
            "n_chapters": "Number of Chapters",
            "n_volumes": "Number of Volumes",
        }
        return labels.get(name, name)

    pretty_names = [pretty_feature(f) for f in feature_objs["feature_names"]]
    n_trees = len(model.estimators_)

    tree_id = st.slider("Which tree to inspect", 0, n_trees - 1, 0, key="tree_id")
    tree = model.estimators_[tree_id]

    tab_tree, tab_path = st.tabs(
        ["Tree structure", "Path for a recommended novel"]
    )

    with tab_tree:
        depth_shown = st.slider("Depth shown", 1, 5, 3, key="tree_depth_shown")

        fig, ax = plt.subplots(figsize=(14, 7))
        plot_tree(
            tree,
            feature_names=pretty_names,
            class_names=["Not liked", "Liked"],
            max_depth=depth_shown,
            filled=True,
            rounded=True,
            impurity=False,
            fontsize=8,
            ax=ax,
        )
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "Each box is a yes/no question about a novel's features. "
            "Left = condition true, right = condition false. "
            "Colour shows the majority class in that node."
        )

    with tab_path:
        rec_titles = [t for t, _ in recommendations[:50]]
            
        chosen = st.selectbox(
            "Pick a recommended novel",
            rec_titles,
            key="path_title"
        )
        
        x = X[title_to_idx[chosen]]
        x_dense = x.toarray().ravel()

        node_ids = tree.decision_path(x).indices
        t = tree.tree_

        steps = []

        for step_number, node in enumerate(node_ids, start=1):

            if t.children_left[node] == -1:
                break

            f = t.feature[node]
            thr = t.threshold[node]
            val = x_dense[f]

            went_left = val <= thr
            feature_name = pretty_names[f]

            original_name = feature_objs["feature_names"][f]

            if original_name.startswith("genre_"):
                if val > 0:
                    condition = "HAS genre"
                    result = "TRUE"
                else:
                    condition = "HAS genre"
                    result = "FALSE"

            else:
                condition = f"{val:.3f} {'≤' if went_left else '>'} {thr:.3f}"
                result = "TRUE" if went_left else "FALSE"

            steps.append({
                "Step": step_number,
                "Feature": feature_name,
                "Condition": condition,
                "Result": result
            })

        if steps:

            path_df = pd.DataFrame(steps)

            st.dataframe(
                path_df,
                use_container_width=True,
                hide_index=True
            )

        else:
            st.info(
                "This tree is a single leaf, so there are no decision splits."
            )

            c1, c2 = st.columns(2)
            c1.metric("This tree's probability of 'Liked'",
                    f"{tree.predict_proba(x)[0, 1]:.2f}")
            c2.metric("Whole forest's probability of 'Liked'",
                    f"{model.predict_proba(x)[0, 1]:.2f}")

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
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import recall_score, precision_score, f1_score, average_precision_score, ndcg_score
from scipy.sparse import vstack, hstack, csr_matrix
from sklearn.tree import export_text
from difflib import SequenceMatcher
import warnings
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import streamlit as st
warnings.filterwarnings('ignore')

# -------------------------------
# Remove similar titles(sequel)
# -------------------------------
def is_similar_title(title1, title2, threshold=0.75):
    title1 = str(title1).lower().strip()
    title2 = str(title2).lower().strip()

    similarity = SequenceMatcher(None, title1, title2).ratio()

    return (
        similarity >= threshold
        or title1 in title2
        or title2 in title1
    )

def find_similar_titles(df):
    suspicious_pairs = []

    titles = df['title'].tolist()

    for i in range(len(titles)):
        title_i = titles[i]

        for j in range(i + 1, len(titles)):
            title_j = titles[j]

            if is_similar_title(title_i, title_j):
                suspicious_pairs.append((i, j))

    return suspicious_pairs

def remove_sequel_duplicates(df, suspicious_pairs):
    rows_to_remove = set()

    for i, j in suspicious_pairs:

        author_i = str(df.iloc[i].get('authors', '')).lower().strip()
        author_j = str(df.iloc[j].get('authors', '')).lower().strip()

        # Only remove if same author
        if author_i == author_j:

            fav_i = df.iloc[i].get('favorites', 0)
            fav_j = df.iloc[j].get('favorites', 0)

            # Keep more popular entry
            if fav_i >= fav_j:
                rows_to_remove.add(j)
            else:
                rows_to_remove.add(i)

    if rows_to_remove:
        df = df.drop(df.index[list(rows_to_remove)])

    return df.reset_index(drop=True)

# -------------------------------
# Load and clean data
# -------------------------------
def load_and_clean_data(filepath):
    df = pd.read_csv(filepath, encoding='latin1')
    
    # Fill text columns
    text_cols = [
        'title', 'title_eng', 'authors',
        'genres', 'synopsis', 'status', 'image'
    ]

    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # Fill numeric columns
    num_cols = [
        'score', 'scored_by', 'popularty',
        'favorites', 'year_start',
        'n_chapters', 'n_volumes'
    ]

    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    df = df.drop_duplicates(subset=['title'])

    # suspicious_pairs = find_similar_titles(df)
    # df = remove_sequel_duplicates(df, suspicious_pairs)

    df = df.reset_index(drop=True)

    assert df.isnull().sum().sum() == 0, "Dataset still contains missing values"

    print(f"After cleaning: {len(df)} rows")

    return df

# -------------------------------
# Feature Engineering
# -------------------------------
def preprocess_synopsis(text):
    text = str(text).lower()

    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

    text = re.sub(r'\s+', ' ', text).strip()

    words = text.split()
    words = [w for w in words if w not in ENGLISH_STOP_WORDS]

    return ' '.join(words)

def create_features(df):
    genre_series = df['genres'].str.split(',')
    mlb = MultiLabelBinarizer()
    genre_features = mlb.fit_transform(genre_series)
    genre_df = pd.DataFrame(genre_features, columns=mlb.classes_)
    
    df['clean_synopsis'] = df['synopsis'].apply(preprocess_synopsis)
    tfidf = TfidfVectorizer(max_features=1000, stop_words='english', min_df=2)
    syn_features = tfidf.fit_transform(df['clean_synopsis'])
    
    num_cols = ['score', 'scored_by', 'popularty', 'favorites', 'year_start', 'n_chapters', 'n_volumes']
    for col in num_cols:
        if col not in df.columns:
            df[col] = 0
    num_data = df[num_cols].fillna(0).values
    
    if 'status' in df.columns:
        status_ohe = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
        status_features = status_ohe.fit_transform(df[['status']])
    else:
        status_features = csr_matrix((len(df), 0))
    
    genre_feature_names = [f"genre_{g}" for g in mlb.classes_]

    synopsis_feature_names = [
        f"synopsis_{w}" for w in tfidf.get_feature_names_out()
    ]

    numeric_feature_names = num_cols

    if 'status' in df.columns:
        status_feature_names = [
            f"status_{c}"
            for c in status_ohe.get_feature_names_out(['status'])
        ]
    else:
        status_feature_names = []

    feature_names = (
        genre_feature_names +
        synopsis_feature_names +
        numeric_feature_names +
        status_feature_names
    )

    X = hstack([
        csr_matrix(genre_features),
        syn_features,
        csr_matrix(num_data),
        status_features
    ])
    
    title_to_idx = {title: idx for idx, title in enumerate(df['title'])}
    
    return X, df['title'].values, title_to_idx, {
        'genre_columns': mlb.classes_,
        'tfidf': tfidf,
        'status_ohe': status_ohe if 'status' in df.columns else None,
        'feature_names': feature_names,
    }

def get_hard_negatives(df, liked_idx, n_samples):

    liked_genres = set()

    for idx in liked_idx:

        genres = str(
            df.iloc[idx]["genres"]
        ).split(",")

        liked_genres.update(
            g.strip().lower()
            for g in genres
        )

    hard_candidates = []

    for idx in range(len(df)):

        if idx in liked_idx:
            continue

        novel_genres = set(
            g.strip().lower()
            for g in str(
                df.iloc[idx]["genres"]
            ).split(",")
        )

        overlap = len(
            liked_genres.intersection(
                novel_genres
            )
        )

        if overlap <= 1:
            hard_candidates.append(idx)

    if len(hard_candidates) >= n_samples:
        return np.random.choice(
            hard_candidates,
            size=n_samples,
            replace=False
        )

    return None

# -------------------------------
# Recommendation using Random Forest
# -------------------------------
def recommend(liked_titles, X, titles, title_to_idx, df, disliked_titles=None, top_k=50):
    liked_idx = [title_to_idx[t] for t in liked_titles if t in title_to_idx]
    if len(liked_idx) == 0:
        raise ValueError("None of the liked titles found in dataset.")
    
    if disliked_titles is None:
        disliked_titles = []

    disliked_idx = [
        title_to_idx[t]
        for t in disliked_titles
        if t in title_to_idx
    ]

    X_pos = X[liked_idx]
    y_pos = np.ones(len(liked_idx))
    
    all_idx = set(range(X.shape[0]))
    # neg_candidates = list(all_idx - set(liked_idx))
    # if len(neg_candidates) < len(liked_idx):
    #     neg_idx = np.random.choice(neg_candidates, size=len(liked_idx), replace=True)
    # else:
    #     neg_idx = np.random.choice(neg_candidates, size=len(liked_idx), replace=False)
    if len(disliked_idx) > 0:
        neg_idx = disliked_idx

        needed = len(liked_idx)/2 - len(neg_idx)

        if needed>0:
            hard_negatives = get_hard_negatives(
                df,
                liked_idx,
                needed
            )
            neg_idx.extend(hard_negatives)
    else:
        neg_idx = get_hard_negatives(
            df,
            liked_idx,
            len(liked_idx)/2
        )

        if neg_idx is None:
            neg_candidates = list(
                all_idx - set(liked_idx)
            )

            neg_idx = np.random.choice(
                neg_candidates,
                size=len(liked_idx)/2,
                replace=False
            )
    X_neg = X[neg_idx]
    y_neg = np.zeros(len(neg_idx))
    
    X_train = vstack([X_pos, X_neg])
    y_train = np.concatenate([y_pos, y_neg])
    
    best_params, _ = tune_random_forest(X_train, y_train)

    clf = RandomForestClassifier(
        **best_params,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    
    proba = clf.predict_proba(X)[:, 1]
    
    results = []
    for idx in range(X.shape[0]):
        if idx not in liked_idx:
            results.append((titles[idx], proba[idx]))
    results.sort(key=lambda x: x[1], reverse=True)
    top_results = results[:top_k]
    
    feature_importances = clf.feature_importances_
    
    avg_liked = X_pos.mean(axis=0).A1
    explanations = []
    for title, prob in top_results:
        idx = title_to_idx[title]
        feat_vector = X[idx].toarray().flatten()
        diff = feat_vector - avg_liked
        top_feat_indices = np.argsort(-np.abs(diff))[:5]
        top_features_desc = []
        for fi in top_feat_indices:
            if abs(diff[fi]) > 0.01:
                top_features_desc.append((f"Feature_{fi}", diff[fi]))
        explanations.append((title, prob, top_features_desc[:3]))
    
    return top_results, clf, explanations, feature_importances

# -------------------------------
# Evaluation metrics
# -------------------------------
def evaluate_recommendations(X, titles, title_to_idx, test_ratio=0.2, top_k=50):
    all_indices = list(range(X.shape[0]))
    np.random.seed(42)
    liked_idx = np.random.choice(all_indices, size=int(len(all_indices)*0.2), replace=False)
    not_liked_idx = list(set(all_indices) - set(liked_idx))
    
    train_liked, test_liked = train_test_split(liked_idx, test_size=test_ratio, random_state=42)
    
    X_train_pos = X[train_liked]
    y_train_pos = np.ones(len(train_liked))
    neg_sample = np.random.choice(not_liked_idx, size=len(train_liked), replace=False)
    X_train_neg = X[neg_sample]
    y_train_neg = np.zeros(len(neg_sample))
    X_train = vstack([X_train_pos, X_train_neg])
    y_train = np.concatenate([y_train_pos, y_train_neg])
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    scores = clf.predict_proba(X)[:, 1]
    
    candidate_indices = list(set(all_indices) - set(train_liked))
    candidate_scores = [(idx, scores[idx]) for idx in candidate_indices]
    candidate_scores.sort(key=lambda x: x[1], reverse=True)
    
    y_true = np.zeros(len(candidate_indices))
    for i, idx in enumerate(candidate_indices):
        if idx in test_liked:
            y_true[i] = 1
    
    pred_scores = np.array([score for _, score in candidate_scores])
    recall = recall_score(y_true, pred_scores >= 0.5)  # threshold 0.5 for binary
    precision = precision_score(y_true, pred_scores >= 0.5)
    f1 = f1_score(y_true, pred_scores >= 0.5)
    
    map_score = average_precision_score(y_true, pred_scores)
    
    ndcg = ndcg_score([y_true], [pred_scores], k=top_k)
    
    relevant_retrieved = sum(1 for i, idx in enumerate(candidate_indices[:top_k]) if idx in test_liked)
    recall_at_k = relevant_retrieved / len(test_liked) if len(test_liked) > 0 else 0
    
    return {
        'Recall@K': recall_at_k,
        'F1-Score': f1,
        'MAP': map_score,
        'NDCG': ndcg,
        'Precision': precision
    }

# -------------------------------
# Visualization (Explainability)
# -------------------------------
def plot_feature_importance(feature_importances, top_n=20, title="Feature Importance"):
    plt.figure(figsize=(10, 6))
    indices = np.argsort(feature_importances)[-top_n:]
    plt.barh(range(len(indices)), feature_importances[indices])
    plt.yticks(range(len(indices)), [f"F{i}" for i in indices])
    plt.xlabel("Importance")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def get_explainable_feature_importance(model, feature_names, top_n=10):
    """
    Returns:
        top_features_df
        grouped_importance_df
    """

    import pandas as pd

    importances = model.feature_importances_

    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })

    fi_df = fi_df.sort_values(
        by="importance",
        ascending=False
    )

    # -------------------------
    # Human-readable labels
    # -------------------------

    def prettify_feature(feature):

        if feature.startswith("genre_"):
            return f"Genre: {feature.replace('genre_', '').title()}"

        elif feature.startswith("synopsis_"):
            return f"Story Theme: {feature.replace('synopsis_', '').title()}"

        elif feature.startswith("status_"):
            return f"Status: {feature.replace('status_', '').replace('_', ' ').title()}"

        elif feature == "score":
            return "Novel Rating"

        elif feature == "favorites":
            return "Reader Favorites"

        elif feature == "scored_by":
            return "Number of Ratings"

        elif feature == "popularty":
            return "Popularity"

        elif feature == "year_start":
            return "Publication Year"

        elif feature == "n_chapters":
            return "Number of Chapters"

        elif feature == "n_volumes":
            return "Number of Volumes"

        return feature

    fi_df["display_name"] = fi_df["feature"].apply(
        prettify_feature
    )

    top_features_df = fi_df.head(top_n)

    # -------------------------
    # Group by feature type
    # -------------------------

    groups = {
        "Genres": 0,
        "Story Themes": 0,
        "Novel Statistics": 0,
        "Publication Status": 0
    }

    for _, row in fi_df.iterrows():

        feature = row["feature"]
        importance = row["importance"]

        if feature.startswith("genre_"):
            groups["Genres"] += importance

        elif feature.startswith("synopsis_"):
            groups["Story Themes"] += importance

        elif feature.startswith("status_"):
            groups["Publication Status"] += importance

        else:
            groups["Novel Statistics"] += importance

    grouped_importance_df = pd.DataFrame({
        "Factor": groups.keys(),
        "Importance": groups.values()
    })

    grouped_importance_df["Percentage"] = (
        grouped_importance_df["Importance"]
        / grouped_importance_df["Importance"].sum()
        * 100
    )

    return top_features_df, grouped_importance_df

def print_explanation(recommendations, explanations):
    print("\n=== Recommendations with Explanations ===\n")
    for (title, prob), (_, _, feat_list) in zip(recommendations, explanations):
        print(f"Title: {title} (Score: {prob:.4f})")
        print("  Top contributing features (difference from your liked novels):")
        for feat_name, diff in feat_list:
            print(f"    - {feat_name}: {diff:+.4f}")
        print()

# -------------------------------
# Main execution
# -------------------------------
def main(csv_path, liked_titles, top_k=50):
    df = load_and_clean_data(csv_path)
    X, titles, title_to_idx, feature_objs = create_features(df)
    
    recommendations, model, explanations, feature_importances = recommend(
        liked_titles, X, titles, title_to_idx, top_k=top_k
    )

    recommendations = rerank_results(recommendations, df, title_to_idx)
    
    print("\n=== Top Recommendations ===")
    for i, (title, prob) in enumerate(recommendations, 1):
        print(f"{i}. {title} (relevance score: {prob:.4f})")
    
    plot_feature_importance(feature_importances, top_n=20, title="Global Feature Importance")
    
    print_explanation(recommendations, explanations)
    
    print("\n=== Model Evaluation (Simulated User) ===")
    metrics = evaluate_recommendations(X, titles, title_to_idx, test_ratio=0.2, top_k=top_k)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    return recommendations

def rerank_results(results, df, title_to_idx, alpha=0.7, beta=0.2, gamma=0.1):
    reranked = []

    max_pop = df['popularty'].max() if 'popularty' in df.columns else 1

    selected_genres = []

    for title, score in results:
        idx = title_to_idx[title]
        row = df.iloc[idx]

        # Popularity
        popularity = row.get('popularty', 0) / max_pop if max_pop > 0 else 0

        # Genre diversity penalty
        genres = set(str(row['genres']).split(","))
        overlap_penalty = 0

        for g in selected_genres:
            overlap = len(genres.intersection(g))
            overlap_penalty += overlap * 0.05

        final_score = (alpha * score) + (beta * popularity) - (gamma * overlap_penalty)

        reranked.append((title, final_score))
        selected_genres.append(genres)

    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked

def get_decision_path_text(model, feature_names):
    tree = model.estimators_[0]
    return export_text(tree, feature_names=list(feature_names), max_depth=3)

def generate_explanations(df, recommendations, liked_titles, title_to_idx):
    explanations = {}

    liked_genres = set()
    liked_keywords = set()

    for t in liked_titles:
        if t in title_to_idx:
            row = df.iloc[title_to_idx[t]]
            liked_genres.update(str(row['genres']).split(","))
            liked_keywords.update(str(row['synopsis']).lower().split()[:30])

    for title, score in recommendations:
        idx = title_to_idx[title]
        row = df.iloc[idx]

        reasons = []

        # Genre similarity
        genres = set(str(row['genres']).split(","))
        common_genres = genres.intersection(liked_genres)
        if len(common_genres) > 0:
            reasons.append(f"Shares genres you like ({', '.join(list(common_genres)[:2])})")

        # Synopsis similarity
        synopsis_words = set(str(row['synopsis']).lower().split()[:30])
        overlap = synopsis_words.intersection(liked_keywords)
        if len(overlap) > 5:
            reasons.append("Has a similar story theme to your liked novels")

        # Popularity / score
        if row.get('score', 0) > 8:
            reasons.append("Highly rated by many readers")

        if row.get('popularty', 0) > df['popularty'].median():
            reasons.append("Popular among readers")

        if not reasons:
            reasons.append("Matches your overall reading preferences")

        explanations[title] = reasons[:2]

    return explanations

def get_gsheet_client():
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]

    creds_dict = st.secrets["gcp_service_account"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    # creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
    client = gspread.authorize(creds)

    return client

def send_to_google_sheets(feedback, liked_titles, metrics, liked_from_recs):
    client = get_gsheet_client()

    sheet = client.open("LN Recommendations Results").sheet1

    sheet.append_row([
        str(datetime.now()),
        ", ".join(liked_titles),
        ", ".join(liked_from_recs),
        feedback["accuracy"],
        feedback["diversity"],
        feedback["serendipity"],
        metrics.get("precision@k", ""),
        metrics.get("recall@k", ""),
        metrics.get("f1", ""),
        metrics.get("map", ""),
        metrics.get("ndcg", "")
    ])

def compute_metrics(recommendations, liked_from_recs, top_k=10):
    recommended_titles = [title for title, _ in recommendations[:top_k]]

    y_true = []
    y_pred = []

    for title in recommended_titles:
        if title in liked_from_recs:
            y_true.append(1)
        else:
            y_true.append(0)
        
        y_pred.append(1)

    # Precision@K
    precision = sum(y_true) / len(recommended_titles) if recommended_titles else 0

    # Recall@K
    recall = sum(y_true) / len(liked_from_recs) if liked_from_recs else 0

    # F1 Score
    try:
        f1 = f1_score(y_true, y_pred)
    except:
        f1 = 0

    # MAP
    ap = 0
    hit_count = 0
    for i, val in enumerate(y_true):
        if val == 1:
            hit_count += 1
            ap += hit_count / (i + 1)
    map_score = ap / max(1, sum(y_true))

    # NDCG
    dcg = sum([rel / np.log2(idx + 2) for idx, rel in enumerate(y_true)])
    idcg = sum([1 / np.log2(i + 2) for i in range(sum(y_true))]) if sum(y_true) > 0 else 1
    ndcg = dcg / idcg if idcg > 0 else 0

    return {
        "precision@k": round(precision, 4),
        "recall@k": round(recall, 4),
        "f1": round(f1, 4),
        "map": round(map_score, 4),
        "ndcg": round(ndcg, 4)
    }

def tune_random_forest(X, y):
    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5]
    }

    best_score = -1
    best_params = None

    for params in ParameterGrid(param_grid):
        clf = RandomForestClassifier(
            **params,
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X, y)
        preds = clf.predict(X)
        score = f1_score(y, preds)

        if score > best_score:
            best_score = score
            best_params = params

    return best_params, best_score

# -------------------------------
# Hard Testing
# -------------------------------
if __name__ == "__main__":
    csv_file = "novels.csv"
    liked = ["Omniscient Reader's Viewpoint", "Solo Leveling", "No.6", "Kumo desu ga, Nani ka?", "No Game No Life"]  # dummy
    recommendations = main(csv_file, liked, top_k=50)
import pandas as pd
import requests
import time
from tqdm import tqdm

# =========================
# CONFIGURATION
# =========================

CLIENT_ID = "adbce9a7c9ac8aa8960eb4767bc54cdb"

INPUT_CSV = "novels.csv"
OUTPUT_CSV = "novels_enriched.csv"

HEADERS = {
    "X-MAL-CLIENT-ID": CLIENT_ID
}

# =========================
# MAL API FUNCTIONS
# =========================

def search_manga(title):
    """
    Search MAL by title and return first matching MAL ID.
    """
    try:
        response = requests.get(
            "https://api.myanimelist.net/v2/manga",
            headers=HEADERS,
            params={
                "q": title,
                "limit": 1
            },
            timeout=10
        )

        if response.status_code != 200:
            return None

        data = response.json()

        if not data.get("data"):
            return None

        return data["data"][0]["node"]["id"]

    except Exception as e:
        print(f"Search error for '{title}': {e}")
        return None


def get_manga_details(mal_id):
    """
    Retrieve detailed manga/light novel information.
    """
    fields = ",".join([
        "alternative_titles",
        "synopsis",
        "num_chapters",
        "num_volumes",
        "mean",
        "num_scoring_users",
        "authors",
        "genres"
    ])

    try:
        response = requests.get(
            f"https://api.myanimelist.net/v2/manga/{mal_id}",
            headers=HEADERS,
            params={"fields": fields},
            timeout=10
        )

        if response.status_code != 200:
            return None

        return response.json()

    except Exception as e:
        print(f"Detail error for MAL ID {mal_id}: {e}")
        return None


# =========================
# LOAD DATASET
# =========================

df = pd.read_csv(INPUT_CSV, encoding="latin1")

# Add MAL ID column if missing
if "mal_id" not in df.columns:
    df["mal_id"] = pd.NA

# =========================
# IDENTIFY ROWS NEEDING WORK
# =========================

columns_to_fill = [
    "score",
    "scored_by",
    "title_eng",
    "n_chapters",
    "n_volumes",
    "genres",
    "authors",
    "synopsis"
]

mask = df[columns_to_fill].isna().any(axis=1)

rows_to_process = df[mask]

print(f"Found {len(rows_to_process)} rows with missing values.")

# =========================
# PROCESS DATA
# =========================

for idx in tqdm(rows_to_process.index):

    title = str(df.at[idx, "title"]).strip()

    if not title:
        continue

    # Use existing MAL ID if available
    mal_id = df.at[idx, "mal_id"]

    if pd.isna(mal_id):
        mal_id = search_manga(title)

        if mal_id is None:
            continue

        df.at[idx, "mal_id"] = mal_id

        # avoid rate limits
        time.sleep(0.5)

    details = get_manga_details(int(mal_id))

    if details is None:
        continue

    # =========================
    # Fill missing values only
    # =========================

    if pd.isna(df.at[idx, "score"]):
        df.at[idx, "score"] = details.get("mean")

    if pd.isna(df.at[idx, "scored_by"]):
        df.at[idx, "scored_by"] = details.get("num_scoring_users")

    if pd.isna(df.at[idx, "title_eng"]):
        alt_titles = details.get("alternative_titles", {})
        df.at[idx, "title_eng"] = alt_titles.get("en")

    if pd.isna(df.at[idx, "n_chapters"]):
        df.at[idx, "n_chapters"] = details.get("num_chapters")

    if pd.isna(df.at[idx, "n_volumes"]):
        df.at[idx, "n_volumes"] = details.get("num_volumes")

    if pd.isna(df.at[idx, "synopsis"]):
        df.at[idx, "synopsis"] = details.get("synopsis")

    if pd.isna(df.at[idx, "genres"]):
        genres = details.get("genres", [])
        genre_names = [g["name"] for g in genres]
        df.at[idx, "genres"] = ", ".join(genre_names)

    if pd.isna(df.at[idx, "authors"]):
        authors = details.get("authors", [])

        author_names = []

        for author in authors:
            node = author.get("node", {})

            first = node.get("first_name", "")
            last = node.get("last_name", "")
            name = node.get("name", "")

            if first or last:
                full_name = f"{first} {last}".strip()
            elif name:
                full_name = name
            else:
                full_name = "Unknown"

            author_names.append(full_name)

        df.at[idx, "authors"] = ", ".join(author_names)

    # avoid rate limits
    if idx % 50 == 0:
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"Checkpoint saved at row {idx}")
    time.sleep(0.5)

# =========================
# SAVE RESULTS
# =========================

df.to_csv(OUTPUT_CSV, index=False)

print("\nFinished!")
print(f"Saved enriched dataset to: {OUTPUT_CSV}")
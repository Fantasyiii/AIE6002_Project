"""
Data preprocessing pipeline for TMDB 5000 Movie Dataset.
Processes raw CSV into structured documents for vectorization.
"""
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict


def parse_json_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Parse JSON string columns into Python objects."""
    import ast
    df[column] = df[column].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
    return df


def extract_genres(genres_list: List[Dict]) -> str:
    """Extract genre names from JSON list."""
    if not genres_list:
        return ""
    return ", ".join([g["name"] for g in genres_list])


def extract_keywords(keywords_list: List[Dict]) -> str:
    """Extract keyword names from JSON list."""
    if not keywords_list:
        return ""
    return ", ".join([k["name"] for k in keywords_list[:10]])  # Top 10 keywords


def create_movie_document(row: pd.Series) -> Dict:
    """Create a structured document from a movie row."""
    genres = extract_genres(row.get("genres", []))
    keywords = extract_keywords(row.get("keywords", []))

    # Rich text combining all metadata
    content = f"""Title: {row['title']}
Year: {row.get('release_date', 'Unknown')[:4] if pd.notna(row.get('release_date')) else 'Unknown'}
Genres: {genres}
Keywords: {keywords}
Overview: {row.get('overview', '')}"""

    return {
        "id": str(row["id"]),
        "title": row["title"],
        "content": content,
        "metadata": {
            "title": row["title"],
            "year": row.get("release_date", "Unknown")[:4] if pd.notna(row.get("release_date")) else "Unknown",
            "genres": genres,
            "keywords": keywords,
            "overview": row.get("overview", ""),
            "vote_average": row.get("vote_average", 0),
            "vote_count": row.get("vote_count", 0),
        }
    }


def process_dataset(
    movies_path: str = "../Dataset/tmdb_5000_movies.csv",
    output_path: str = "./data/movies_processed.json"
) -> List[Dict]:
    """
    Process TMDB dataset into structured documents.

    Args:
        movies_path: Path to tmdb_5000_movies.csv
        output_path: Path to save processed JSON

    Returns:
        List of movie documents
    """
    print(f"Loading dataset from {movies_path}...")
    df = pd.read_csv(movies_path)
    print(f"Loaded {len(df)} movies")

    # Parse JSON columns
    print("Parsing JSON columns...")
    df = parse_json_column(df, "genres")
    df = parse_json_column(df, "keywords")

    # Remove movies with missing overview (critical for embedding)
    initial_count = len(df)
    df = df.dropna(subset=["overview"])
    df = df[df["overview"].str.strip() != ""]
    print(f"Removed {initial_count - len(df)} movies with missing overview")
    print(f"Final dataset: {len(df)} movies")

    # Create documents
    print("Creating movie documents...")
    documents = []
    for _, row in df.iterrows():
        doc = create_movie_document(row)
        documents.append(doc)

    # Save to JSON
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(documents)} documents to {output_path}")
    return documents


if __name__ == "__main__":
    documents = process_dataset()
    print(f"\nSample document:")
    print(json.dumps(documents[0], ensure_ascii=False, indent=2))

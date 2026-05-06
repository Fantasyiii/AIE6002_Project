"""
Fetch new movies (2017-2025) from TMDB API to supplement the existing dataset.

Usage:
    1. Add TMDB_API_KEY to backend/.env
    2. Run: python fetch_tmdb_new.py

TMDB API Key: Free at https://www.themoviedb.org/settings/api
"""
import json
import time
import os
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
BASE_URL = "https://api.themoviedb.org/3"
OUTPUT_PATH = "./data/tmdb_new_movies.json"

# Use local Clash proxy to reach TMDB (foreign site, blocked without proxy in CN)
PROXIES = {"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"}

# Fetch movies from 2017 to 2025, sorted by popularity
YEAR_START = 2017
YEAR_END = 2025
MAX_PAGES_PER_YEAR = 20  # 20 pages × 20 movies = 400 per year, ~3200 total


def fetch_movies_for_year(year: int, max_pages: int = MAX_PAGES_PER_YEAR) -> list:
    """Fetch popular movies for a given year using TMDB discover endpoint."""
    movies = []
    for page in range(1, max_pages + 1):
        url = f"{BASE_URL}/discover/movie"
        params = {
            "api_key": TMDB_API_KEY,
            "language": "en-US",
            "sort_by": "popularity.desc",
            "include_adult": "false",
            "primary_release_date.gte": f"{year}-01-01",
            "primary_release_date.lte": f"{year}-12-31",
            "vote_count.gte": 50,
            "page": page,
        }
        for attempt in range(3):
            try:
                resp = requests.get(url, params=params, timeout=20, proxies=PROXIES)
                resp.raise_for_status()
                data = resp.json()
                results = data.get("results", [])
                if not results:
                    return movies
                movies.extend(results)
                total_pages = data.get("total_pages", 1)
                if page >= total_pages:
                    return movies
                time.sleep(0.35)
                break
            except Exception as e:
                if attempt < 2:
                    print(f"  Retry {attempt+1} on page {page}: {str(e)[:80]}")
                    time.sleep(3 + attempt * 3)
                else:
                    print(f"  Failed page {page} after 3 attempts: {str(e)[:80]}")
                    return movies
    return movies


def fetch_genre_map() -> dict:
    """Fetch TMDB genre id -> name mapping, with retry."""
    url = f"{BASE_URL}/genre/movie/list"
    for attempt in range(5):
        try:
            resp = requests.get(url, params={"api_key": TMDB_API_KEY, "language": "en-US"},
                                timeout=15, proxies=PROXIES)
            resp.raise_for_status()
            genres = resp.json().get("genres", [])
            return {g["id"]: g["name"] for g in genres}
        except Exception as e:
            if attempt < 4:
                wait = 5 + attempt * 5
                print(f"  Genre map retry {attempt+1} in {wait}s: {str(e)[:80]}")
                time.sleep(wait)
            else:
                raise


def movie_to_doc(movie: dict, genre_map: dict) -> dict:
    """Convert a TMDB API movie object to the same format as the processed dataset."""
    genre_names = ", ".join(
        genre_map.get(gid, "") for gid in movie.get("genre_ids", [])
        if genre_map.get(gid)
    )
    year = movie.get("release_date", "Unknown")[:4] if movie.get("release_date") else "Unknown"
    title = movie.get("title", "Unknown")
    overview = movie.get("overview", "").strip()

    content = (
        f"Title: {title}\n"
        f"Year: {year}\n"
        f"Genres: {genre_names}\n"
        f"Overview: {overview}"
    )

    return {
        "id": f"tmdb_{movie['id']}",
        "title": title,
        "content": content,
        "metadata": {
            "title": title,
            "year": year,
            "genres": genre_names,
            "keywords": "",
            "overview": overview,
            "vote_average": movie.get("vote_average", 0),
            "vote_count": movie.get("vote_count", 0),
        }
    }


def fetch_all_new_movies(year_start: int = YEAR_START, year_end: int = YEAR_END):
    if not TMDB_API_KEY:
        print("ERROR: TMDB_API_KEY not set in backend/.env")
        return

    print(f"Fetching genre map...")
    genre_map = fetch_genre_map()
    print(f"  Got {len(genre_map)} genres")

    # Load existing data to resume/merge
    existing_docs = []
    if Path(OUTPUT_PATH).exists():
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            existing_docs = json.load(f)
        print(f"  Loaded {len(existing_docs)} existing movies from {OUTPUT_PATH}")

    seen_ids = {doc["id"] for doc in existing_docs}
    all_docs = list(existing_docs)

    for year in range(year_start, year_end + 1):
        print(f"Fetching {year}...", end=" ", flush=True)
        movies = fetch_movies_for_year(year)
        docs = []
        for m in movies:
            doc_id = f"tmdb_{m['id']}"
            if doc_id in seen_ids:
                continue
            if not m.get("overview", "").strip():
                continue
            seen_ids.add(doc_id)
            docs.append(movie_to_doc(m, genre_map))
        all_docs.extend(docs)
        print(f"{len(docs)} new movies (total: {len(all_docs)})")

    # Save
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_docs, f, ensure_ascii=False, indent=2)

    print(f"\nDone! Saved {len(all_docs)} total movies to {OUTPUT_PATH}")


if __name__ == "__main__":
    import sys
    # Allow: python fetch_tmdb_new.py [year_start] [year_end]
    args = sys.argv[1:]
    ys = int(args[0]) if args else YEAR_START
    ye = int(args[1]) if len(args) > 1 else YEAR_END
    fetch_all_new_movies(ys, ye)

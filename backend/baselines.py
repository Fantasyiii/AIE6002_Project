"""
Baseline systems for comparison experiments.
"""
import json
import re
from typing import List, Dict
from difflib import SequenceMatcher

from langchain_core.documents import Document

from rag_chain import create_pure_llm_chain
from vectorstore import load_vectorstore, similarity_search


def baseline_pure_llm(query: str) -> Dict:
    """
    Baseline 1: Pure LLM without retrieval.
    Directly asks LLM to recommend movies.
    """
    chain = create_pure_llm_chain()
    answer = chain.invoke(query)

    return {
        "query": query,
        "answer": answer,
        "sources": [],
        "baseline": "pure-llm"
    }


def baseline_tag_based(query: str, movies_data: List[Dict] = None) -> Dict:
    """
    Baseline 2: Tag-based filtering using genre keywords.
    Extracts genre keywords from query and matches against movie genres.
    """
    # Load movies data if not provided
    if movies_data is None:
        with open("./data/movies_processed.json", "r", encoding="utf-8") as f:
            movies_data = json.load(f)

    # Extract genre keywords from query
    genre_keywords = {
        "action": "Action",
        "adventure": "Adventure",
        "animation": "Animation",
        "comedy": "Comedy",
        "crime": "Crime",
        "documentary": "Documentary",
        "drama": "Drama",
        "family": "Family",
        "fantasy": "Fantasy",
        "history": "History",
        "horror": "Horror",
        "music": "Music",
        "mystery": "Mystery",
        "romance": "Romance",
        "sci-fi": "Science Fiction",
        "science fiction": "Science Fiction",
        "thriller": "Thriller",
        "war": "War",
        "western": "Western"
    }

    query_lower = query.lower()
    matched_genres = []

    for keyword, genre in genre_keywords.items():
        if keyword in query_lower:
            matched_genres.append(genre)

    # If no genre matched, return empty
    if not matched_genres:
        return {
            "query": query,
            "answer": "No specific genre detected in query. Please mention a genre like 'action', 'comedy', 'sci-fi', etc.",
            "sources": [],
            "baseline": "tag-based"
        }

    # Filter movies by genre
    matched_movies = []
    for movie in movies_data:
        movie_genres = movie["metadata"].get("genres", "").lower()
        if any(g.lower() in movie_genres for g in matched_genres):
            matched_movies.append(movie)

    # Sort by vote_average (popularity proxy)
    matched_movies.sort(
        key=lambda x: x["metadata"].get("vote_average", 0),
        reverse=True
    )

    # Take top 5
    top_movies = matched_movies[:5]

    # Format answer
    if top_movies:
        answer_lines = [f"Based on genre matching ({', '.join(matched_genres)}):"]
        for i, movie in enumerate(top_movies, 1):
            answer_lines.append(
                f"{i}. {movie['metadata']['title']} ({movie['metadata'].get('year', 'N/A')})\n"
                f"   Genres: {movie['metadata'].get('genres', 'N/A')}"
            )
        answer = "\n\n".join(answer_lines)
    else:
        answer = f"No movies found matching genres: {', '.join(matched_genres)}"

    return {
        "query": query,
        "answer": answer,
        "sources": [
            {
                "title": m["metadata"]["title"],
                "year": m["metadata"].get("year", "N/A"),
                "genres": m["metadata"].get("genres", "N/A")
            }
            for m in top_movies
        ],
        "baseline": "tag-based"
    }


def baseline_retrieval_only(query: str, top_k: int = 5) -> Dict:
    """
    Baseline 3: Retrieval-only without LLM generation.
    Returns raw retrieved documents.
    """
    vectorstore = load_vectorstore()
    if not vectorstore:
        return {
            "query": query,
            "answer": "Vector store not available.",
            "sources": [],
            "baseline": "retrieval-only"
        }

    docs = similarity_search(vectorstore, query, k=top_k)

    # Format as simple list
    answer_lines = ["Retrieved movies (no LLM generation):"]
    for i, doc in enumerate(docs, 1):
        answer_lines.append(
            f"{i}. {doc.metadata['title']} ({doc.metadata.get('year', 'N/A')})\n"
            f"   Genres: {doc.metadata.get('genres', 'N/A')}\n"
            f"   Overview: {doc.metadata.get('overview', 'N/A')[:150]}..."
        )

    answer = "\n\n".join(answer_lines)

    return {
        "query": query,
        "answer": answer,
        "sources": [
            {
                "title": doc.metadata["title"],
                "year": doc.metadata.get("year", "N/A"),
                "genres": doc.metadata.get("genres", "N/A")
            }
            for doc in docs
        ],
        "baseline": "retrieval-only"
    }


if __name__ == "__main__":
    test_query = "sci-fi movie about space travel"

    print("=" * 60)
    print("Baseline 1: Pure-LLM")
    print("=" * 60)
    result = baseline_pure_llm(test_query)
    print(result["answer"][:300])

    print("\n" + "=" * 60)
    print("Baseline 2: Tag-Based")
    print("=" * 60)
    result = baseline_tag_based(test_query)
    print(result["answer"])

    print("\n" + "=" * 60)
    print("Baseline 3: Retrieval-Only")
    print("=" * 60)
    result = baseline_retrieval_only(test_query, top_k=3)
    print(result["answer"])

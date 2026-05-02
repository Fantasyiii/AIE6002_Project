"""
Evaluation metrics for movie recommendation system.
"""
import re
import time
from typing import List, Dict, Set
from difflib import SequenceMatcher

from langchain_core.documents import Document


def extract_movie_titles(text: str) -> List[str]:
    """Extract movie titles from text."""
    if not text:
        return []

    titles = []

    # Pattern 1: "Title" (in quotes)
    quoted = re.findall(r'"([^"]+)"', text)
    titles.extend(quoted)

    # Pattern 2: **Title** (bold markdown)
    bold = re.findall(r'\*\*([^*]+)\*\*', text)
    titles.extend(bold)

    # Pattern 3: Number. Title (Year) - capture title before year
    numbered = re.findall(r'\d+\.\s*([^\n(]+?)(?:\s*\(\d{4}\))?', text)
    for t in numbered:
        t = t.strip()
        # Remove trailing dash or colon
        t = re.sub(r'[\s:-]+$', '', t)
        if t:
            titles.append(t)

    # Pattern 4: "Title (Year)" without numbering
    year_pattern = re.findall(r'\b([A-Z][A-Za-z0-9\s:&\'-]+?)\s*\(\d{4}\)', text)
    titles.extend(year_pattern)

    # Clean and filter
    cleaned = []
    for t in titles:
        t = t.strip()
        # Remove common prefixes
        t = re.sub(r'^(?:Title|Movie|Film)\s*[:\-]?\s*', '', t, flags=re.I)
        # Filter out noise
        if len(t) > 1 and len(t) < 100 and not t.isdigit():
            cleaned.append(t)

    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for t in cleaned:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            unique.append(t)

    return unique


def detect_hallucination(response: str, sources: List[Dict]) -> Dict:
    """
    Detect hallucinated movies in response.
    Returns: {"hallucinated": [...], "hallucination_rate": float}
    """
    if not sources:
        # Pure-LLM baseline: can't verify, assume 100% hallucinated
        return {"hallucinated": [], "hallucination_rate": 1.0}

    source_titles = [s["title"].lower() for s in sources]
    response_titles = extract_movie_titles(response)

    if not response_titles:
        # No titles found in response, can't evaluate
        return {"hallucinated": [], "hallucination_rate": 0.0}

    hallucinated = []
    for title in response_titles:
        title_lower = title.lower()
        # Check if title is in sources (fuzzy match)
        matched = any(
            SequenceMatcher(None, title_lower, st).ratio() > 0.75
            for st in source_titles
        )
        if not matched:
            hallucinated.append(title)

    total_titles = len(response_titles) if response_titles else 1
    hallucination_rate = len(hallucinated) / total_titles

    return {
        "hallucinated": hallucinated,
        "hallucination_rate": round(hallucination_rate, 2)
    }


def calculate_diversity(recommendations: List[Document]) -> float:
    """
    Calculate intra-list diversity using genre overlap.
    Higher score = more diverse.
    """
    if len(recommendations) < 2:
        return 0.0

    # Extract genre sets
    genre_sets = []
    for doc in recommendations:
        genres = doc.metadata.get("genres", "")
        if genres:
            genre_set = set(g.strip().lower() for g in genres.split(","))
            genre_sets.append(genre_set)

    if not genre_sets:
        return 0.0

    # Calculate pairwise Jaccard distance (1 - Jaccard similarity)
    total_distance = 0.0
    count = 0

    for i in range(len(genre_sets)):
        for j in range(i + 1, len(genre_sets)):
            intersection = len(genre_sets[i] & genre_sets[j])
            union = len(genre_sets[i] | genre_sets[j])
            if union > 0:
                jaccard_sim = intersection / union
                jaccard_dist = 1 - jaccard_sim
                total_distance += jaccard_dist
                count += 1

    return round(total_distance / count, 2) if count > 0 else 0.0


def calculate_relevance(query: str, recommendations: List[Document]) -> float:
    """
    Calculate relevance score based on genre matching.
    Simple heuristic: check if query keywords match movie genres/overview.
    """
    if not recommendations:
        return 0.0

    query_lower = query.lower()
    query_keywords = set(query_lower.split())

    relevance_scores = []
    for doc in recommendations:
        text = (doc.metadata.get("genres", "") + " " + doc.metadata.get("overview", "")).lower()
        text_words = set(text.split())

        # Calculate overlap
        overlap = len(query_keywords & text_words)
        relevance = overlap / len(query_keywords) if query_keywords else 0
        relevance_scores.append(relevance)

    return round(sum(relevance_scores) / len(relevance_scores), 2)


def measure_latency(func, *args, **kwargs) -> tuple:
    """Measure execution time of a function."""
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    latency_ms = (end - start) * 1000
    return result, latency_ms


def evaluate_system(system_name: str, results: List[Dict]) -> Dict:
    """
    Aggregate evaluation results for a system.

    Args:
        system_name: Name of the system
        results: List of result dicts with keys: query, answer, sources, latency_ms

    Returns:
        Dict with aggregated metrics
    """
    total_queries = len(results)

    # Hallucination rate
    hallucination_rates = []
    for r in results:
        if r.get("sources"):
            h = detect_hallucination(r["answer"], r["sources"])
            hallucination_rates.append(h["hallucination_rate"])
        else:
            # No sources = pure LLM, count as 100% hallucinated
            hallucination_rates.append(1.0)

    avg_hallucination = round(sum(hallucination_rates) / len(hallucination_rates), 2) if hallucination_rates else 0.0

    # Latency
    latencies = [r.get("latency_ms", 0) for r in results]
    avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0.0

    # Source count (how many movies recommended)
    avg_sources = round(sum(len(r.get("sources", [])) for r in results) / total_queries, 2)

    return {
        "system": system_name,
        "total_queries": total_queries,
        "avg_hallucination_rate": avg_hallucination,
        "avg_latency_ms": avg_latency,
        "avg_recommendations": avg_sources
    }


if __name__ == "__main__":
    # Test metrics
    test_response = """
    1. Interstellar (2014) - A great sci-fi movie
    2. "The Matrix" - Another good one
    3. Gravity (2013)
    """
    test_sources = [
        {"title": "Interstellar"},
        {"title": "Gravity"}
    ]

    h = detect_hallucination(test_response, test_sources)
    print(f"Hallucination test: {h}")

    # Test with real RAG-like output
    test_response2 = """
    Based on your query, I recommend:

    1. Inception (2010) - A mind-bending thriller about dreams
    2. The Dark Knight (2008) - A superhero masterpiece
    """
    test_sources2 = [
        {"title": "Inception"},
        {"title": "The Dark Knight"}
    ]
    h2 = detect_hallucination(test_response2, test_sources2)
    print(f"Hallucination test 2: {h2}")

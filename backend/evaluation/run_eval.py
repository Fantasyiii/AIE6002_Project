"""
Automated evaluation script for all systems.
"""
import json
import sys
from typing import List, Dict
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from rag_chain import RAGPipeline
from baselines import baseline_pure_llm, baseline_tag_based, baseline_retrieval_only
from evaluation.metrics import evaluate_system, measure_latency


def load_test_queries():
    """Load test queries from JSON."""
    with open("./evaluation/test_queries.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["queries"]


def run_evaluation():
    """Run evaluation for all systems."""
    queries = load_test_queries()
    print(f"Loaded {len(queries)} test queries")

    # Initialize systems
    systems = {
        "VibeMatch (RAG)": RAGPipeline(retrieval_mode="similarity", top_k=5),
        "VibeMatch (MMR)": RAGPipeline(retrieval_mode="mmr", top_k=5),
    }

    results = {
        "VibeMatch (RAG)": [],
        "VibeMatch (MMR)": [],
        "Pure-LLM": [],
        "Tag-Based": [],
        "Retrieval-Only": []
    }

    # Run each query through each system
    for q in queries:
        query_text = q["query"]
        print(f"\nQuery: {query_text}")

        # VibeMatch RAG
        try:
            result, latency = measure_latency(
                systems["VibeMatch (RAG)"].recommend, query_text
            )
            result["latency_ms"] = latency
            results["VibeMatch (RAG)"].append(result)
            print(f"  RAG: {len(result['sources'])} sources, {latency:.0f}ms")
        except Exception as e:
            print(f"  RAG failed: {e}")

        # VibeMatch MMR
        try:
            result, latency = measure_latency(
                systems["VibeMatch (MMR)"].recommend, query_text
            )
            result["latency_ms"] = latency
            results["VibeMatch (MMR)"].append(result)
            print(f"  MMR: {len(result['sources'])} sources, {latency:.0f}ms")
        except Exception as e:
            print(f"  MMR failed: {e}")

        # Pure-LLM
        try:
            result, latency = measure_latency(baseline_pure_llm, query_text)
            result["latency_ms"] = latency
            results["Pure-LLM"].append(result)
            print(f"  Pure-LLM: {latency:.0f}ms")
        except Exception as e:
            print(f"  Pure-LLM failed: {e}")

        # Tag-Based
        try:
            result, latency = measure_latency(baseline_tag_based, query_text)
            result["latency_ms"] = latency
            results["Tag-Based"].append(result)
            print(f"  Tag-Based: {len(result['sources'])} sources, {latency:.0f}ms")
        except Exception as e:
            print(f"  Tag-Based failed: {e}")

        # Retrieval-Only
        try:
            result, latency = measure_latency(baseline_retrieval_only, query_text)
            result["latency_ms"] = latency
            results["Retrieval-Only"].append(result)
            print(f"  Retrieval-Only: {len(result['sources'])} sources, {latency:.0f}ms")
        except Exception as e:
            print(f"  Retrieval-Only failed: {e}")

    # Aggregate results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    summary = []
    for system_name, system_results in results.items():
        if system_results:
            metrics = evaluate_system(system_name, system_results)
            summary.append(metrics)

            print(f"\n{system_name}:")
            print(f"  Queries: {metrics['total_queries']}")
            print(f"  Avg Hallucination Rate: {metrics['avg_hallucination_rate']:.2%}")
            print(f"  Avg Latency: {metrics['avg_latency_ms']:.0f}ms")
            print(f"  Avg Recommendations: {metrics['avg_recommendations']}")

    # Save results
    output_dir = Path("./evaluation/results")
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Generate markdown report
    generate_report(summary, output_dir / "evaluation_report.md")

    print(f"\nResults saved to {output_dir}/")


def generate_report(summary: List[Dict], output_path: Path):
    """Generate markdown evaluation report."""
    lines = [
        "# VibeMatch Evaluation Report",
        "",
        "## Systems Compared",
        "",
        "| System | Queries | Hallucination Rate | Avg Latency | Avg Recommendations |",
        "|--------|---------|-------------------|-------------|-------------------|"
    ]

    for s in summary:
        lines.append(
            f"| {s['system']} | {s['total_queries']} | "
            f"{s['avg_hallucination_rate']:.2%} | "
            f"{s['avg_latency_ms']:.0f}ms | "
            f"{s['avg_recommendations']} |"
        )

    lines.extend([
        "",
        "## Key Findings",
        "",
        "- **Hallucination Rate**: Lower is better. VibeMatch (RAG) should have near 0% hallucination.",
        "",
        "- **Latency**: Measured end-to-end response time.",
        "",
        "- **Recommendations**: Average number of movies recommended per query.",
        ""
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    run_evaluation()

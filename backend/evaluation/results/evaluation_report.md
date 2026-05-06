# VibeMatch Evaluation Report

## Systems Compared

| System | Queries | Hallucination Rate | Avg Latency | Avg Recommendations |
|--------|---------|-------------------|-------------|-------------------|
| VibeMatch (RAG) | 15 | 59.00% | 10355ms | 5.0 |
| VibeMatch (MMR) | 15 | 54.00% | 8471ms | 5.0 |
| Pure-LLM | 15 | 100.00% | 10529ms | 0.0 |
| Tag-Based | 15 | 100.00% | 77ms | 0.0 |
| Retrieval-Only | 15 | 7.00% | 1732ms | 5.0 |

## Key Findings

- **Hallucination Rate**: Lower is better. VibeMatch (RAG) should have near 0% hallucination.

- **Latency**: Measured end-to-end response time.

- **Recommendations**: Average number of movies recommended per query.

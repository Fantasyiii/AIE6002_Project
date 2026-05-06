# VibeMatch: A Retrieval-Augmented Movie Recommendation System

## Abstract
This report presents VibeMatch, a semantic movie recommendation system that combines retrieval-augmented generation (RAG) with a local movie vector database. The system is designed to support natural language, intent-rich queries such as mood, theme, setting, and narrative constraints, while reducing hallucination risk by grounding responses in retrieved movie records. The implementation uses a Next.js frontend, a FastAPI backend, LangChain orchestration, Chroma as the persistent vector store, and the local embedding model all-MiniLM-L6-v2. In our reproducible local run (without external LLM API keys), we evaluated Tag-Based and Retrieval-Only baselines on 15 test queries. Retrieval-Only achieved substantially lower hallucination rate (13%) than Tag-Based (100% under this metric implementation) and produced stable top-k outputs, at the cost of higher latency (445 ms vs. 40 ms). These results support the core motivation of the project: retrieval grounding materially improves recommendation faithfulness compared with naive keyword filtering. We also discuss limitations due to unavailable LLM credentials and provide a clear path to complete full RAG and MMR experiments.

## 1. Introduction
Traditional recommender systems often rely on rigid filters (e.g., genre tags) or collaborative signals, which can underperform for nuanced user intent. In movie search, users frequently express needs in natural language with rich constraints (e.g., "warm ending", "set in Europe", "not too violent"). Large language models can interpret such intent but are prone to factual hallucination when unconstrained.

This project addresses that tension by using a retrieval-first architecture: candidate movies are retrieved from a curated corpus, and generation (when enabled) is expected to remain grounded in retrieved evidence. The system was built for the AIE6002 Large Language Models course, with emphasis on reproducibility, local control, and measurable evaluation.

## 2. Proposed Idea / Exploration
The key idea is to decompose recommendation into two stages:

1. Semantic retrieval over a local vector store to ensure factual grounding.
2. LLM-based response synthesis (optional/fallback-aware) for natural explanations.

We explore the following research-oriented direction:

- Whether grounding by retrieval reduces hallucination compared with non-grounded approaches.
- Whether MMR retrieval can improve relevance-diversity tradeoff (planned full run when API key is available).
- How simple baselines (Tag-Based, Retrieval-Only) compare under consistent metrics.

In this local environment, because no valid OpenAI/NVIDIA API key was present, we implemented and tested a robust retrieval fallback path for the main chat endpoint so the system remains usable and evaluable.

## 3. Technical Details
### 3.1 System Architecture
- Frontend: Next.js 15 + React 19 + TypeScript.
- Backend API: FastAPI.
- RAG orchestration: LangChain.
- Vector DB: Chroma (persistent local directory).
- Embedding model: sentence-transformers/all-MiniLM-L6-v2 (downloaded locally).
- Dataset: TMDB 5000 Movies (processed into 4,799 valid movie documents after dropping empty-overview records).

### 3.2 Data Processing and Indexing
- Parse TMDB CSV fields (genres/keywords JSON-like columns).
- Build normalized document text with title, year, genres, keywords, and overview.
- Persist processed records to JSON.
- Embed records with all-MiniLM-L6-v2 and build Chroma collection.

### 3.3 Retrieval and Generation Pipeline
- Similarity retrieval returns top-k nearest movie documents.
- MMR retrieval (available in code) is designed to improve diversity while preserving relevance.
- RAG mode: retrieved context + prompt template + chat LLM.
- Fallback mode (added in this run): if no LLM key, /chat returns retrieval-grounded recommendation list directly instead of failing.

### 3.4 Evaluation Setup
- Test set: 15 Chinese natural-language recommendation queries from backend/evaluation/test_queries.json.
- Metrics from evaluation code:
  - Hallucination Rate
  - Avg Latency (ms)
  - Avg Recommendations
- Systems successfully evaluated in this environment:
  - Tag-Based
  - Retrieval-Only
- LLM-dependent systems (RAG/MMR/Pure-LLM) were skipped due to missing API credentials.

## 4. Results
From backend/evaluation/results/evaluation_summary.json:

| System | Queries | Avg Hallucination Rate | Avg Latency | Avg Recommendations |
|---|---:|---:|---:|---:|
| Tag-Based | 15 | 100.00% | 40.22 ms | 0.0 |
| Retrieval-Only | 15 | 13.00% | 444.58 ms | 5.0 |

Qualitative runtime checks:
- Backend health endpoint responded normally.
- /chat endpoint returned grounded movie lists in fallback mode.
- Frontend was reachable at localhost:3000 and connected to backend API.

## 5. Analysis and Discussion
### 5.1 Why Retrieval-Only Outperformed Tag-Based on Faithfulness
Tag-Based baseline often produced zero sources for Chinese free-form queries because keyword matching logic is English-genre oriented and brittle. Under the current hallucination metric implementation, no-source outputs are penalized heavily, leading to 100% average hallucination rate. Retrieval-Only, in contrast, always returns source-backed candidates from Chroma and therefore maintains much lower hallucination.

### 5.2 Latency Tradeoff
Retrieval-Only incurs higher average latency than Tag-Based because vector search + embedding stack initialization is computationally heavier than simple keyword filters. The first query cold-start cost is notably large and inflates average latency; warm requests are much faster.

### 5.3 Validity and Limitations
- Missing LLM API key prevented full end-to-end RAG/MMR/Pure-LLM comparison in this run.
- The hallucination metric is heuristic and title-extraction based; it may under/over-estimate in edge cases.
- Tag-based baseline is not multilingual-robust; Chinese query tokenization/translation was not included.

### 5.4 Practical Implications
Even without generation, retrieval grounding already provides a strong reliability baseline for semantic recommendation. For production use, adding LLM synthesis with strict source constraints is likely to improve readability while preserving factual trustworthiness.

## 6. Reproducibility Notes (What Was Run)
The following steps were executed and verified locally on Windows:

1. Installed backend dependencies with pinned compatible versions.
2. Downloaded local embedding model all-MiniLM-L6-v2.
3. Built Chroma vector store from 4,799 processed movie documents.
4. Started FastAPI backend on port 8000.
5. Added retrieval fallback to /chat when no LLM key exists.
6. Installed frontend dependencies and started Next.js dev server on port 3000.
7. Ran evaluation script and generated results under backend/evaluation/results.

## 7. Sources
1. TMDB 5000 Movie Dataset (Kaggle): https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata
2. LangChain documentation: https://python.langchain.com/
3. Chroma documentation: https://docs.trychroma.com/
4. Sentence-Transformers model (all-MiniLM-L6-v2): https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
5. FastAPI documentation: https://fastapi.tiangolo.com/
6. Next.js documentation: https://nextjs.org/docs
7. Movies++ reference project: https://github.com/datastax/movies_plus_plus

## Appendix: Planned Full Experiment After API Key Configuration
Once OPENAI_API_KEY or NVIDIA_API_KEY is provided, rerun backend/evaluation/run_eval.py to include:
- VibeMatch (RAG)
- VibeMatch (MMR)
- Pure-LLM
and complete the originally intended ablation and comparison study.

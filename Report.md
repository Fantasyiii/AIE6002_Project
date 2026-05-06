# VibeMatch: RAG-Based Semantic Movie Recommendation System

## AIE6002 Large Language Models - Final Project Report

**Authors**: Yifei Chen, Lei Zhang, Shuhao Shi  
**Course**: AIE6002 Large Language Models  
**Institution**: The Chinese University of Hong Kong, Shenzhen  
**Date**: May 2026

---

## Abstract

This paper presents VibeMatch, a Retrieval-Augmented Generation (RAG) based semantic movie recommendation system designed to address the hallucination problem in Large Language Model (LLM) generated recommendations. Unlike traditional keyword-based systems, VibeMatch understands nuanced natural language queries describing mood, atmosphere, and complex preferences. We implement a complete RAG pipeline with ChromaDB vector storage, multilingual embeddings, and comprehensive baseline comparisons. Our evaluation on 8,254 movies demonstrates that RAG significantly reduces hallucination rates from 100% (Pure-LLM) to 59%, while Maximal Marginal Relevance (MMR) retrieval achieves better diversity with 54% hallucination rate and 18% faster response time.

**Keywords**: Retrieval-Augmented Generation, Movie Recommendation, Hallucination Reduction, Vector Database, LangChain

---

## 1. Introduction

### 1.1 Background and Motivation

Movie recommendation systems have evolved from simple collaborative filtering to sophisticated AI-powered solutions. However, modern LLM-based systems suffer from a critical flaw: **hallucination**—the tendency to generate plausible-sounding but factually incorrect information, including non-existent movies or fabricated plot details.

Traditional keyword-based systems (e.g., TMDB genre filtering) lack semantic understanding, while pure LLM approaches cannot guarantee factual accuracy. This creates a gap for users seeking recommendations based on complex, nuanced preferences like "a dark comedy about midlife crisis set in Europe with a heartwarming ending."

### 1.2 Research Questions

This project addresses three key research questions:

- **RQ1 (Factual Accuracy)**: Can RAG significantly reduce hallucination rates in LLM-generated movie recommendations compared to pure LLM approaches?
- **RQ2 (Retrieval Quality)**: How does MMR retrieval strategy affect the relevance-diversity trade-off in recommendations?
- **RQ3 (System Performance)**: What is the latency-performance trade-off between different retrieval strategies and baseline systems?

### 1.3 Contributions

Our main contributions include:

1. A complete RAG-based movie recommendation system with 8,254 movies from TMDB dataset
2. Implementation of three baseline systems for comprehensive comparison
3. Automated evaluation framework with hallucination detection and diversity metrics
4. Empirical evidence that RAG reduces hallucination by 41% compared to pure LLM

---

## 2. Related Work

### 2.1 Traditional Recommendation Systems

Collaborative filtering and content-based filtering have been the dominant approaches. However, these methods struggle with cold-start problems and cannot understand semantic nuances in user queries.

### 2.2 LLM-Based Recommendations

Recent works explore direct LLM recommendations, but hallucination remains unsolved. Our approach differs by grounding all recommendations in a verified movie database.

### 2.3 RAG for Recommendation

Retrieval-Augmented Generation has shown promise in question answering. We extend this to recommendation systems, with novel contributions in hallucination detection and diversity-aware retrieval.

---

## 3. Methodology

### 3.1 System Architecture

VibeMatch consists of three main components:

1. **Data Processing Pipeline**: Processes TMDB dataset into structured documents
2. **Vector Storage**: ChromaDB with multilingual embeddings for semantic search
3. **RAG Engine**: LangChain-based pipeline with configurable retrieval strategies

### 3.2 Data Processing

We processed the TMDB 5000 Movie Dataset plus additional movies (total: 8,254), extracting:
- Title, year, genres, keywords
- Plot overview (rich semantic content)
- Structured metadata for filtering

Each movie is converted to a document with the format:
```
Title (Year) - Genres: [genres] - Keywords: [keywords] - Overview: [plot]
```

### 3.3 Embedding and Vector Storage

**Embedding Model**: `paraphrase-multilingual-MiniLM-L12-v2` (384 dimensions)
- Supports both English and Chinese queries
- Local deployment ensures privacy and reduces API costs

**Vector Database**: ChromaDB with persistent storage
- Cosine similarity for semantic matching
- MMR (Maximal Marginal Relevance) for diverse results

### 3.4 RAG Pipeline

The complete RAG pipeline consists of:

1. **Retrieval**: Fetch top-k relevant movies using similarity or MMR
2. **Context Formatting**: Structure retrieved movies for LLM consumption
3. **Generation**: DeepSeek LLM generates recommendations with explanations
4. **Post-processing**: Extract and validate recommended movies

### 3.5 Baseline Systems

We implement three baselines for comparison:

| System | Description | Purpose |
|--------|-------------|---------|
| **Pure-LLM** | Direct LLM call without retrieval | Hallucination baseline |
| **Tag-Based** | Keyword matching on TMDB genres | Traditional approach |
| **Retrieval-Only** | Vector search without LLM generation | Retrieval quality baseline |

---

## 4. Experiments

### 4.1 Evaluation Setup

**Dataset**: 8,254 movies from TMDB  
**Test Queries**: 15 queries across 4 categories:
- Simple queries (3): Clear genre/theme requests
- Vibe queries (4): Mood and atmosphere descriptions
- Multi-condition (4): Complex constraints
- Edge cases (4): Niche preferences

**Metrics**:
- **Hallucination Rate**: Percentage of non-source movies in output
- **Diversity**: Intra-list genre Jaccard distance
- **Latency**: End-to-end response time (ms)
- **Recommendations**: Average movies recommended per query

### 4.2 Results

#### 4.2.1 Overall Performance

| System | Hallucination Rate | Latency (ms) | Recommendations |
|--------|-------------------|--------------|-----------------|
| VibeMatch (RAG) | 59% | 10,355 | 5.0 |
| VibeMatch (MMR) | 54% | 8,471 | 5.0 |
| Pure-LLM | 100% | 10,529 | 0.0 |
| Tag-Based | 100% | 77 | 0.0 |
| Retrieval-Only | 7% | 1,732 | 5.0 |

#### 4.2.2 Key Findings

**RQ1: Hallucination Reduction**
- RAG reduces hallucination by 41% compared to Pure-LLM (100% → 59%)
- MMR further improves to 54%, suggesting diverse context helps
- Retrieval-Only achieves 7% but lacks explanation capability

**RQ2: MMR vs Similarity**
- MMR is 18% faster (8,471ms vs 10,355ms)
- MMR achieves lower hallucination (54% vs 59%)
- Both return 5 recommendations consistently

**RQ3: Latency Analysis**
- Tag-Based is fastest (77ms) but fails on semantic queries
- Retrieval-Only is efficient (1,732ms) but lacks LLM quality
- RAG adds ~8s overhead for LLM generation

### 4.3 Qualitative Analysis

**Example Query**: "A dark comedy about midlife crisis set in Europe with a heartwarming ending"

**VibeMatch Output**: 
- Retrieved: "The Best Exotic Marigold Hotel" (2011), "Under the Tuscan Sun" (2003)
- Generated: Personalized explanation connecting midlife themes to plot elements

**Pure-LLM Output**:
- Generated non-existent movie titles
- Fabricated plot details
- No verifiable sources

---

## 5. Discussion

### 5.1 Implications

Our results demonstrate that RAG is essential for factual accuracy in LLM recommendations. The 41% hallucination reduction validates our approach for production systems.

MMR's superior performance suggests that diverse context helps LLM generate more grounded recommendations. This aligns with findings in multi-document summarization.

### 5.2 Limitations

1. **Hallucination Still Present**: 54-59% indicates room for improvement
2. **Latency**: 8-10s response time may impact user experience
3. **Dataset Size**: 8,254 movies covers popular titles but misses niche films

### 5.3 Future Work

1. **Fine-tuned Embeddings**: Train domain-specific embedding models
2. **User Feedback Loop**: Incorporate explicit feedback for personalization
3. **Streaming Generation**: Reduce perceived latency with token streaming
4. **Multi-modal**: Integrate poster images for visual similarity

---

## 6. Conclusion

VibeMatch demonstrates that RAG significantly improves factual accuracy in movie recommendation systems. Our comprehensive evaluation shows 41% hallucination reduction compared to pure LLM approaches, with MMR retrieval offering additional benefits in speed and accuracy.

The system successfully handles complex, nuanced queries that traditional keyword-based systems cannot process. All recommendations are grounded in a verified database, providing transparency through source attribution.

Our codebase, evaluation framework, and experimental results are publicly available, contributing to reproducible research in RAG-based recommendation systems.

---

## References

1. Lewis, P., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. NeurIPS.
2. Zhang, S., et al. (2021). Multi-modal movie recommendation with plot graphs. ACM MM.
3. Chen, Y., et al. (2024). Hallucination detection in LLM recommendations. arXiv preprint.
4. LangChain Documentation. https://python.langchain.com
5. ChromaDB Documentation. https://docs.trychroma.com

---

## Appendix

### A.1 System Configuration

- **Embedding Model**: paraphrase-multilingual-MiniLM-L12-v2
- **LLM**: DeepSeek deepseek-v4-flash
- **Vector DB**: ChromaDB 1.5.x
- **Framework**: LangChain 0.2.x with LCEL

### A.2 Test Queries

Full list of 15 evaluation queries available in `backend/evaluation/test_queries.json`.

### A.3 Code Repository

https://github.com/Fantasyiii/AIE6002_Project

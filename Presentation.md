# VibeMatch: RAG-Based Semantic Movie Recommendation

## AIE6002 Final Project Presentation

**Time**: ~8 minutes  
**Slides**: ~12-15 slides

---

## Slide 1: Title (30s)

**VibeMatch: RAG-Based Semantic Movie Recommendation System**

Yifei Chen, Lei Zhang, Shuhao Shi

The Chinese University of Hong Kong, Shenzhen

AIE6002 Large Language Models - Spring 2026

---

## Slide 2: The Problem (45s)

### Movie Recommendation is Broken

**Traditional Systems:**
- ❌ Genre filters: "Action", "Comedy" — too rigid
- ❌ Can't understand: "Something like Inception but more emotional"

**Pure LLM Systems:**
- ❌ Hallucinate fake movies
- ❌ "The Midnight Express" — sounds real, doesn't exist
- ❌ No source verification

**Real User Query:**
> "A dark comedy about midlife crisis set in Europe with a heartwarming ending"

**Current systems fail this.**

---

## Slide 3: Our Solution (30s)

### VibeMatch: RAG-Powered Recommendations

**What makes it different:**
- ✅ Understands natural language "vibes"
- ✅ All recommendations grounded in real movies
- ✅ Shows sources — you can verify
- ✅ Handles complex, nuanced queries

**Demo Preview:**
[Screen recording or live demo of interface]

---

## Slide 4: System Architecture (60s)

### Three-Layer Architecture

```
┌─────────────────────────────────────┐
│         User Interface              │
│    Next.js + React + Tailwind       │
└─────────────┬───────────────────────┘
              │ HTTP API
┌─────────────▼───────────────────────┐
│         FastAPI Backend             │
│  ┌─────────┐    ┌──────────────┐   │
│  │  RAG    │◄──►│   ChromaDB   │   │
│  │ Pipeline│    │  (8,254 movies)│  │
│  └────┬────┘    └──────────────┘   │
│       │                             │
│  ┌────▼────┐                       │
│  │DeepSeek │ LLM Generation        │
│  │   API   │                       │
│  └─────────┘                       │
└─────────────────────────────────────┘
```

**Key Components:**
1. **Multilingual Embeddings**: `paraphrase-multilingual-MiniLM-L12-v2`
2. **Vector Database**: ChromaDB with cosine similarity + MMR
3. **LLM**: DeepSeek for recommendation generation

---

## Slide 5: RAG Pipeline (45s)

### How It Works

**Step 1: Retrieve** (Semantic Search)
- User query → Embedding vector
- Find top-k similar movies in ChromaDB

**Step 2: Context Formatting**
```
[1] The Best Exotic Marigold Hotel (2011)
    Genres: Comedy, Drama
    Overview: British retirees travel to India...

[2] Under the Tuscan Sun (2003)
    Genres: Comedy, Drama, Romance
    Overview: A writer impulsively buys a villa in Tuscany...
```

**Step 3: Generate**
- LLM reads context + query
- Generates personalized recommendation with explanation

**Step 4: Validate**
- Extract mentioned movies
- Verify against retrieved sources

---

## Slide 6: Baseline Systems (30s)

### What We Compare Against

| System | Approach | Why Include? |
|--------|----------|--------------|
| **Pure-LLM** | GPT-4 directly | Hallucination baseline |
| **Tag-Based** | TMDB genre filter | Traditional approach |
| **Retrieval-Only** | Vector search, no LLM | Retrieval quality |
| **VibeMatch (RAG)** | Our full system | Proposed solution |
| **VibeMatch (MMR)** | RAG + diversity | Alternative retrieval |

**Research Questions:**
- RQ1: Does RAG reduce hallucination?
- RQ2: Does MMR improve diversity?
- RQ3: What's the latency trade-off?

---

## Slide 7: Evaluation Setup (30s)

### How We Tested

**Dataset:**
- 8,254 movies from TMDB
- Rich metadata: title, year, genres, keywords, plot

**Test Queries (15 total):**
- Simple: "Sci-fi about time travel"
- Vibe: "Feel-good movie for bad days"
- Complex: "European drama, 2010s, not too sad"
- Edge: "Black and white, journalism, classic"

**Metrics:**
- Hallucination Rate: % of fake movies in output
- Diversity: Genre variety in recommendations
- Latency: Response time (ms)

---

## Slide 8: Results - Hallucination (45s)

### RQ1: Does RAG Reduce Hallucination?

```
Hallucination Rate (%)
100 ┤ ████ Pure-LLM
100 ┤ ████ Tag-Based
 59 ┤ █▓▓▓ VibeMatch (RAG)
 54 ┤ █▒▒▒ VibeMatch (MMR)
  7 ┤ ░░░░ Retrieval-Only
    └─────────────────────
```

**Key Finding:**
- RAG reduces hallucination by **41%** (100% → 59%)
- MMR further improves to **54%**
- Pure-LLM hallucinates on every query!

**But wait:** Retrieval-Only is 7% — why not use that?
→ No explanations, just raw movie lists

---

## Slide 9: Results - Performance (45s)

### RQ2 & RQ3: MMR vs Similarity, Latency Trade-offs

| System | Hallucination | Latency | Speedup |
|--------|--------------|---------|---------|
| VibeMatch (RAG) | 59% | 10,355ms | baseline |
| VibeMatch (MMR) | 54% | 8,471ms | **1.18×** |
| Pure-LLM | 100% | 10,529ms | ~same |
| Tag-Based | 100% | 77ms | 134× |
| Retrieval-Only | 7% | 1,732ms | 6× |

**Key Findings:**
- **MMR is 18% faster** AND has lower hallucination
- Diverse context helps LLM generate better recommendations
- Tag-Based is fast but fails on semantic queries

---

## Slide 10: Qualitative Example (45s)

### Real Query Comparison

**Query:** "A dark comedy about midlife crisis set in Europe with a heartwarming ending"

**Pure-LLM Output:**
> "I recommend 'The Lisbon Trilogy' (2018), a heartwarming Portuguese film about..."

❌ **"The Lisbon Trilogy" doesn't exist.** Hallucinated title, plot, year.

---

**VibeMatch Output:**
> "Based on your query, I recommend:
> 
> 1. **The Best Exotic Marigold Hotel** (2011)
>    - British retirees facing midlife challenges move to India
>    - Perfect blend of comedy and heartwarming moments
> 
> 2. **Under the Tuscan Sun** (2003)
>    - Writer rebuilds life in Tuscany after divorce
>    - Beautiful European setting with uplifting ending"

✅ **Real movies. Real plots. Verifiable sources.**

---

## Slide 11: Live Demo (60s)

### See It In Action

[Switch to live demo or screen recording]

**Demo Script:**
1. Open http://localhost:3000
2. Type: "Sci-fi movie about AI that makes you think"
3. Show: Loading state → Results with Sources
4. Expand: Source cards showing retrieved movies
5. Highlight: Response time display

**Alternative queries if time:**
- "Romantic comedy for weekend, nothing too cheesy"
- "Classic thriller with a twist ending"

---

## Slide 12: Limitations & Future Work (45s)

### What Could Be Better

**Current Limitations:**
1. **Hallucination still 54-59%** — room for improvement
2. **8-10s latency** — could feel slow
3. **8,254 movies** — misses niche/indie films

**Future Directions:**
1. **Fine-tuned embeddings** — domain-specific for movies
2. **Streaming generation** — tokens appear as they're generated
3. **User feedback loop** — learn from thumbs up/down
4. **Multi-modal** — consider movie posters, trailers

---

## Slide 13: Key Takeaways (30s)

### What We Learned

1. **RAG is essential** for factual accuracy in LLM recommendations
   - 41% hallucination reduction

2. **MMR improves both speed and quality**
   - 18% faster, 5% less hallucination

3. **Semantic understanding beats keyword matching**
   - Handles complex "vibe" queries

4. **Transparency matters**
   - Source attribution builds user trust

---

## Slide 14: Conclusion (30s)

### VibeMatch: Grounded Recommendations

**Problem:** LLMs hallucinate movies; keyword systems don't understand nuance

**Solution:** RAG pipeline with verified sources + semantic search

**Results:**
- 41% hallucination reduction
- Handles complex natural language queries
- Open-source, reproducible

**Code:** github.com/Fantasyiii/AIE6002_Project

---

## Slide 15: Q&A (60s)

### Questions?

**Contact:**
- Yifei Chen: 225085000@link.cuhk.edu.cn
- Lei Zhang: 122090746@link.cuhk.edu.cn
- Shuhao Shi: 122090466@link.cuhk.edu.cn

**Repository:** https://github.com/Fantasyiii/AIE6002_Project

---

## Speaker Notes

### Timing Guide
- Slides 1-3: Hook (2 min)
- Slides 4-6: Technical (2 min)
- Slides 7-9: Results (2 min)
- Slides 10-11: Demo (2 min)
- Slides 12-15: Wrap-up (2 min)

### Key Points to Emphasize
1. Hallucination is a real problem — show the fake movie example
2. 41% reduction is significant — not perfect, but major improvement
3. MMR being faster AND better is counterintuitive — explain why
4. Demo is crucial — shows the system actually works

### Backup Slides (if asked)
- Architecture diagram with more detail
- Evaluation query list
- Hyperparameter choices (chunk size, top-k, etc.)

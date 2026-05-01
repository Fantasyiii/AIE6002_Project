"""
FastAPI backend for VibeMatch movie recommendation system.
"""
import os
import time
from typing import List, Optional
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_chain import RAGPipeline, create_pure_llm_chain, create_retrieval_chain

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="VibeMatch API",
    description="RAG-based semantic movie recommendation system",
    version="1.0.0"
)

# CORS configuration for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ChatRequest(BaseModel):
    query: str
    retrieval_mode: str = "similarity"  # "similarity" or "mmr"
    top_k: int = 5
    lambda_mult: Optional[float] = 0.5  # For MMR mode

class ChatResponse(BaseModel):
    query: str
    answer: str
    sources: List[dict]
    retrieval_mode: str
    top_k: int
    latency_ms: float

class HealthResponse(BaseModel):
    status: str
    vector_db_loaded: bool

# Global pipeline instance
pipeline = None

@app.on_event("startup")
async def startup_event():
    """Initialize RAG pipeline on startup."""
    global pipeline
    print("Initializing RAG pipeline...")
    try:
        pipeline = RAGPipeline(retrieval_mode="similarity", top_k=5)
        print("RAG pipeline initialized successfully")
    except Exception as e:
        print(f"Failed to initialize pipeline: {e}")
        print("Please ensure vector store is created by running: python vectorstore.py")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        vector_db_loaded=pipeline is not None
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint for movie recommendations.

    Args:
        request: ChatRequest with query and optional parameters

    Returns:
        ChatResponse with answer, sources, and metadata
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    start_time = time.time()

    try:
        # Update pipeline configuration if needed
        if request.retrieval_mode != pipeline.retrieval_mode or request.top_k != pipeline.top_k:
            global pipeline
            pipeline = RAGPipeline(
                retrieval_mode=request.retrieval_mode,
                top_k=request.top_k
            )

        # Get recommendation
        result = pipeline.recommend(request.query)

        latency_ms = (time.time() - start_time) * 1000

        return ChatResponse(
            query=result["query"],
            answer=result["answer"],
            sources=result["sources"],
            retrieval_mode=result["retrieval_mode"],
            top_k=result["top_k"],
            latency_ms=round(latency_ms, 2)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/baseline/pure-llm")
async def baseline_pure_llm(request: ChatRequest):
    """Pure LLM baseline endpoint (no retrieval)."""
    start_time = time.time()

    try:
        chain = create_pure_llm_chain()
        answer = chain.invoke(request.query)

        latency_ms = (time.time() - start_time) * 1000

        return {
            "query": request.query,
            "answer": answer,
            "sources": [],
            "baseline": "pure-llm",
            "latency_ms": round(latency_ms, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/baseline/retrieval-only")
async def baseline_retrieval_only(request: ChatRequest):
    """Retrieval-only baseline endpoint (no LLM generation)."""
    start_time = time.time()

    try:
        retriever = create_retrieval_chain(
            retrieval_mode=request.retrieval_mode,
            top_k=request.top_k
        )
        docs = retriever(request.query)

        # Format as simple list
        answer = "\n\n".join([
            f"{i+1}. {doc.metadata['title']} ({doc.metadata.get('year', 'N/A')})\n"
            f"Genres: {doc.metadata.get('genres', 'N/A')}\n"
            f"Overview: {doc.metadata.get('overview', 'N/A')[:150]}..."
            for i, doc in enumerate(docs)
        ])

        latency_ms = (time.time() - start_time) * 1000

        return {
            "query": request.query,
            "answer": answer,
            "sources": [
                {
                    "title": doc.metadata["title"],
                    "year": doc.metadata.get("year", "N/A"),
                    "genres": doc.metadata.get("genres", "N/A"),
                }
                for doc in docs
            ],
            "baseline": "retrieval-only",
            "latency_ms": round(latency_ms, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

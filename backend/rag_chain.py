"""
RAG chain implementation using LangChain.
"""
import os
from typing import List, Dict, Optional
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.documents import Document

from prompts import create_rag_prompt, create_pure_llm_prompt
from vectorstore import load_vectorstore, similarity_search, mmr_search

# Load environment variables
load_dotenv()

# Configuration
LLM_TEMPERATURE = 0.7


def get_llm():
    """Initialize LLM (NVIDIA API or OpenAI)."""
    # Try NVIDIA API first
    nvidia_key = os.getenv("NVIDIA_API_KEY")
    if nvidia_key and nvidia_key != "nvapi-":
        return ChatOpenAI(
            model=os.getenv("NVIDIA_MODEL", "minimaxai/minimax-m2.5"),
            temperature=LLM_TEMPERATURE,
            api_key=nvidia_key,
            base_url=os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
            timeout=120,
            max_retries=2
        )
    
    # Fallback to OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=LLM_TEMPERATURE,
            api_key=openai_key
        )
    
    raise ValueError("No API key found. Please set NVIDIA_API_KEY or OPENAI_API_KEY in .env")


def format_docs(docs: List[Document]) -> str:
    """Format retrieved documents into context string."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        formatted.append(
            f"[{i}] {doc.metadata['title']} ({doc.metadata.get('year', 'N/A')})\n"
            f"Genres: {doc.metadata.get('genres', 'N/A')}\n"
            f"Overview: {doc.metadata.get('overview', 'N/A')}\n"
        )
    return "\n---\n".join(formatted)


def create_rag_chain(retrieval_mode: str = "similarity", top_k: int = 5):
    """
    Create RAG chain with specified retrieval mode.
    """
    vectorstore = load_vectorstore()
    if not vectorstore:
        raise ValueError("Vector store not found. Please run vectorstore.init_vectorstore() first.")

    if retrieval_mode == "mmr":
        def _retriever(query):
            return mmr_search(vectorstore, query, k=top_k)
    else:
        def _retriever(query):
            return similarity_search(vectorstore, query, k=top_k)

    retriever = RunnableLambda(_retriever)
    prompt = create_rag_prompt()
    llm = get_llm()

    rag_chain = (
        RunnableParallel({
            "context": retriever | format_docs,
            "query": RunnablePassthrough()
        })
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def create_retrieval_chain(retrieval_mode: str = "similarity", top_k: int = 5):
    """Create retrieval-only chain (no LLM generation)."""
    vectorstore = load_vectorstore()
    if not vectorstore:
        raise ValueError("Vector store not found.")

    if retrieval_mode == "mmr":
        def retriever(query):
            return mmr_search(vectorstore, query, k=top_k)
    else:
        def retriever(query):
            return similarity_search(vectorstore, query, k=top_k)

    return retriever


def create_pure_llm_chain():
    """Create pure LLM chain (no retrieval)."""
    prompt = create_pure_llm_prompt()
    llm = get_llm()

    return (
        RunnablePassthrough()
        | prompt
        | llm
        | StrOutputParser()
    )


class RAGPipeline:
    """High-level RAG pipeline interface."""

    def __init__(self, retrieval_mode: str = "similarity", top_k: int = 5):
        self.retrieval_mode = retrieval_mode
        self.top_k = top_k
        self.chain = create_rag_chain(retrieval_mode, top_k)
        self.retriever = create_retrieval_chain(retrieval_mode, top_k)

    def recommend(self, query: str) -> Dict:
        """Get movie recommendations for a query."""
        sources = self.retriever(query)
        answer = self.chain.invoke(query)

        return {
            "query": query,
            "answer": answer,
            "sources": [
                {
                    "title": doc.metadata["title"],
                    "year": doc.metadata.get("year", "N/A"),
                    "genres": doc.metadata.get("genres", "N/A"),
                    "overview": doc.metadata.get("overview", "N/A")[:200],
                }
                for doc in sources
            ],
            "retrieval_mode": self.retrieval_mode,
            "top_k": self.top_k
        }


if __name__ == "__main__":
    print("Initializing RAG pipeline...")
    pipeline = RAGPipeline(retrieval_mode="similarity", top_k=3)

    test_queries = [
        "sci-fi movie about artificial intelligence",
        "romantic comedy set in New York",
        "dark thriller with a twist ending"
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")

        result = pipeline.recommend(query)
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nSources:")
        for source in result["sources"]:
            print(f"  - {source['title']} ({source['year']})")

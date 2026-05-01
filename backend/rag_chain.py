"""
RAG chain implementation using LangChain.
"""
import os
from typing import List, Dict, Optional
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.documents import Document

from prompts import create_rag_prompt, create_pure_llm_prompt
from vectorstore import load_vectorstore, similarity_search, mmr_search

# Load environment variables
load_dotenv()

# Configuration
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.7


def get_llm() -> ChatOpenAI:
    """Initialize OpenAI LLM."""
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        api_key=os.getenv("OPENAI_API_KEY")
    )


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

    Args:
        retrieval_mode: "similarity" or "mmr"
        top_k: Number of documents to retrieve

    Returns:
        Runnable chain
    """
    # Load vector store
    vectorstore = load_vectorstore()
    if not vectorstore:
        raise ValueError("Vector store not found. Please run vectorstore.init_vectorstore() first.")

    # Select retriever
    if retrieval_mode == "mmr":
        def retriever(query):
            return mmr_search(vectorstore, query, k=top_k)
    else:
        def retriever(query):
            return similarity_search(vectorstore, query, k=top_k)

    # Create prompt
    prompt = create_rag_prompt()

    # Create LLM
    llm = get_llm()

    # Build RAG chain
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
    """
    Create retrieval-only chain (no LLM generation).
    Returns raw retrieved documents.
    """
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
        """
        Get movie recommendations for a query.

        Returns:
            Dict with keys: query, answer, sources, retrieval_mode
        """
        # Get retrieved documents
        sources = self.retriever(query)

        # Generate response
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
    # Test RAG pipeline
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

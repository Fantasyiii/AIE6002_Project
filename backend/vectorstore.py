"""
Vector store management using ChromaDB.
Handles embedding generation and similarity search.
"""
import os
from pathlib import Path
from typing import List, Dict, Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


# Configuration
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "movies"
MODEL_PATH = "./models/all-MiniLM-L6-v2"


def get_embeddings():
    """Initialize local HuggingFace embedding model."""
    model_path = Path(MODEL_PATH)

    if model_path.exists():
        print(f"Loading local model from {model_path}...")
        return HuggingFaceEmbeddings(
            model_name=str(model_path),
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    else:
        print("Local model not found. Falling back to OpenAI embeddings...")
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY")
        )


def load_documents(json_path: str = "./data/movies_processed.json") -> List[Document]:
    """Load processed movie documents from JSON."""
    import json

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    for item in data:
        doc = Document(
            page_content=item["content"],
            metadata=item["metadata"]
        )
        documents.append(doc)

    return documents


def create_vectorstore(
    documents: List[Document],
    persist_dir: str = CHROMA_PERSIST_DIR
) -> Chroma:
    """
    Create and persist ChromaDB vector store from documents.

    Args:
        documents: List of LangChain Document objects
        persist_dir: Directory to persist the vector store

    Returns:
        Chroma vector store instance
    """
    print(f"Initializing embedding model...")
    embeddings = get_embeddings()

    print(f"Creating vector store with {len(documents)} documents...")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name=COLLECTION_NAME
    )

    print(f"Vector store persisted to {persist_dir}")
    return vectorstore


def load_vectorstore(persist_dir: str = CHROMA_PERSIST_DIR) -> Optional[Chroma]:
    """
    Load existing ChromaDB vector store.

    Args:
        persist_dir: Directory where vector store is persisted

    Returns:
        Chroma vector store instance, or None if not found
    """
    if not os.path.exists(persist_dir):
        print(f"Vector store not found at {persist_dir}")
        return None

    print(f"Loading vector store from {persist_dir}...")
    embeddings = get_embeddings()

    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

    count = vectorstore._collection.count()
    print(f"Loaded vector store with {count} documents")
    return vectorstore


def similarity_search(
    vectorstore: Chroma,
    query: str,
    k: int = 5
) -> List[Document]:
    """
    Perform similarity search on vector store.

    Args:
        vectorstore: Chroma vector store instance
        query: User query string
        k: Number of results to return

    Returns:
        List of relevant Document objects
    """
    return vectorstore.similarity_search(query, k=k)


def mmr_search(
    vectorstore: Chroma,
    query: str,
    k: int = 5,
    fetch_k: int = 20,
    lambda_mult: float = 0.5
) -> List[Document]:
    """
    Perform Maximal Marginal Relevance search.

    Args:
        vectorstore: Chroma vector store instance
        query: User query string
        k: Number of results to return
        fetch_k: Number of documents to fetch initially
        lambda_mult: Diversity parameter (0 = max diversity, 1 = max relevance)

    Returns:
        List of diverse and relevant Document objects
    """
    return vectorstore.max_marginal_relevance_search(
        query,
        k=k,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult
    )


def init_vectorstore():
    """Initialize vector store from processed data."""
    # Check if already exists
    if os.path.exists(CHROMA_PERSIST_DIR):
        print("Vector store already exists. Loading...")
        return load_vectorstore()

    # Process data if needed
    data_path = "./data/movies_processed.json"
    if not os.path.exists(data_path):
        print("Processed data not found. Running data processor...")
        from data_processor import process_dataset
        process_dataset()

    # Load and create vector store
    documents = load_documents(data_path)
    return create_vectorstore(documents)


if __name__ == "__main__":
    # Initialize vector store
    vectorstore = init_vectorstore()

    # Test search
    print("\nTesting similarity search...")
    results = similarity_search(vectorstore, "sci-fi movie about space travel", k=3)
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. {doc.metadata['title']}")
        print(f"   Genres: {doc.metadata['genres']}")
        print(f"   Overview: {doc.metadata['overview'][:100]}...")

    print("\nTesting MMR search...")
    results = mmr_search(vectorstore, "sci-fi movie about space travel", k=3)
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. {doc.metadata['title']}")
        print(f"   Genres: {doc.metadata['genres']}")
        print(f"   Overview: {doc.metadata['overview'][:100]}...")

"""
Vector store management using ChromaDB.
Handles embedding generation and similarity search.
"""
import os
from pathlib import Path
from typing import List, Dict, Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


# Configuration
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "movies"
# Multilingual model — supports Chinese and English queries
MODEL_PATH = "./models/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_FALLBACK_PATH = "./models/all-MiniLM-L6-v2"


def get_embeddings():
    """Initialize local HuggingFace embedding model. Prefers multilingual model."""
    multilingual_path = Path("./models/paraphrase-multilingual-MiniLM-L12-v2")
    fallback_path = Path("./models/all-MiniLM-L6-v2")

    if multilingual_path.exists():
        print(f"Loading multilingual model from {multilingual_path}...")
        return HuggingFaceEmbeddings(
            model_name=str(multilingual_path),
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    elif fallback_path.exists():
        print(f"Multilingual model not found. Falling back to {fallback_path}...")
        return HuggingFaceEmbeddings(
            model_name=str(fallback_path),
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    else:
        print("No local model found. Falling back to OpenAI embeddings...")
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY")
        )


def load_documents(json_path: str = None) -> List[Document]:
    """Load processed movie documents. Prefers merged dataset if available."""
    if json_path is None:
        merged = "./data/movies_merged.json"
        json_path = merged if os.path.exists(merged) else "./data/movies_processed.json"
    print(f"Loading documents from {json_path}...")
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
    persist_dir: str = CHROMA_PERSIST_DIR,
    batch_size: int = 5000
) -> Chroma:
    """
    Create and persist ChromaDB vector store from documents.
    Processes documents in batches to avoid ChromaDB batch size limits.

    Args:
        documents: List of LangChain Document objects
        persist_dir: Directory to persist the vector store
        batch_size: Number of documents to process per batch (default 5000)

    Returns:
        Chroma vector store instance
    """
    print(f"Initializing embedding model...")
    embeddings = get_embeddings()

    print(f"Creating vector store with {len(documents)} documents...")
    
    # Create empty vector store first
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )
    
    # Process documents in batches
    total = len(documents)
    for i in range(0, total, batch_size):
        batch = documents[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(total + batch_size - 1)//batch_size}: {len(batch)} documents...")
        vectorstore.add_documents(batch)
    
    # Persist the vector store
    vectorstore.persist()

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

    # Prefer merged dataset, fall back to original
    merged_path = "./data/movies_merged.json"
    original_path = "./data/movies_processed.json"
    if os.path.exists(merged_path):
        data_path = merged_path
    elif os.path.exists(original_path):
        data_path = original_path
    else:
        print("Processed data not found. Running data processor...")
        from data_processor import process_dataset
        process_dataset()
        data_path = original_path

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

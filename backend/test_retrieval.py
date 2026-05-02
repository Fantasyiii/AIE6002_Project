"""
Test retrieval without LLM.
"""
from vectorstore import load_vectorstore, similarity_search, mmr_search

print("Loading vector store...")
vectorstore = load_vectorstore()

if vectorstore:
    print("\nTest Similarity Search:")
    results = similarity_search(vectorstore, "sci-fi movie about space travel", k=3)
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.metadata['title']} ({doc.metadata.get('year', 'N/A')})")
        print(f"   Genres: {doc.metadata.get('genres', 'N/A')}")

    print("\nTest MMR Search:")
    results = mmr_search(vectorstore, "sci-fi movie about space travel", k=3)
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.metadata['title']} ({doc.metadata.get('year', 'N/A')})")
        print(f"   Genres: {doc.metadata.get('genres', 'N/A')}")
else:
    print("Vector store not found!")

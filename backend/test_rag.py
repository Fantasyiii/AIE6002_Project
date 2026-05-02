"""
Simple test for RAG pipeline with NVIDIA API.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Test 1: Direct API call
print("Test 1: Direct NVIDIA API call")
from openai import OpenAI

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
    timeout=60,
    max_retries=2
)

try:
    response = client.chat.completions.create(
        model="minimaxai/minimax-m2.5",
        messages=[{"role": "user", "content": "Recommend one sci-fi movie"}],
        max_tokens=100,
        temperature=0.7
    )
    print(f"Response: {response.choices[0].message.content[:100]}...")
    print("Test 1 PASSED\n")
except Exception as e:
    print(f"Test 1 FAILED: {e}\n")

# Test 2: RAG pipeline
print("Test 2: RAG pipeline")
from rag_chain import RAGPipeline

try:
    pipeline = RAGPipeline(retrieval_mode="similarity", top_k=3)
    result = pipeline.recommend("sci-fi movie about space travel")
    print(f"Query: {result['query']}")
    print(f"Answer: {result['answer'][:200]}...")
    print(f"Sources: {[s['title'] for s in result['sources']]}")
    print("Test 2 PASSED\n")
except Exception as e:
    print(f"Test 2 FAILED: {e}\n")

"""
Prompt templates for RAG movie recommendation.
"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# System prompt for movie recommendation
SYSTEM_PROMPT = """You are an expert film critic and movie recommendation assistant.
Your task is to recommend movies based on the user's query and the retrieved movie context.

STRICT RULES:
1. ONLY recommend movies that are explicitly listed in the provided context. NEVER invent or hallucinate movies.
2. For each recommendation, explain WHY it matches the user's requested "vibe" or criteria.
3. If no movies in the context match the query, honestly state that no suitable recommendations were found.
4. Be concise but insightful in your explanations.
5. Format your response with clear numbering and brief explanations.

CONTEXT (Retrieved Movies):
{context}
"""

# User prompt template
USER_PROMPT = """User Query: {query}

Please recommend movies from the context above that best match this query."""


def create_rag_prompt() -> ChatPromptTemplate:
    """Create the RAG prompt template."""
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT)
    ])


# Prompt for pure LLM baseline (no retrieval context)
PURE_LLM_PROMPT = """You are a movie recommendation assistant.
Recommend movies based on the user's query.

User Query: {query}

Please recommend some movies that match this description."""


def create_pure_llm_prompt() -> ChatPromptTemplate:
    """Create prompt for pure LLM baseline."""
    return ChatPromptTemplate.from_messages([
        ("human", PURE_LLM_PROMPT)
    ])


# Prompt for evaluating hallucinations
HALLUCINATION_CHECK_PROMPT = """You are a fact-checking assistant.
Your task is to identify if the assistant's response contains any hallucinated movies
(movies that don't exist or incorrect plot details).

Retrieved Context:
{context}

Assistant Response:
{response}

List any hallucinated or factually incorrect information. If none, say "No hallucinations detected.""""


def create_hallucination_check_prompt() -> ChatPromptTemplate:
    """Create prompt for hallucination detection."""
    return ChatPromptTemplate.from_messages([
        ("system", "You are a fact-checking assistant."),
        ("human", HALLUCINATION_CHECK_PROMPT)
    ])

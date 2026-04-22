"""rag_synthesize.py — LLM-powered RAG synthesis for natural language Q&A.

Takes a question + retrieved context → natural language answer with citations.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re
from typing import Optional

from src import llm_cache


RAG_SYNTHESIS_PROMPT = """You are a knowledgeable colleague chatting about documents.

Answer the question conversationally. If the context doesn't have what you need, just say so directly — don't hedge.

Keep it brief and natural. No bullet points, no headers, no bold text. Just clear prose.

---
CONTEXT:
{context}
---
QUESTION: {question}

ANSWER:"""


def _get_minimax_key() -> str:
    """Get MiniMax API key from CCR config or openclaw config."""
    ccr_config = os.path.expanduser("~/.claude-code-router/config.json")
    if os.path.exists(ccr_config):
        try:
            with open(ccr_config) as f:
                config = json.load(f)
            for provider in config.get("Providers", []):
                if provider.get("name") == "minimax":
                    return provider.get("api_key", "")
        except Exception:
            pass

    openclaw = os.path.expanduser("~/.openclaw/openclaw.json")
    if os.path.exists(openclaw):
        try:
            with open(openclaw) as f:
                config = json.load(f)
            models = config.get("models", {})
            providers = models.get("providers", {})
            for name, prov in providers.items():
                if name == "minimax":
                    return prov.get("apiKey", "")
        except Exception:
            pass

    return ""


def rag_synthesize(question: str, context_blocks: list[dict], conversation_history: Optional[list[dict]] = None) -> dict:
    """Generate a natural language answer from retrieved context.

    Args:
        question: The user's question
        context_blocks: List of {"text": str, "source": str} from hybrid search
        conversation_history: Optional list of {"role": "user"|"assistant", "content": str}

    Returns:
        dict with:
            - answer: natural language response
            - sources: list of cited sources
            - model: model used
            - cached: whether result was cached
    """
    import httpx

    if not context_blocks:
        return {
            "answer": "I don't have any relevant context to answer that question. Try rephrasing or searching for a different topic.",
            "sources": [],
            "model": "MiniMax-M2.7",
            "cached": False,
        }

    # Build context string from retrieved blocks
    # Clean sources: strip hashes, temp file paths, chunk IDs for readable display
    def _clean_source(source: str) -> str:
        # Remove hash-like IDs (64 char hex) and temp file names
        if not source:
            return "document"
        import re
        # Remove 64-char hex hashes
        if re.match(r'^[a-f0-9]{64}$', source):
            return "document"
        # Remove /tmp/ paths
        if '/tmp/' in source:
            return "document"
        # Remove chunk IDs like b3a2... or similar short hashes
        if len(source) < 20 and re.match(r'^[a-f0-9]+$', source):
            return "document"
        return source

    context_parts = []
    for i, block in enumerate(context_blocks):
        text = block.get("text", "")[:800]  # Truncate long blocks
        source = _clean_source(block.get("source", f"source_{i}"))
        context_parts.append(f"[{source}]\n{text}")

    context = "\n\n---\n\n".join(context_parts)

    # Check cache first (key based on question + first 200 chars of context)
    import hashlib
    cache_key = hashlib.md5(f"rag:{question}:{context[:200]}".encode()).hexdigest()[:24]
    cached = llm_cache.get_cached_response(cache_key, collection="rag_cache")
    if cached:
        cached["cached"] = True
        return cached

    # Build messages
    messages = []
    if conversation_history:
        messages.extend(conversation_history)

    prompt = RAG_SYNTHESIS_PROMPT.format(context=context, question=question)
    messages.append({"role": "user", "content": prompt})

    api_key = _get_minimax_key()
    if not api_key:
        return {
            "answer": "MiniMax API key not found. Please configure your API key in ~/.claude-code-router/config.json or ~/.openclaw/openclaw.json",
            "sources": [b.get("source", "unknown") for b in context_blocks],
            "model": "MiniMax-M2.7",
            "cached": False,
        }

    try:
        with httpx.Client(timeout=60) as client:
            response = client.post(
                "https://api.minimax.io/anthropic/v1/messages",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "x-api-key": api_key,
                },
                json={
                    "model": "MiniMax-M2.7",
                    "messages": messages,
                    "max_tokens": 512,
                },
            )
            response.raise_for_status()
            data = response.json()

        # Extract all text blocks
        text_blocks = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]
        answer = text_blocks[-1] if text_blocks else "I couldn't generate an answer."

        # Extract sources from context
        sources = [b.get("source", "unknown") for b in context_blocks]

        result = {
            "answer": answer,
            "sources": sources,
            "model": "MiniMax-M.2-7",
            "cached": False,
        }

        # Cache the result
        llm_cache.cache_response(cache_key, result, collection="rag_cache", ttl_days=30)

        return result

    except Exception as e:
        return {
            "answer": f"I encountered an error generating an answer: {str(e)}",
            "sources": [b.get("source", "unknown") for b in context_blocks],
            "model": "MiniMax-M2.7",
            "cached": False,
            "error": str(e),
        }


def _detect_self_aware_query(question: str) -> bool:
    """Detect if query is about the system itself (stats, ingestion, etc.)."""
    q = question.lower()
    patterns = [
        r"\bhow many\b", r"\bhow much\b", r"\bcount\b",
        r"\bdocument", r"\bfile", r"\bingest",
        r"\bindex", r"\bstored\b", r"\bentities\b",
        r"\brelation", r"\bschema\b", r"\bsystem\b",
        r"\bwhat.*have.*been\b", r"\bwhat.*ingested\b",
        r"\bwhat.*indexed\b",
    ]
    return any(re.search(p, q) for p in patterns)


def rag_with_query(question: str, top_k: int = 5, max_hops: int = 2, conversation_history: Optional[list[dict]] = None) -> dict:
    """Full RAG pipeline: hybrid search + synthesis.

    Args:
        question: The user's question
        top_k: Number of context blocks to retrieve
        max_hops: Max graph hops for multi-hop queries
        conversation_history: Optional conversation context

    Returns:
        dict with hybrid search results + synthesized answer
    """
    from src.query_router import hybrid_search
    from src import graph_quality

    # Step 0: Self-aware queries — inject system stats as context
    context_blocks = []
    if _detect_self_aware_query(question):
        stats = graph_quality.get_graph_statistics()
        totals = stats.get("totals", {})
        context_blocks.append({
            "text": (
                f"GraphRAG System Status:\n"
                f"- Documents indexed: {totals.get('documents', 0)}\n"
                f"- Entities in graph: {totals.get('entities', 0)}\n"
                f"- Relations stored: {totals.get('relations', 0)}\n"
            ),
            "source": "system_stats",
        })

    # Step 1: Retrieve relevant context
    search_results = hybrid_search(question, top_k=top_k, max_hops=max_hops)

    # Vector results
    for h in search_results.get("vector_results", []):
        context_blocks.append({
            "text": h.get("content", ""),
            "source": h.get("chunk_id", "vector"),
        })

    # Graph results
    for g in search_results.get("graph_results", []):
        for item in g.get("traversal", []):
            if isinstance(item, dict):
                context_blocks.append({
                    "text": item.get("text", "") or item.get("name", ""),
                    "source": item.get("entity_id", "graph"),
                })

    # Document results
    for d in search_results.get("document_results", []):
        context_blocks.append({
            "text": d.get("content", ""),
            "source": d.get("source", "document"),
        })

    # Step 2: Synthesize natural language answer
    synthesis = rag_synthesize(question, context_blocks, conversation_history)

    return {
        **search_results,
        "answer": synthesis.get("answer", ""),
        "sources": synthesis.get("sources", []),
        "answer_cached": synthesis.get("cached", False),
        "model": synthesis.get("model", "MiniMax-M2.7"),
    }


if __name__ == "__main__":
    # Test synthesis with mock context
    test_context = [
        {"text": "Q1 2024 Financial Summary: Revenue $1.2M, Expenses $800K, Net Income $400K.", "source": "financials Q1 2024"},
        {"text": "Product sales increased 25% year over year. Operating costs remained flat.", "source": "financials Q1 2024"},
    ]

    result = rag_synthesize("What was the net income?", test_context)
    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources']}")
    print(f"Cached: {result['cached']}")
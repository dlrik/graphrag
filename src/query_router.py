"""query_router.py — Query classification and intelligent routing.

Classifies incoming natural language queries and routes them to the optimal
retrieval strategy:
- SIMPLE: Direct keyword/vector search (single turn)
- MULTI_HOP: Requires graph traversal (2-3 hops from detected entities)
- BROAD: Wide search across documents (aggregated results)
- HYBRID: Combines multiple strategies
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re

from src import mongo_memory, graph_extractor, llm_cache


QUERY_CLASSIFICATION_PROMPT = """Query: __QUERY__

Classify this query. Respond with ONLY this exact JSON format, no other text:
{"query_type": "SIMPLE", "detected_entities": [], "reasoning": "", "suggested_hops": 1}

query_type options: SIMPLE | MULTI_HOP | BROAD | HYBRID
- SIMPLE = single fact lookup, direct question
- MULTI_HOP = needs graph traversal, relationship questions
- BROAD = exploratory, summary, "show me everything"
- HYBRID = combination

JSON:"""


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


def classify_query(query: str) -> dict:
    """Use MiniMax to classify the query type and detect entities.

    Returns dict with query_type, detected_entities, reasoning, suggested_hops.
    """
    import httpx

    api_key = _get_minimax_key()
    if not api_key:
        return _fallback_classify(query)

    prompt = QUERY_CLASSIFICATION_PROMPT.replace("__QUERY__", query)

    try:
        with httpx.Client(timeout=30) as client:
            response = client.post(
                "https://api.minimax.io/anthropic/v1/messages",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "x-api-key": api_key,
                },
                json={
                    "model": "MiniMax-M2.7",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 256,
                },
            )
            response.raise_for_status()
            data = response.json()

        # Collect ALL text blocks (MiniMax may return thinking blocks too)
        text_blocks = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]
        content = text_blocks[-1] if text_blocks else ""

        if not content:
            return _fallback_classify(query)

        # Try to parse JSON
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Extract JSON from text
        first_brace = content.find("{")
        last_brace = content.rfind("}")
        if first_brace != -1 and last_brace != -1:
            try:
                return json.loads(content[first_brace:last_brace + 1])
            except json.JSONDecodeError:
                pass

        return _fallback_classify(query)

    except Exception:
        return _fallback_classify(query)


def _fallback_classify(query: str) -> dict:
    """Rule-based fallback classification when MiniMax is unavailable."""
    query_lower = query.lower()

    # Detect multi-hop indicators
    multi_hop_patterns = [
        r"\brelat(ed|ions?|ions?)\b", r"\bconnect(ed|ions?)?\b",
        r"\bhow (does|is|do).*(relate|connect)\b",
        r"\bwho (is|are).*(connect|work|colleague)\b",
        r"\bpaths?\b", r"\bhops?\b", r"\btraversal\b",
        r"\bfind.*connect", r"\bmulti\b",
    ]

    # Detect broad search indicators
    broad_patterns = [
        r"\ball\b", r"\bevery\b", r"\bshow all\b",
        r"\blist (all|every)\b", r"\bsummary\b",
        r"\boverview\b", r"\bwhat (have|did|do).*(learn|know|store)\b",
    ]

    # Detect entities (capitalized words or quoted strings)
    entities = re.findall(r'"([^"]+)"|\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', query)
    entities = [e[0] or e[1] for e in entities if e[0] or e[1]]
    entities = [e for e in entities if len(e) > 1]

    multi_hop_score = sum(1 for p in multi_hop_patterns if re.search(p, query_lower))
    broad_score = sum(1 for p in broad_patterns if re.search(p, query_lower))

    if multi_hop_score >= 2:
        query_type = "MULTI_HOP"
        hops = 2
    elif broad_score >= 1:
        query_type = "BROAD"
        hops = 1
    elif len(entities) >= 2:
        query_type = "MULTI_HOP"
        hops = 2
    elif len(entities) == 1:
        query_type = "SIMPLE"
        hops = 1
    else:
        query_type = "HYBRID"
        hops = 2

    return {
        "query_type": query_type,
        "detected_entities": entities[:5],
        "reasoning": "Rule-based fallback classification",
        "suggested_hops": hops,
    }


def hybrid_search(query: str, top_k: int = 5, max_hops: int = 2) -> dict:
    """Execute hybrid search combining vector, graph, and full-text retrieval.

    Args:
        query: Natural language query
        top_k: Number of results per retrieval path
        max_hops: Max graph hops (if multi-hop is detected)

    Returns:
        dict with classification, vector_results, graph_results, document_results
    """
    # Step 1: Classify the query (with caching)
    classification = llm_cache.cached_classify_query(query)
    query_type = classification.get("query_type", "HYBRID")
    detected_entities = classification.get("detected_entities", [])
    hops = min(classification.get("suggested_hops", 2), max_hops)

    results = {
        "query": query,
        "query_type": query_type,
        "classification": classification,
        "vector_results": [],
        "graph_results": [],
        "document_results": [],
        "synthesis": "",
    }

    from src import client as memory_core_client

    # Step 2: Always do vector search (memory-core ChromaDB)
    try:
        vec_hits = memory_core_client.vec_search(query, top_k=top_k)
        results["vector_results"] = [
            {"content": h.get("content", "")[:300], "chunk_id": h.get("chunk_id"), "distance": h.get("distance")}
            for h in vec_hits
        ]
    except Exception as e:
        results["vector_errors"] = str(e)

    # Step 3: Graph traversal for detected entities
    if detected_entities or query_type in ("MULTI_HOP", "HYBRID"):
        graph_hits = []

        for entity_name in detected_entities[:5]:
            ent = mongo_memory.get_canonical_entity(entity_name)
            if not ent:
                ent = mongo_memory.get_entity(name=entity_name)
            if ent:
                try:
                    traversal = mongo_memory.traverse_hops(ent["entity_id"], hops=hops)
                    if traversal:
                        graph_hits.append({
                            "start_entity": entity_name,
                            "entity_id": ent["entity_id"],
                            "traversal": traversal,
                        })
                except Exception as e:
                    print(f"[query_router] traverse_hops error for {entity_name}: {e}")

        results["graph_results"] = graph_hits

    # Step 4: Full-text document search (MongoDB documents)
    if query_type in ("BROAD", "HYBRID"):
        try:
            doc_hits = mongo_memory.search_documents(query, limit=top_k)
            results["document_results"] = [
                {"doc_id": d.get("doc_id"), "content": d.get("content", "")[:200], "source": d.get("source")}
                for d in doc_hits
            ]
        except Exception as e:
            results["document_errors"] = str(e)

    # Step 5: For MULTI_HOP, expand search to entity neighbors
    if query_type == "MULTI_HOP" and not results["graph_results"]:
        # Fall back: try simple entity graph lookup
        try:
            neighbors = memory_core_client.graph_neighbors(query)
            if neighbors:
                results["graph_results"] = [{"direct_neighbors": neighbors[:5]}]
        except Exception:
            pass

    # Step 6: Synthesize results (simple concat for now)
    total_results = (
        len(results["vector_results"])
        + sum(len(g.get("traversal", [])) for g in results["graph_results"])
        + len(results["document_results"])
    )
    results["synthesis"] = f"Found {total_results} results via {query_type} search"

    return results


def detect_entities_in_query(query: str) -> list[str]:
    """Extract entity mentions from a query using simple NLP.

    Returns list of potential entity names found in the query.
    """
    entities = []

    # Exclude common English words that could start sentences
    stopwords = frozenset([
        'what', 'who', 'when', 'where', 'why', 'how', 'which', 'that',
        'this', 'there', 'their', 'they', 'them', 'then', 'than',
        'from', 'with', 'have', 'has', 'had', 'was', 'are', 'been',
        'being', 'will', 'would', 'could', 'should', 'about', 'after',
        'before', 'between', 'into', 'through', 'during', 'above',
        'below', 'over', 'under', 'again', 'further', 'once', 'here',
        'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
        'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too',
        'very', 'just', 'can', 'does', 'did', 'doing', 'now', 'your',
        'you', 'yours', 'me', 'my', 'mine', 'we', 'our', 'ours',
    ])

    # Look for quoted strings first
    quoted = re.findall(r'"([^"]+)"', query)
    entities.extend(quoted)

    # Look for CamelCase and all-caps words like GraphRAG, MongoDB, MiniMax
    # \b[A-Z][a-zA-Z]+\b matches: GraphRAG, MongoDB, MiniMax, etc.
    # but also matches sentence-start words like "What" — we filter those
    camel = re.findall(r'\b[A-Z][a-zA-Z]+\b', query)
    entities.extend([w for w in camel if w.lower() not in stopwords])

    # Also find acronyms: MongoDB, GraphRAG, LLM, API, etc.
    # At least 2 consecutive capitals followed by optional lowercase
    acronym = re.findall(r'\b[A-Z]{2,}[a-z]*\b', query)
    entities.extend([w for w in acronym if w.lower() not in stopwords])

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for e in entities:
        e_lower = e.lower()
        if e_lower not in seen and len(e) > 1:
            seen.add(e_lower)
            unique.append(e)

    return unique[:10]


if __name__ == "__main__":
    # Test queries
    test_queries = [
        "what is GraphRAG",
        "how does Doug's memory system work and what projects is he involved in",
        "show me everything about my knowledge graph",
        "who are Sarah's colleagues",
        "tell me about AI projects",
    ]

    print("Testing query router...\n")
    for q in test_queries:
        classification = classify_query(q)
        print(f"Query: {q}")
        print(f"  Type: {classification.get('query_type')}")
        print(f"  Entities: {classification.get('detected_entities', [])}")
        print(f"  Hops: {classification.get('suggested_hops')}")
        print()

    print("\nTesting hybrid search...")
    result = hybrid_search("how does GraphRAG relate to MongoDB", top_k=3)
    print(f"Query type: {result['query_type']}")
    print(f"Vector hits: {len(result['vector_results'])}")
    print(f"Graph results: {len(result['graph_results'])}")
    print(f"Document results: {len(result['document_results'])}")

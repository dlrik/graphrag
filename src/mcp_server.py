"""mcp_server.py — FastMCP server exposing GraphRAG tools to Claude Code.

Run: source .venv/bin/activate && python3 -m src.mcp_server
Or:   uv run python src/mcp_server.py
"""
import sys, os, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastmcp import FastMCP
from typing import Optional

from src import ingestion, mongo_memory, mongo_ingestion, client

# Initialize FastMCP server
mcp = FastMCP("graphrag")


@mcp.tool()
def nl_query_memory(query: str, top_k: int = 5) -> dict:
    """Natural language query against unified memory. Combines semantic vector search with keyword facts.

    Args:
        query: Natural language question or topic to search for
        top_k: Number of results to return (default 5)

    Returns:
        dict with semantic_hits (vector search results) and fact_hits (keyword search results)
    """
    # Vector search
    vec_hits = client.vec_search(query, top_k=top_k)
    # Also keyword search on facts
    fact_results = client.search_facts(query=query, limit=top_k)

    return {
        "query": query,
        "semantic_hits": [
            {"content": h["content"][:300], "chunk_id": h["chunk_id"], "distance": h.get("distance")}
            for h in vec_hits
        ],
        "fact_hits": [
            {"entity": r["entity"], "content": r["content"][:200], "category": r.get("category")}
            for r in fact_results
        ],
    }


@mcp.tool()
def query_memory(entity: Optional[str] = None, category: Optional[str] = None, limit: int = 20) -> dict:
    """Structured query against memory facts.

    Args:
        entity: Filter by entity name
        category: Filter by category (preference|decision|fact|goal|context|error|learning|entity)
        limit: Max results (default 20)

    Returns:
        dict with matching facts
    """
    results = client.search_facts(entity=entity, category=category, limit=limit)
    return {"entity": entity, "category": category, "results": results, "count": len(results)}


@mcp.tool()
def deep_search_memory(query: str, hops: int = 2, top_k: int = 3) -> dict:
    """Progressive multi-hop graph expansion search. Combines vector search with MongoDB graph traversal.

    Args:
        query: Initial search query
        hops: Number of graph traversal hops (1-3, default 2)
        top_k: Results per hop (default 3)

    Returns:
        dict with initial hits, MongoDB traversal results, and expanded graph context
    """
    # Step 1: initial vector search via memory-core
    initial = client.vec_search(query, top_k=top_k)

    results = {
        "query": query,
        "hops": hops,
        "initial_results": [
            {"content": h["content"][:300], "chunk_id": h["chunk_id"]}
            for h in initial
        ],
        "mongo_traversal": [],
        "graph_expansion": [],
    }

    # Step 2: Extract entities from top results and do MongoDB multi-hop traversal
    initial_entities = []
    for hit in initial[:2]:
        content = hit.get("content", "")
        words = content.split()
        for word in words[:15]:
            word_clean = word.strip(".,!?;:")
            if len(word_clean) > 4 and word_clean not in initial_entities:
                initial_entities.append(word_clean)

    # Perform true multi-hop traversal in MongoDB
    for ent_name in initial_entities[:5]:
        ent = mongo_memory.get_canonical_entity(ent_name)
        if ent:
            entity_id = ent["entity_id"]
            try:
                traversal = mongo_memory.traverse_hops(entity_id, hops=hops)
                if traversal:
                    results["mongo_traversal"].append({
                        "start_entity": ent_name,
                        "start_id": entity_id,
                        "results": traversal,
                    })
            except Exception as e:
                print(f"[mcp] traverse_hops error for {ent_name}: {e}")

    # Step 3: Also use existing memory-core graph for fallback/graph expansion
    seen = set()
    for ent_name in initial_entities[:3]:
        try:
            neighbors = client.graph_neighbors(ent_name)
            if neighbors:
                for n in neighbors[:3]:
                    results["graph_expansion"].append({
                        "anchor": ent_name,
                        "edges": [dict(n) for n in neighbors[:5]]
                    })
        except Exception:
            pass

    return results


@mcp.tool()
def ingest_url(url: str, use_mongo: bool = True) -> dict:
    """Fetch a URL, extract content, extract entities and relationships, and store in memory.

    Args:
        url: The URL to ingest
        use_mongo: Use MongoDB unified memory layer (default True)

    Returns:
        dict with ingestion results (chunks_processed, entities_stored, relations_stored, errors)
    """
    if use_mongo:
        result = mongo_ingestion.ingest_url_mongo(url)
    else:
        result = ingestion.ingest_url(url)
    return {
        "status": "ok" if not result.get("errors") else "partial",
        **result,
    }


@mcp.tool()
def ingest_file(path: str, entity: Optional[str] = None, use_mongo: bool = True) -> dict:
    """Read a local file, extract entities and relationships, and store in memory.

    Args:
        path: Absolute path to the file to ingest
        entity: Optional entity label to tag this content
        use_mongo: Use MongoDB unified memory layer (default True)

    Returns:
        dict with ingestion results
    """
    if use_mongo:
        result = mongo_ingestion.ingest_file_mongo(path, entity=entity)
    else:
        result = ingestion.ingest_file(path, entity=entity)
    return {
        "status": "ok" if not result.get("errors") else "partial",
        **result,
    }


@mcp.tool()
def ingest_conversation(messages: list[dict], session_id: Optional[str] = None, use_mongo: bool = True) -> dict:
    """Ingest a conversation history. Messages should be a list of {role, content} dicts.

    Args:
        messages: List of message dicts with 'role' (user|assistant|system) and 'content' fields
        session_id: Optional session identifier
        use_mongo: Use MongoDB unified memory layer (default True)

    Returns:
        dict with ingestion results
    """
    if use_mongo:
        result = mongo_ingestion.ingest_conversation_mongo(messages, session_id=session_id)
    else:
        result = ingestion.ingest_conversation(messages, session_id=session_id)
    return {
        "status": "ok" if not result.get("errors") else "partial",
        **result,
    }


@mcp.tool()
def graph_traverse(entity: str, hops: int = 1, predicate: Optional[str] = None, use_mongo: bool = True) -> dict:
    """Traverse the knowledge graph from a starting entity.

    Args:
        entity: Starting entity name
        hops: Number of hops (default 1, max 3)
        predicate: Optional filter by predicate type (e.g., MENTIONS, HAS, RELATED_TO)
        use_mongo: Use MongoDB unified memory layer (default True)

    Returns:
        dict with entity and its graph neighbors/relationships
    """
    hops = min(max(1, hops), 3)

    if use_mongo:
        # MongoDB-based multi-hop traversal
        ent = mongo_memory.get_canonical_entity(entity)
        if not ent:
            # Try non-canonical lookup
            ent = mongo_memory.get_entity(name=entity)

        if ent:
            entity_id = ent["entity_id"]
            traversal = mongo_memory.traverse_hops(entity_id, hops=hops)
            return {
                "entity": entity,
                "entity_id": entity_id,
                "hops": hops,
                "traversal": traversal,
                "count": len(traversal),
            }
        return {"entity": entity, "entity_id": None, "hops": hops, "traversal": [], "count": 0, "error": "Entity not found in graph"}
    else:
        # Existing memory-core based traversal
        try:
            neighbors = client.graph_neighbors(entity)
            if predicate:
                neighbors = [n for n in neighbors if n.get("predicate") == predicate]
            return {"entity": entity, "edges": [dict(n) for n in neighbors], "count": len(neighbors)}
        except Exception as e:
            return {"entity": entity, "edges": [], "count": 0, "error": str(e)}


@mcp.tool()
def hybrid_search_memory(query: str, top_k: int = 5, max_hops: int = 2) -> dict:
    """Hybrid search combining vector, graph, and document retrieval with intelligent query routing.

    Classifies the query (SIMPLE/MULTI_HOP/BROAD/HYBRID) and combines retrieval paths:
    - Vector search (ChromaDB via memory-core)
    - Graph traversal (MongoDB multi-hop)
    - Full-text document search (MongoDB)

    Args:
        query: Natural language query
        top_k: Results per path (default 5)
        max_hops: Max graph traversal hops (default 2)

    Returns:
        dict with query_type, classification, vector_results, graph_results, document_results, synthesis
    """
    from src import query_router
    try:
        result = query_router.hybrid_search(query, top_k=top_k, max_hops=max_hops)
        return result
    except Exception as e:
        return {"query": query, "error": str(e), "query_type": "ERROR"}


@mcp.tool()
def classify_query(query: str) -> dict:
    """Classify a natural language query into routing type.

    Args:
        query: The query to classify

    Returns:
        dict with query_type (SIMPLE|MULTI_HOP|BROAD|HYBRID), detected_entities, reasoning, suggested_hops
    """
    from src import query_router
    try:
        return query_router.classify_query(query)
    except Exception as e:
        return {"query_type": "ERROR", "error": str(e)}


@mcp.tool()
def record_query_feedback(query: str, result_entities: list[str] = None,
                           result_relations: list[dict] = None) -> dict:
    """Record successful query results back to the knowledge graph (feedback loop Step 12).

    Args:
        query: The original query
        result_entities: List of entity names found in results
        result_relations: List of {subject, predicate, object} relations found

    Returns:
        dict with status and counts of stored entities/relations
    """
    from src import mongo_ingestion
    try:
        stored = mongo_ingestion.record_query_feedback(
            query=query,
            result_entities=result_entities or [],
            result_relations=result_relations or [],
        )
        return {"status": "ok", "query": query, "stored": stored}
    except Exception as e:
        return {"status": "error", "query": query, "error": str(e)}


@mcp.tool()
def ingest_image(path: str, description: str = "") -> dict:
    """Ingest an image or diagram. Uses GPT-4o vision to extract entities and relations.

    Args:
        path: Absolute path to the image file (JPEG, PNG, WebP, etc.)
        description: Optional text description to guide extraction

    Returns:
        dict with ingestion results including extracted entities and relations
    """
    import base64, json, hashlib

    if not os.path.exists(path):
        return {"status": "error", "error": f"File not found: {path}"}

    try:
        # Get OpenRouter credentials
        ccr_config = os.path.expanduser("~/.claude-code-router/config.json")
        openrouter_key = None
        openrouter_base = "https://openrouter.ai/api/v1/chat/completions"

        if os.path.exists(ccr_config):
            with open(ccr_config) as f:
                config = json.load(f)
            for provider in config.get("Providers", []):
                if provider.get("name") == "openrouter":
                    openrouter_key = provider.get("api_key")
                    break

        if not openrouter_key:
            return {"status": "error", "error": "No OpenRouter API key found"}

        # Read and encode image
        with open(path, "rb") as f:
            img_bytes = f.read()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        mime = "image/jpeg" if path.lower().endswith((".jpg", ".jpeg")) else "image/png"
        data_url = f"data:{mime};base64,{img_base64}"

        # Build vision prompt
        hint = f"\nDescription context: {description}" if description else ""
        prompt = f"""Analyze this image and extract all visible entities and their relationships.

Return ONLY a valid JSON object (no markdown, no explanation):
{{"entities": [{{"name": "...", "type": "PERSON|ORGANIZATION|LOCATION|CONCEPT|TASK|DOCUMENT|..."}}], "relations": [{{"subject": "...", "predicate": "MENTIONS|CONNECTS_TO|HAS|RELATED_TO|USES|BUILDS", "object": "..."}}]}}{hint}

Respond ONLY with the JSON object."""

        # Call GPT-4o via OpenRouter (OpenAI-compatible)
        headers = {
            "Authorization": f"Bearer {openrouter_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "openai/gpt-4o-mini",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
            "max_tokens": 1024,
        }

        import urllib.request
        req = urllib.request.Request(
            openrouter_base,
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())

        content = result["choices"][0]["message"]["content"]

        # Parse JSON response
        extraction = None
        for attempt in [content.strip(), content.strip().strip("```json").strip("```")]:
            try:
                extraction = json.loads(attempt)
                break
            except Exception:
                pass

        if not extraction:
            # Try brace scanning
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                extraction = json.loads(content[start:end])

        entities = extraction.get("entities", []) if extraction else []
        relations = extraction.get("relations", []) if extraction else []

        # Store in MongoDB
        from src import mongo_memory, mongo_ingestion

        img_doc = {
            "doc_id": hashlib.md5(img_bytes).hexdigest()[:16],
            "content": f"Image: {os.path.basename(path)}",
            "source": f"image:{path}",
            "metadata": {"image_path": path, "description": description},
            "media_type": "image",
        }
        mongo_memory.store_document(
            content=json.dumps(img_doc),
            source=f"image:{path}",
            metadata={"type": "image_reference"},
        )

        stored_entities, entity_ids = mongo_ingestion._store_entities_mongo(
            entities, f"image:{path}", "img_0"
        )
        stored_relations = mongo_ingestion._store_relations_mongo(
            relations, entity_ids, f"image:{path}", "img_0"
        )

        return {
            "status": "ok",
            "image_path": path,
            "description": description,
            "entities_stored": len(stored_entities),
            "relations_stored": len(stored_relations),
            "extracted_entities": entities,
            "extracted_relations": relations,
            "vision_model": "openai/gpt-4o-mini",
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}


if __name__ == "__main__":
    # Run the MCP server
    mcp.run()

"""mongo_ingestion.py — MongoDB-backed ingestion pipeline.
Extends the base ingestion pipeline with MongoDB unified memory storage,
entity resolution, and feedback loop support.
"""
import os, sys, hashlib
from typing import Optional

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import graph_extractor
from src import mongo_memory
from src import entity_resolver
from src import client as memory_core_client

# ---------------------------------------------------------------------------
# MongoDB-backed entity/relation storage
# ---------------------------------------------------------------------------

def _resolve_and_store_entity(name: str, entity_type: str, source: str = "") -> str:
    """Resolve entity through entity resolution, then store in MongoDB."""
    entity_id = entity_resolver.resolve_entity(name, entity_type, source=source)
    return entity_id


def _store_entities_mongo(entities: list[dict], source: str, chunk_id: str) -> tuple[list, dict]:
    """Store extracted entities in MongoDB with entity resolution.

    Returns (stored_list, entity_id_map).
    """
    stored = []
    entity_ids = {}

    for ent in entities:
        name = ent.get("name", "").strip()
        ent_type = ent.get("type", "CONCEPT").upper()

        if not name:
            continue

        # Resolve to canonical entity
        entity_id = _resolve_and_store_entity(name, ent_type, source=source)
        entity_ids[name] = entity_id

        # Store canonical entity in MongoDB
        try:
            mongo_memory.store_entity(
                name=name,
                entity_type=ent_type,
                source=source,
            )
            stored.append(f"entity:{entity_id}")
        except Exception as e:
            print(f"[mongo_ingest] store_entity error for {name}: {e}")

    return stored, entity_ids


def _store_relations_mongo(relations: list[dict], entity_ids: dict, source: str, chunk_id: str) -> list[str]:
    """Store extracted relationships in MongoDB as triplets."""
    stored = []

    for rel in relations:
        subject = rel.get("subject", "").strip()
        predicate = rel.get("predicate", "RELATED_TO").strip().upper()
        obj = rel.get("object", "").strip()

        if not subject or not predicate or not obj:
            continue

        try:
            relation_id = mongo_memory.store_relation(
                subject=subject,
                predicate=predicate,
                object=obj,
                weight=1.0,
                source=f"graphrag:{source}",
                subject_id=entity_ids.get(subject),
                object_id=entity_ids.get(obj),
            )
            stored.append(f"relation:{relation_id}")
        except Exception as e:
            print(f"[mongo_ingest] store_relation error: {subject} --{predicate}--> {obj}: {e}")

    return stored


def _store_document_mongo(text: str, source: str, metadata: dict = None) -> str:
    """Store raw document in MongoDB documents collection."""
    try:
        return mongo_memory.store_document(text, source, metadata)
    except Exception as e:
        print(f"[mongo_ingest] store_document error: {e}")
        return ""


# ---------------------------------------------------------------------------
# MongoDB-backed ingestion pipeline
# ---------------------------------------------------------------------------

def ingest_url_mongo(url: str) -> dict:
    """Ingest URL with MongoDB storage and entity resolution."""
    from src import ingestion as base_ingestion

    text = base_ingestion.extract_from_url(url)
    if not text:
        return {"url": url, "chunks_processed": 0, "entities_stored": 0, "relations_stored": 0, "errors": ["Failed to extract text from URL"]}

    return _ingest_text_mongo(text, source=f"url:{url}", entity=None)


def ingest_file_mongo(path: str, entity: Optional[str] = None) -> dict:
    """Ingest file with MongoDB storage and entity resolution."""
    from src import ingestion as base_ingestion

    text = base_ingestion.extract_from_file(path)
    if not text:
        return {"file": path, "chunks_processed": 0, "entities_stored": 0, "relations_stored": 0, "errors": ["Failed to extract text from file"]}

    return _ingest_text_mongo(text, source=f"file:{path}", entity=entity)


def ingest_conversation_mongo(messages: list[dict], session_id: Optional[str] = None) -> dict:
    """Ingest conversation with MongoDB storage and entity resolution."""
    from src import ingestion as base_ingestion

    text = base_ingestion.extract_from_conversation(messages)
    if not text:
        return {"session_id": session_id, "chunks_processed": 0, "entities_stored": 0, "relations_stored": 0, "errors": ["No text extracted from conversation"]}

    return _ingest_text_mongo(text, source=f"conversation:{session_id or 'unknown'}", entity=None)


def _ingest_text_mongo(text: str, source: str, entity: Optional[str] = None) -> dict:
    """Core MongoDB ingestion: clean → chunk → extract → resolve → store."""
    from src import ingestion as base_ingestion

    result = {
        "source": source,
        "chunks_processed": 0,
        "entities_stored": 0,
        "relations_stored": 0,
        "errors": [],
    }

    # Clean
    text = base_ingestion.clean_text(text)
    if not text:
        result["errors"].append("Text empty after cleaning")
        return result

    # Store raw document in MongoDB
    doc_id = _store_document_mongo(text, source)
    if doc_id:
        result["doc_id"] = doc_id

    # Chunk
    chunks = base_ingestion.chunk_text(text)
    if not chunks:
        result["errors"].append("No chunks generated")
        return result

    result["chunks_processed"] = len(chunks)

    # Process each chunk
    all_entity_ids = {}

    for i, chunk in enumerate(chunks):
        chunk_id = f"{hashlib.md5(source.encode()).hexdigest()[:8]}_{i}"

        # Graph extraction via LLM (MiniMax)
        extraction = graph_extractor.extract(chunk)
        entities = extraction.get("entities", [])
        relations = extraction.get("relations", [])

        if extraction.get("error"):
            result["errors"].append(f"extraction chunk {i}: {extraction['error']}")

        # Deduplicate entities within this chunk
        seen_names = set()
        unique_entities = []
        for e in entities:
            name = e.get("name", "").strip()
            if name and name not in seen_names:
                seen_names.add(name)
                unique_entities.append(e)

        # Store entities in MongoDB (with entity resolution)
        stored_entities, entity_ids = _store_entities_mongo(unique_entities, source, chunk_id)
        all_entity_ids.update(entity_ids)
        result["entities_stored"] += len(stored_entities)

        # Store relations in MongoDB
        stored_relations = _store_relations_mongo(relations, entity_ids, source, chunk_id)
        result["relations_stored"] += len(stored_relations)

    result["total_unique_entities"] = len(all_entity_ids)
    return result


# ---------------------------------------------------------------------------
# Feedback loop: store query results back to graph
# ---------------------------------------------------------------------------

def record_query_feedback(query: str, result_entities: list[str],
                         result_relations: list[dict], source: str = "query_feedback") -> dict:
    """Store query results as new knowledge in the graph (Step 12 in diagram).

    This enables the feedback loop where successful retrievals are stored
    back into the knowledge graph.
    """
    stored = {"entities": 0, "relations": 0}

    # Store entities from query results
    for ent_name in result_entities:
        try:
            mongo_memory.store_entity(ent_name, "CONCEPT", source=source)
            stored["entities"] += 1
        except Exception:
            pass

    # Store relations from query results
    for rel in result_relations:
        subject = rel.get("subject", "")
        predicate = rel.get("predicate", "RELATED_TO").upper()
        obj = rel.get("object", "")
        if subject and obj:
            try:
                mongo_memory.store_relation(subject, predicate, obj, source=source)
                stored["relations"] += 1
            except Exception:
                pass

    return stored


def update_graph_from_query(query: str, hits: list[dict]) -> dict:
    """Update the knowledge graph based on query results (Step 13 in diagram).

    Analyzes query results and adds new entities/relations discovered.
    """
    stored = {"entities": 0, "relations": 0}

    for hit in hits:
        content = hit.get("content", "") or hit.get("text", "")
        if not content:
            continue

        # Extract entities from result content
        extraction = graph_extractor.extract(content[:1000])
        entities = extraction.get("entities", [])
        relations = extraction.get("relations", [])

        seen_names = set()
        unique_entities = [e for e in entities if e.get("name", "") and e["name"] not in seen_names and not seen_names.add(e["name"])]

        for ent in unique_entities:
            try:
                mongo_memory.store_entity(
                    ent.get("name", ""),
                    ent.get("type", "CONCEPT").upper(),
                    source=f"query_update:{query[:50]}",
                )
                stored["entities"] += 1
            except Exception:
                pass

        for rel in relations:
            try:
                mongo_memory.store_relation(
                    rel.get("subject", ""),
                    rel.get("predicate", "RELATED_TO").upper(),
                    rel.get("object", ""),
                    source=f"query_update:{query[:50]}",
                )
                stored["relations"] += 1
            except Exception:
                pass

    return stored


# ---------------------------------------------------------------------------
# Traverse with feedback
# ---------------------------------------------------------------------------

def traverse_with_feedback(start_entity: str, hops: int = 2,
                          record: bool = True) -> dict:
    """Multi-hop graph traversal with optional feedback recording.

    Traverses the graph and optionally records the traversal results
    back into the graph (Steps 12-13).
    """
    # Get canonical entity ID
    canonical = mongo_memory.get_canonical_entity(start_entity)
    start_id = canonical["entity_id"] if canonical else None

    if not start_id:
        # Entity not in graph yet - store it first
        start_id = mongo_memory.store_entity(start_entity, "CONCEPT", source="traverse_feedback")

    # Perform traversal
    traversal_results = mongo_memory.traverse_hops(start_id, hops=hops)

    if record and traversal_results:
        # Record traversal as a knowledge pattern
        for r in traversal_results:
            try:
                mongo_memory.store_relation(
                    subject=start_entity,
                    predicate="CONNECTED_VIA",
                    object=r["entity"],
                    weight=0.5,
                    source=f"traversal:{hops}_hops",
                )
            except Exception:
                pass

    return {
        "start_entity": start_entity,
        "start_id": start_id,
        "hops": hops,
        "traversal": traversal_results,
        "count": len(traversal_results),
    }


if __name__ == "__main__":
    import json

    # Initialize MongoDB
    mongo_memory.init()

    # Test: ingest a URL
    print("Testing MongoDB ingestion...")
    r = ingest_url_mongo("https://example.com")
    print(f"URL ingest result: {json.dumps(r, indent=2)}")

    # Test: traverse with feedback
    print("\nTesting traverse with feedback...")
    traverse_result = traverse_with_feedback("Example Domain", hops=2)
    print(f"Traversal: {json.dumps(traverse_result, indent=2)}")

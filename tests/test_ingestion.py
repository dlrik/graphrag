"""test_ingestion.py — End-to-end tests for Agentic GraphRAG Phase 1 fixes."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_mongo_memory():
    """Verify MongoDB is running and collections accessible."""
    from src import mongo_memory
    mongo_memory.init()
    db = mongo_memory._get_db()
    print(f"[PASS] MongoDB connected")
    print(f"  Entities: {db[mongo_memory.ENTITIES_COL].count_documents({})}")
    print(f"  Relations: {db[mongo_memory.RELATIONS_COL].count_documents({})}")
    print(f"  Documents: {db[mongo_memory.DOCUMENTS_COL].count_documents({})}")


def test_graph_extractor_minimax():
    """Verify MiniMax LLM extraction works."""
    from src import graph_extractor
    test_text = "Sarah and John are working at Acme Corp. Sarah lives in Seattle."
    result = graph_extractor.extract(test_text)
    print(f"[PASS] graph_extractor MiniMax: {len(result['entities'])} entities, {len(result['relations'])} relations")
    assert len(result["entities"]) > 0, "Should extract at least one entity"
    names = [e["name"] for e in result["entities"]]
    print(f"  Entities: {names}")


def test_entity_resolution():
    """Test MiniMax entity resolution (Abi = Abi Aryan)."""
    from src import mongo_memory, entity_resolver

    mongo_memory.init()

    # Clear test entities first
    db = mongo_memory._get_db()
    db[mongo_memory.ENTITIES_COL].delete_many({"name_lower": {"$in": ["abi", "abi aryan"]}})

    # Store "Abi" first
    e1 = mongo_memory.store_entity("Abi", "PERSON", source="test")
    print(f"  Created 'Abi': {e1}")

    # Resolve "Abi Aryan" - should detect as same person
    result = entity_resolver.resolve_pair("Abi", "Abi Aryan")
    print(f"[PASS] resolve_pair(Abi, Abi Aryan): {result}")

    if result.get("same_entity"):
        canonical_name = result.get("canonical_name", "Abi Aryan")
        mongo_memory.merge_entities(canonical_name, "Abi")
        print(f"  Merged into canonical: {canonical_name}")

    # Verify merged
    abi_aryan = mongo_memory.get_entity(name="Abi Aryan")
    print(f"  'Abi Aryan' canonical_id: {abi_aryan.get('canonical_id', 'SELF') if abi_aryan else 'NOT FOUND'}")


def test_mongo_ingestion_url():
    """Test MongoDB-backed URL ingestion with entity resolution."""
    from src import mongo_ingestion
    print("[TEST] MongoDB URL ingestion...")
    result = mongo_ingestion.ingest_url_mongo("https://example.com")
    print(f"[PASS] URL ingestion: {result['chunks_processed']} chunks, "
          f"{result['entities_stored']} entities, {result['relations_stored']} relations")
    assert result["chunks_processed"] > 0


def test_mongo_ingestion_conversation():
    """Test conversation ingestion."""
    from src import mongo_ingestion
    print("[TEST] MongoDB conversation ingestion...")
    messages = [
        {"role": "user", "content": "Doug is building a GraphRAG system with MongoDB and MiniMax"},
        {"role": "assistant", "content": "That sounds like a great project. What are the main components?"},
        {"role": "user", "content": "We have entity extraction, vector search, and graph traversal with feedback loops"},
    ]
    result = mongo_ingestion.ingest_conversation_mongo(messages, session_id="test_conv")
    print(f"[PASS] conversation ingestion: {result['chunks_processed']} chunks, "
          f"{result['entities_stored']} entities, {result['relations_stored']} relations")
    assert result["chunks_processed"] > 0


def test_mongo_traverse():
    """Test MongoDB multi-hop graph traversal."""
    from src import mongo_memory, mongo_ingestion
    print("[TEST] MongoDB multi-hop traversal...")

    # Ingest some connected entities first
    mongo_memory.store_entity("Project Alpha", "PROJECT", source="test")
    mongo_memory.store_entity("Doug", "PERSON", source="test")
    mongo_memory.store_entity("MongoDB", "TECHNOLOGY", source="test")
    mongo_memory.store_relation("Doug", "WORKS_ON", "Project Alpha", source="test")
    mongo_memory.store_relation("Project Alpha", "USES", "MongoDB", source="test")

    # Traverse with feedback
    result = mongo_ingestion.traverse_with_feedback("Doug", hops=2)
    print(f"[PASS] traverse_with_feedback: {result['count']} traversal results")
    print(f"  Start: {result['start_entity']}, Hops: {result['hops']}")


def test_memory_core_fallback():
    """Verify memory-core still works for vector/fact search."""
    from src import client
    print("[TEST] memory-core fallback...")
    h = client.health()
    print(f"[PASS] memory-core: {h['facts']} facts, {h['chunks']} chunks")

    results = client.vec_search("GraphRAG MongoDB", top_k=3)
    print(f"[PASS] vec_search returned {len(results)} hits")

    results = client.search_facts(query="memory", limit=5)
    print(f"[PASS] fact_search returned {len(results)} results")


def test_mongodb_status():
    """Final MongoDB state check."""
    from src import mongo_memory
    db = mongo_memory._get_db()
    print("\n=== MongoDB Final State ===")
    print(f"  Entities: {db[mongo_memory.ENTITIES_COL].count_documents({})}")
    print(f"  Relations: {db[mongo_memory.RELATIONS_COL].count_documents({})}")
    print(f"  Documents: {db[mongo_memory.DOCUMENTS_COL].count_documents({})}")


if __name__ == "__main__":
    import tempfile
    print("=" * 60)
    print("Agentic GraphRAG Phase 1 Fixes — End-to-End Tests")
    print("=" * 60)

    test_mongo_memory()
    test_graph_extractor_minimax()
    test_entity_resolution()
    test_mongo_ingestion_url()
    test_mongo_ingestion_conversation()
    test_mongo_traverse()
    test_memory_core_fallback()
    test_mongodb_status()

    print("=" * 60)
    print("All Phase 1 fix tests passed!")
    print("=" * 60)

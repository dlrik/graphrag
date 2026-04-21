"""test_phase2.py — Phase 2 tests for Agentic GraphRAG."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_query_router():
    """Test query classification and hybrid search."""
    from src import query_router

    test_queries = [
        ("what is GraphRAG", "SIMPLE"),
        ("how does X relate to Y", "MULTI_HOP"),
        ("show me everything", "BROAD"),
        ("what does Doug know about AI", "HYBRID"),
    ]

    print("\n=== Query Router Tests ===")
    for query, expected_type in test_queries:
        result = query_router.classify_query(query)
        print(f"Query: {query}")
        print(f"  Type: {result.get('query_type')} (expected {expected_type})")
        print(f"  Entities: {result.get('detected_entities', [])}")
        print(f"  Hops: {result.get('suggested_hops')}")

        # Test hybrid search
        hs = query_router.hybrid_search(query, top_k=3)
        print(f"  Hybrid: {hs['query_type']} - {len(hs['vector_results'])} vec, {len(hs['graph_results'])} graph, {len(hs['document_results'])} doc")
        print()


def test_observability():
    """Test observability logging."""
    from src import observability

    print("\n=== Observability Tests ===")

    # Log some test events
    observability.log_info("TEST_RUN", test="phase2")
    observability.log_entity_resolution("Test", "Test2", True, 0.9, "Test", 50.0)

    # Get metrics summary
    summary = observability.get_metrics_summary()
    print(f"Metrics summary: {summary}")

    # Get recent logs
    recent = observability.get_recent_logs(lines=5)
    print(f"Recent logs: {len(recent)} entries")


def test_mongodb_state():
    """Check MongoDB state after Phase 1 and 2 tests."""
    from src import mongo_memory

    print("\n=== MongoDB State ===")
    db = mongo_memory._get_db()

    entities = db[mongo_memory.ENTITIES_COL].count_documents({})
    relations = db[mongo_memory.RELATIONS_COL].count_documents({})
    documents = db[mongo_memory.DOCUMENTS_COL].count_documents({})

    print(f"Entities: {entities}")
    print(f"Relations: {relations}")
    print(f"Documents: {documents}")

    # Show some sample entities
    print("\nSample entities:")
    for ent in db[mongo_memory.ENTITIES_COL].find({}).limit(5):
        print(f"  {ent['name']} ({ent['entity_type']}) - canonical: {ent.get('canonical_id', 'SELF')}")


def test_memory_core():
    """Check memory-core is still functioning."""
    from src import client

    print("\n=== Memory Core State ===")
    h = client.health()
    print(f"Health: {h}")

    results = client.vec_search("GraphRAG memory", top_k=3)
    print(f"Vector search: {len(results)} hits")

    results = client.search_facts(query="memory", limit=5)
    print(f"Fact search: {len(results)} results")


def test_feedback_loop():
    """Test feedback loop recording."""
    from src import mongo_ingestion, mongo_memory

    print("\n=== Feedback Loop Test ===")

    # Record a query feedback
    result = mongo_ingestion.record_query_feedback(
        query="what is the memory system",
        result_entities=["memory system", "MongoDB"],
        result_relations=[
            {"subject": "memory system", "predicate": "USES", "object": "MongoDB"},
            {"subject": "memory system", "predicate": "USES", "object": "ChromaDB"},
        ],
    )
    print(f"Feedback recorded: {result}")


def test_multi_document_ingestion():
    """Test ingestion from multiple sources."""
    from src import mongo_ingestion

    print("\n=== Multi-Document Ingestion ===")

    # Ingest a file
    test_content = """
    Doug is building a unified memory system called GraphRAG.
    It combines vector search with knowledge graphs.
    The system uses MongoDB for storage and MiniMax for entity extraction.
    Sarah is helping with the frontend integration.
    """

    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(test_content)
        tmp_path = f.name

    try:
        result = mongo_ingestion.ingest_file_mongo(tmp_path, entity="Doug")
        print(f"File ingest: {result['chunks_processed']} chunks, {result['entities_stored']} entities, {result['relations_stored']} relations")
    finally:
        os.unlink(tmp_path)


def test_ambiguous_entity_query():
    """Test queries with ambiguous entity names."""
    from src import query_router, mongo_memory

    print("\n=== Ambiguous Entity Query Test ===")

    # Store two similar entities
    mongo_memory.store_entity("Abi", "PERSON", source="test")
    mongo_memory.store_entity("Abi Aryan", "PERSON", source="test")
    mongo_memory.store_relation("Abi", "SIBLING_OF", "Abi Aryan", source="test")

    # Query that mentions "Abi"
    result = query_router.hybrid_search("tell me about Abi and their work", top_k=3)
    print(f"Query 'Abi': {result['query_type']}")
    print(f"  Entities found: {result['classification'].get('detected_entities', [])}")


if __name__ == "__main__":
    print("=" * 60)
    print("Agentic GraphRAG Phase 2 — Tests")
    print("=" * 60)

    test_query_router()
    test_observability()
    test_mongodb_state()
    test_memory_core()
    test_feedback_loop()
    test_multi_document_ingestion()
    test_ambiguous_entity_query()

    print("\n" + "=" * 60)
    print("Phase 2 tests complete!")
    print("=" * 60)

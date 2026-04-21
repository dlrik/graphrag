"""benchmark_queries.py — Query latency benchmarking."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import mongo_memory, query_router, mongo_ingestion

def benchmark(name, fn, runs=3):
    times = []
    for _ in range(runs):
        start = time.time()
        fn()
        times.append((time.time() - start) * 1000)
    avg = sum(times) / len(times)
    print(f"{name}: {avg:.1f}ms avg ({runs} runs)")
    return avg

def main():
    db = mongo_memory._get_db()

    # Get sample entity IDs
    entities = list(db[mongo_memory.ENTITIES_COL].find({}).limit(5))
    entity_ids = [e["entity_id"] for e in entities]

    print("=== Query Latency Benchmarks ===\n")

    # Graph traversal
    if entity_ids:
        eid = entity_ids[2]
        name = db[mongo_memory.ENTITIES_COL].find_one({"entity_id": eid})["name"]
        benchmark(f"traverse_hops({name}, hops=2)", lambda: mongo_memory.traverse_hops(eid, hops=2))

    # Entity lookup
    if entities:
        name = entities[0]["name_lower"]
        benchmark("get_canonical_entity", lambda: mongo_memory.get_canonical_entity(name))

    # Hybrid search (fast path - no LLM call for classification)
    benchmark("hybrid_search (vector+graph+doc)", lambda: query_router.hybrid_search(
        "what is GraphRAG", top_k=3, max_hops=2
    ))

    # Graph neighbors
    if entity_ids:
        benchmark("graph_neighbors", lambda: mongo_memory.graph_neighbors(entity_ids[0]))

    # Document search
    benchmark("search_documents", lambda: mongo_memory.search_documents("memory", limit=10))

    # Entity search
    benchmark("entity_search", lambda: mongo_memory.entity_search("GraphRAG", limit=5))

    # Detect communities (graph quality)
    benchmark("detect_communities", lambda: mongo_memory.detect_communities() if hasattr(mongo_memory, 'detect_communities') else None)

    # Stats query
    from src.graph_quality import get_graph_statistics
    benchmark("get_graph_statistics", lambda: get_graph_statistics())

    print("\n=== LLM-Dependent Queries ===\n")

    # These involve network calls - show individual times
    for query in [
        "what is GraphRAG",
        "how does MiniMax relate to MongoDB",
        "show me everything about Doug",
    ]:
        start = time.time()
        result = query_router.classify_query(query)
        elapsed = (time.time() - start) * 1000
        print(f"classify_query('{query[:30]}...'): {elapsed:.0f}ms -> {result.get('query_type')}")

if __name__ == "__main__":
    main()
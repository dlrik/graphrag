#!/usr/bin/env python3
"""graphrag — CLI for Agentic GraphRAG unified memory.

Usage:
    graphrag ask "what is GraphRAG"
    graphrag ingest file notes.md
    graphrag ingest url https://example.com
    graphrag neighbors "Doug" --hops 2
    graphrag stats
    graphrag cache clear
    graphrag --help

Requires: MongoDB running at localhost:27017, memory-core at localhost:8765
"""
import sys, os, argparse, json, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import mongo_memory, mongo_ingestion, client as mc_client, llm_cache
from src.graph_quality import get_graph_statistics, find_duplicate_entities, detect_communities
from src.query_router import hybrid_search


def cmd_ask(query: str, top_k: int, max_hops: int):
    """Ask a question against the unified memory."""
    print(f"Query: {query}")
    t0 = time.time()
    result = hybrid_search(query, top_k=top_k, max_hops=max_hops)
    classification = result.get("classification", {})
    qtype = classification.get("query_type", "?")
    elapsed = time.time() - t0

    print(f"Type: {qtype} ({elapsed:.1f}s)")
    entities = classification.get("detected_entities", [])
    if entities:
        print(f"Entities: {', '.join(entities)}")

    print(f"\nResults:")

    vec = result.get("vector_results", [])
    if vec:
        print(f"\n  Semantic (ChromaDB): {len(vec)} hits")
        for h in vec[:3]:
            print(f"    • {h.get('content', '')[:100]}...")

    graph = result.get("graph_results", [])
    if graph:
        print(f"\n  Graph (MongoDB): {len(graph)} paths")
        for g in graph[:3]:
            name = g.get("start_entity", "?")
            traversals = g.get("traversal", [])
            print(f"    • {name} → {len(traversals)} connected entities")

    docs = result.get("document_results", [])
    if docs:
        print(f"\n  Documents (MongoDB): {len(docs)} hits")
        for d in docs[:3]:
            print(f"    • {d.get('source', 'unknown')}: {d.get('content', '')[:80]}...")


def cmd_ingest_file(path: str, entity: str = None):
    """Ingest a local file into the knowledge graph."""
    if not os.path.exists(path):
        print(f"Error: file not found: {path}")
        return 1

    print(f"Ingesting: {path}")
    t0 = time.time()
    result = mongo_ingestion.ingest_file_mongo(path, entity=entity)
    elapsed = time.time() - t0

    print(f"Done ({elapsed:.1f}s):")
    print(f"  Chunks: {result.get('chunks_processed', 0)}")
    print(f"  Entities: {result.get('entities_stored', 0)}")
    print(f"  Relations: {result.get('relations_stored', 0)}")
    if result.get("errors"):
        print(f"  Errors: {result['errors'][:3]}")
    return 0


def cmd_ingest_url(url: str):
    """Ingest a URL into the knowledge graph."""
    print(f"Fetching: {url}")
    t0 = time.time()
    result = mongo_ingestion.ingest_url_mongo(url)
    elapsed = time.time() - t0

    print(f"Done ({elapsed:.1f}s):")
    print(f"  Chunks: {result.get('chunks_processed', 0)}")
    print(f"  Entities: {result.get('entities_stored', 0)}")
    print(f"  Relations: {result.get('relations_stored', 0)}")
    if result.get("errors"):
        print(f"  Errors: {result['errors'][:3]}")
    return 0


def cmd_neighbors(entity: str, hops: int = 1, predicate: str = None):
    """Show graph neighbors of an entity."""
    ent = mongo_memory.get_canonical_entity(entity)
    if not ent:
        ent = mongo_memory.get_entity(name=entity)

    if not ent:
        print(f"Entity not found: {entity}")
        return 1

    print(f"Entity: {ent['name']} ({ent['entity_type']})")
    print(f"ID: {ent['entity_id']}")

    neighbors = mongo_memory.graph_neighbors(ent["entity_id"], max_depth=hops)
    if not neighbors:
        print("  No connections found")
        return 0

    # Group by relation type
    by_pred = {}
    for n in neighbors:
        p = n.get("predicate", "?")
        by_pred.setdefault(p, []).append(n)

    print(f"\n{len(neighbors)} connections:")
    for pred, items in by_pred.items():
        print(f"  {pred}: {len(items)}")
        for item in items[:3]:
            s = item.get("subject", "?")
            o = item.get("object", "?")
            print(f"    • {s} → {o}")


def cmd_stats():
    """Show graph statistics."""
    stats = get_graph_statistics()
    tot = stats.get("totals", {})

    print("=== Graph Statistics ===")
    print(f"Entities:   {tot.get('entities', 0)}")
    print(f"Relations:  {tot.get('relations', 0)}")
    print(f"Documents:  {tot.get('documents', 0)}")

    er = stats.get("entity_resolution", {})
    print(f"\nEntity Resolution:")
    print(f"  Canonical: {er.get('canonical_entities', 0)}")
    print(f"  Aliased:  {er.get('aliased_entities', 0)}")

    # Entity types
    et = stats.get("entity_types", {})
    if et:
        print(f"\nEntity Types:")
        for t, c in sorted(et.items(), key=lambda x: -x[1])[:8]:
            print(f"  {t}: {c}")

    # Predicates
    pred = stats.get("predicates", {})
    if pred:
        print(f"\nTop Predicates:")
        for p, c in sorted(pred.items(), key=lambda x: -x[1])[:6]:
            print(f"  {p}: {c}")

    # Cache stats
    cache_stats = llm_cache.get_cache_stats()
    print(f"\nLLM Cache:")
    print(f"  Cached queries: {cache_stats['total_cached']}")
    for t, d in cache_stats.get("by_type", {}).items():
        print(f"  {t}: {d['count']} entries, {d['total_hits']} hits")


def cmd_duplicate():
    """Find potential duplicate entities."""
    dups = find_duplicate_entities()
    if not dups:
        print("No potential duplicates found")
        return 0

    print(f"Found {len(dups)} potential duplicates:")
    for d in dups:
        a = d["entity_a"]
        b = d["entity_b"]
        print(f"  [{d['similarity']:.2f}] {a['name']} ≈ {b['name']}")
    return 0


def cmd_communities():
    """Detect entity communities."""
    comms = detect_communities()
    print(f"Found {len(comms)} communities:")
    for c in comms[:5]:
        names = [e["name"] for e in c.get("entities", [])[:6]]
        print(f"  [{c['size']} entities] {', '.join(names)}")


def cmd_cache_clear(query_type: str = None):
    """Clear LLM cache entries."""
    if query_type:
        count = llm_cache.invalidate_cache(query_type=query_type)
        print(f"Cleared {count} {query_type} cache entries")
    else:
        # Clear all by dropping and recreating collection
        db = mongo_memory._get_db()
        db.drop_collection("llm_cache")
        db.create_collection("llm_cache")
        print("Cleared all LLM cache entries")


def main():
    parser = argparse.ArgumentParser(
        prog="graphrag",
        description="Agentic GraphRAG CLI — unified memory query and management",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ask
    ask_p = sub.add_parser("ask", help="Query the unified memory")
    ask_p.add_argument("query", help="Natural language question")
    ask_p.add_argument("--top-k", type=int, default=3, help="Results per path")
    ask_p.add_argument("--hops", type=int, default=2, help="Max graph hops")

    # ingest file
    file_p = sub.add_parser("ingest", help="Ingest a file or URL")
    file_p.add_argument("type", choices=["file", "url"], help="Ingestion type")
    file_p.add_argument("path", help="File path or URL")
    file_p.add_argument("--entity", help="Tag with entity name")

    # neighbors
    nb_p = sub.add_parser("neighbors", help="Show graph neighbors of an entity")
    nb_p.add_argument("entity", help="Entity name")
    nb_p.add_argument("--hops", type=int, default=1, help="Traversal depth")
    nb_p.add_argument("--predicate", help="Filter by predicate type")

    # stats
    sub.add_parser("stats", help="Show graph statistics")

    # duplicates
    sub.add_parser("duplicates", help="Find potential duplicate entities")

    # communities
    sub.add_parser("communities", help="Detect entity communities")

    # cache
    cache_p = sub.add_parser("cache", help="Cache management")
    cache_p.add_argument("action", choices=["clear"], help="Action")
    cache_p.add_argument("--type", dest="query_type", help="Cache type to clear (e.g. SIMPLE)")

    args = parser.parse_args()

    # Initialize MongoDB
    try:
        mongo_memory.init()
    except Exception as e:
        print(f"MongoDB init error: {e}")
        return 1

    if args.command == "ask":
        cmd_ask(args.query, args.top_k, args.hops)
    elif args.command == "ingest":
        if args.type == "file":
            return cmd_ingest_file(args.path, args.entity)
        else:
            return cmd_ingest_url(args.path)
    elif args.command == "neighbors":
        return cmd_neighbors(args.entity, args.hops, args.predicate)
    elif args.command == "stats":
        cmd_stats()
    elif args.command == "duplicates":
        return cmd_duplicate()
    elif args.command == "communities":
        cmd_communities()
    elif args.command == "cache":
        cmd_cache_clear(args.query_type)

    return 0


if __name__ == "__main__":
    sys.exit(main())
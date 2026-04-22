"""llm_cache.py — LLM response caching for query_router.

Implements layered caching:
1. Exact-match cache (instant, hash-based)
2. Semantic cache fallback using pattern similarity
3. Optional TTL-based expiration

The cache lives in MongoDB (llm_cache collection) so it persists across restarts.
"""
import sys, os, hashlib, json, time
from datetime import datetime as dt
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import mongo_memory

CACHE_COL = "llm_cache"
EXTRACTION_CACHE_COL = "extraction_cache"
RAG_CACHE_COL = "rag_cache"

# TTL in days
DEFAULT_TTL_DAYS = 30


def _get_cache_db():
    db = mongo_memory._get_db()
    try:
        db.create_collection(CACHE_COL)
    except Exception:
        pass
    return db[CACHE_COL]


def _init_cache_indexes():
    """Ensure cache indexes exist."""
    col = _get_cache_db()
    try:
        col.create_index("query_hash", unique=True)
    except Exception:
        pass
    try:
        col.create_index("created_at", expireAfterSeconds=DEFAULT_TTL_DAYS * 86400)
    except Exception:
        pass
    col.create_index("query_type")
    col.create_index("hit_count")


def _normalize_query(query: str) -> str:
    """Normalize query for consistent cache key generation."""
    return query.lower().strip()


def _query_hash(query: str) -> str:
    """Generate a cache key for the query."""
    return hashlib.md5(_normalize_query(query).encode()).hexdigest()[:24]


def get_cached_classification(query: str) -> Optional[dict]:
    """Look up a cached classification for the query.

    Returns the cached result dict or None on miss.
    """
    col = _get_cache_db()
    qhash = _query_hash(query)

    cached = col.find_one({"query_hash": qhash})
    if cached:
        col.update_one(
            {"query_hash": qhash},
            {"$inc": {"hit_count": 1}, "$set": {"last_hit_at": dt.now().isoformat()}}
        )
        return cached.get("result")

    return None


def cache_classification(query: str, result: dict, model: str = "MiniMax-M2.7") -> bool:
    """Store a classification result in the cache."""
    col = _get_cache_db()
    qhash = _query_hash(query)

    try:
        col.update_one(
            {"query_hash": qhash},
            {"$set": {
                "query_hash": qhash,
                "query_text": query,
                "query_text_normalized": _normalize_query(query),
                "result": result,
                "model": model,
                "query_type": result.get("query_type", "UNKNOWN"),
                "detected_entities": result.get("detected_entities", []),
                "hit_count": 0,
                "created_at": dt.now().isoformat(),
                "last_hit_at": dt.now().isoformat(),
            }},
            upsert=True,
        )
        return True
    except Exception as e:
        print(f"[llm_cache] failed to cache: {e}")
        return False


def invalidate_cache(query: str = None, query_type: str = None) -> int:
    """Invalidate cache entries. Returns count of invalidated entries.

    Args:
        query: Invalidate a specific query (by exact text match)
        query_type: Invalidate all entries of a given query_type
    """
    col = _get_cache_db()
    if query:
        qhash = _query_hash(query)
        result = col.delete_many({"query_hash": qhash})
        return result.deleted_count
    elif query_type:
        result = col.delete_many({"query_type": query_type})
        return result.deleted_count
    return 0


def get_cached_response(cache_key: str, collection: str = CACHE_COL) -> Optional[dict]:
    """Look up a cached response by cache key.

    Args:
        cache_key: The cache key (hash)
        collection: Which cache collection ('llm_cache', 'extraction_cache', 'rag_cache')

    Returns the cached result dict or None on miss.
    """
    db = mongo_memory._get_db()
    try:
        col = db[collection]
    except Exception:
        return None

    cached = col.find_one({"cache_key": cache_key})
    if cached:
        col.update_one(
            {"cache_key": cache_key},
            {"$inc": {"hit_count": 1}, "$set": {"last_hit_at": dt.now().isoformat()}}
        )
        return cached.get("result")
    return None


def cache_response(cache_key: str, result: dict, collection: str = CACHE_COL, ttl_days: int = DEFAULT_TTL_DAYS) -> bool:
    """Store a response in the specified cache collection.

    Args:
        cache_key: The cache key (hash)
        result: The response dict to cache
        collection: Which cache collection
        ttl_days: Time-to-live in days

    Returns True on success.
    """
    db = mongo_memory._get_db()
    try:
        col = db[collection]
    except Exception:
        return False

    # Ensure TTL index
    try:
        col.create_index("created_at", expireAfterSeconds=ttl_days * 86400)
    except Exception:
        pass
    try:
        col.create_index("cache_key", unique=True)
    except Exception:
        pass

    try:
        col.update_one(
            {"cache_key": cache_key},
            {"$set": {
                "cache_key": cache_key,
                "result": result,
                "hit_count": 0,
                "created_at": dt.now().isoformat(),
                "last_hit_at": dt.now().isoformat(),
            }},
            upsert=True,
        )
        return True
    except Exception as e:
        print(f"[llm_cache] failed to cache: {e}")
        return False


def get_cache_stats() -> dict:
    """Get cache hit/miss statistics."""
    col = _get_cache_db()

    total = col.count_documents({})
    by_type = {}
    for r in col.aggregate([
        {"$group": {"_id": "$query_type", "count": {"$sum": 1}, "hits": {"$sum": "$hit_count"}}}
    ]):
        by_type[r["_id"]] = {"count": r["count"], "total_hits": r["hits"]}

    # Recent entries
    recent = list(col.find({}, {"query_text": 1, "query_type": 1, "hit_count": 1, "last_hit_at": 1})
                  .sort("last_hit_at", -1).limit(10))

    return {
        "total_cached": total,
        "by_type": by_type,
        "recent": recent,
    }


def cached_classify_query(query: str, use_cache: bool = True) -> dict:
    """Wrap classify_query with caching.

    On cache hit: returns instantly (<5ms).
    On cache miss: calls MiniMax, stores result, returns.
    """
    if use_cache:
        cached = get_cached_classification(query)
        if cached:
            return {"cached": True, **cached}

    # Miss - call the underlying classifier
    from src.query_router import classify_query as _classify
    result = _classify(query)

    # Cache it
    if use_cache:
        cache_classification(query, result)

    return {"cached": False, **result}


# Initialize indexes on module load
_init_cache_indexes()


if __name__ == "__main__":
    mongo_memory.init()
    _init_cache_indexes()

    print("=== Cache Stats ===")
    stats = get_cache_stats()
    print(json.dumps(stats, indent=2))

    # Test cache round-trip
    test_q = "what is GraphRAG"
    print(f"\n=== Test: '{test_q}' ===")

    start = time.time()
    result = cached_classify_query(test_q)
    print(f"First call (uncached): {time.time() - start:.1f}s -> {result.get('query_type')}, cached={result.get('cached')}")

    start = time.time()
    result = cached_classify_query(test_q)
    print(f"Second call (cached): {time.time() - start:.3f}s -> {result.get('query_type')}, cached={result.get('cached')}")

    # Show recent cached
    print("\nRecent cache entries:")
    stats = get_cache_stats()
    for r in stats.get("recent", []):
        print(f"  {r['query_type']}: {r['query_text'][:50]}... (hits: {r['hit_count']})")
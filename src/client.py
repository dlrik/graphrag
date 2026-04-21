"""client.py — Client for memory-core HTTP API."""
import httpx
import os

MEMORY_CORE_URL = os.getenv("MEMORY_CORE_URL", "http://localhost:8765")


def health() -> dict:
    r = httpx.get(f"{MEMORY_CORE_URL}/health", timeout=5)
    r.raise_for_status()
    return r.json()


def add_fact(entity: str, category: str, content: str, importance: int = 5,
             confidence: float = 0.9, source: str = "") -> int:
    r = httpx.post(f"{MEMORY_CORE_URL}/facts", json={
        "entity": entity, "category": category, "content": content,
        "importance": importance, "confidence": confidence, "source": source,
    }, timeout=10)
    r.raise_for_status()
    return r.json()["fact_id"]


def search_facts(query: str = None, entity: str = None, category: str = None, limit: int = 20) -> list[dict]:
    params = {}
    if query: params["query"] = query
    if entity: params["entity"] = entity
    if category: params["category"] = category
    params["limit"] = limit
    r = httpx.get(f"{MEMORY_CORE_URL}/facts", params=params, timeout=10)
    r.raise_for_status()
    return r.json()["results"]


def graph_connect(subject: str, predicate: str, object: str, weight: float = 1.0, source: str = "") -> int:
    r = httpx.post(f"{MEMORY_CORE_URL}/graph", json={
        "subject": subject, "predicate": predicate, "object": object,
        "weight": weight, "source": source,
    }, timeout=10)
    r.raise_for_status()
    return r.json()["edge_id"]


def graph_infer(subject: str, predicate: str) -> list[str]:
    r = httpx.get(f"{MEMORY_CORE_URL}/graph/infer", params={
        "subject": subject, "predicate": predicate,
    }, timeout=10)
    r.raise_for_status()
    return r.json()["objects"]


def graph_neighbors(entity: str) -> list[dict]:
    r = httpx.get(f"{MEMORY_CORE_URL}/graph/neighbors", params={"entity": entity}, timeout=10)
    r.raise_for_status()
    return r.json()["edges"]


def graph_know(who: str) -> list[dict]:
    r = httpx.get(f"{MEMORY_CORE_URL}/graph/know", params={"who": who}, timeout=10)
    r.raise_for_status()
    return r.json()["edges"]


def vec_add(content: str, entity: str = None, source: str = "") -> str:
    meta = {}
    if entity: meta["entity"] = entity
    if source: meta["source"] = source
    r = httpx.post(f"{MEMORY_CORE_URL}/vec", json={
        "content": content, "entity": entity, "source": source,
    }, timeout=30)
    r.raise_for_status()
    return r.json()["chunk_id"]


def vec_search(query: str, top_k: int = 5, entity: str = None) -> list[dict]:
    params = {"query": query, "top_k": top_k}
    if entity: params["entity"] = entity
    r = httpx.get(f"{MEMORY_CORE_URL}/vec/search", params=params, timeout=30)
    r.raise_for_status()
    return r.json()["hits"]


def vec_related(content: str, top_k: int = 5) -> list[dict]:
    r = httpx.get(f"{MEMORY_CORE_URL}/vec/related", params={"content": content, "top_k": top_k}, timeout=30)
    r.raise_for_status()
    return r.json()["hits"]


if __name__ == "__main__":
    h = health()
    print(f"memory-core: {h['facts']} facts, {h['chunks']} chunks, {h['episodes']} episodes")

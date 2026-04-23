"""graph_quality.py — Graph maintenance, community detection, and quality tools.

Provides:
- Entity re-resolution for duplicate detection on growing entity sets
- Temporal timestamps on relations for "what changed" queries
- Community detection via MongoDB aggregation to surface entity clusters
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from datetime import datetime as dt
from typing import Optional
from collections import deque

from src import mongo_memory


def add_temporal_to_relation(relation_id: str, timestamp: str = None) -> bool:
    """Add/update timestamp on a relation for temporal tracking."""
    if timestamp is None:
        timestamp = dt.now().isoformat()

    db = mongo_memory._get_db()
    result = db[mongo_memory.RELATIONS_COL].update_one(
        {"relation_id": relation_id},
        {"$set": {"updated_at": timestamp}},
    )
    return result.modified_count > 0


def get_recent_relations(limit: int = 20) -> list[dict]:
    """Get most recently created/updated relations."""
    db = mongo_memory._get_db()
    cursor = db[mongo_memory.RELATIONS_COL].find(
        {},
        sort=[("created_at", -1)],
        limit=limit,
    )
    return [mongo_memory._doc_to_dict(r) for r in cursor]


def get_entity_age(entity_id: str) -> Optional[dict]:
    """Get creation and update timestamps for an entity."""
    ent = mongo_memory._get_db()[mongo_memory.ENTITIES_COL].find_one(
        {"entity_id": entity_id}
    )
    if not ent:
        return None
    return {
        "entity_id": entity_id,
        "name": ent.get("name"),
        "created_at": ent.get("created_at"),
        "updated_at": ent.get("updated_at"),
    }


def find_duplicate_entities() -> list[dict]:
    """Find potential duplicate entities that may need resolution.

    Uses string similarity on entity names to find candidates.
    """
    from src.entity_resolver import _string_similarity

    db = mongo_memory._get_db()
    entities = list(db[mongo_memory.ENTITIES_COL].find({}))

    duplicates = []
    # Length-based blocking: only compare entities with similar name lengths
    by_len = {}
    for ent in entities:
        name = ent.get("name", "")
        if name:
            by_len.setdefault(len(name), []).append(ent)

    checked = set()
    for ent_a in entities:
        name_a = ent_a.get("name", "")
        if not name_a:
            continue
        if ent_a.get("canonical_id"):
            continue
        # Only compare with entities of similar length (±30%)
        candidates = by_len.get(len(name_a), [])
        for ent_b in candidates:
            if ent_a["entity_id"] >= ent_b["entity_id"]:
                continue
            if ent_b.get("canonical_id"):
                continue
            name_b = ent_b.get("name", "")
            if not name_b:
                continue
            # Quick length filter
            if abs(len(name_a) - len(name_b)) > max(len(name_a), len(name_b)) * 0.3:
                continue
            score = _string_similarity(name_a.lower(), name_b.lower())
            if score >= 0.6 and score < 1.0:
                duplicates.append({
                    "entity_a": {"name": name_a, "entity_id": ent_a["entity_id"]},
                    "entity_b": {"name": name_b, "entity_id": ent_b["entity_id"]},
                    "similarity": score,
                })

    return duplicates


def detect_communities(min_cluster_size: int = 2) -> list[dict]:
    """Detect entity communities using MongoDB aggregation.

    Finds clusters of entities that are densely connected.
    Uses a simple connected-component approach.
    """
    db = mongo_memory._get_db()

    # Build adjacency map
    adjacency = {}
    for rel in db[mongo_memory.RELATIONS_COL].find({}):
        subj = rel.get("subject_id")
        obj = rel.get("object_id")
        if subj and obj:
            adjacency.setdefault(subj, set()).add(obj)
            adjacency.setdefault(obj, set()).add(subj)

    # Find connected components (simple BFS)
    visited = set()
    communities = []

    def bfs(start_id):
        component = []
        queue = deque([start_id])
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            queue.extend(adjacency.get(node, []))
        return component

    for node in adjacency:
        if node not in visited:
            component = bfs(node)
            if len(component) >= min_cluster_size:
                # Get entity names for this component
                entity_ids = component
                entities = db[mongo_memory.ENTITIES_COL].find(
                    {"entity_id": {"$in": entity_ids}}
                )
                entity_info = [
                    {"entity_id": e["entity_id"], "name": e.get("name"), "type": e.get("entity_type")}
                    for e in entities
                ]

                # Get relation types within this community
                relations = db[mongo_memory.RELATIONS_COL].find({
                    "$or": [
                        {"subject_id": {"$in": entity_ids}},
                        {"object_id": {"$in": entity_ids}},
                    ]
                })

                predicate_counts = {}
                for r in relations:
                    p = r.get("predicate", "UNKNOWN")
                    predicate_counts[p] = predicate_counts.get(p, 0) + 1

                communities.append({
                    "size": len(component),
                    "entity_ids": entity_ids,
                    "entities": entity_info,
                    "predicate_counts": predicate_counts,
                })

    return communities


def re_resolve_candidates(candidates: list[dict] = None) -> dict:
    """Re-run entity resolution on potential duplicate candidates.

    Args:
        candidates: List of {entity_a, entity_b, similarity} from find_duplicate_entities

    Returns dict with merge_count, error_count
    """
    from src import entity_resolver

    if candidates is None:
        candidates = find_duplicate_entities()

    from src import mongo_memory

    merge_count = 0
    error_count = 0

    for cand in candidates:
        name_a = cand["entity_a"]["name"]
        name_b = cand["entity_b"]["name"]

        try:
            result = entity_resolver.resolve_pair(name_a, name_b)
            if result.get("same_entity"):
                canonical = result.get("canonical_name", name_a)
                # Only pass the non-canonical name as alias to avoid self-reference
                alias_to_merge = name_b if canonical == name_a else name_a
                mongo_memory.merge_entities(canonical, alias_to_merge)
                merge_count += 1
                print(f"[graph_quality] Merged {name_a} + {name_b} → {canonical}")
        except Exception as e:
            print(f"[graph_quality] Error resolving {name_a}/{name_b}: {e}")
            error_count += 1

    return {
        "merge_count": merge_count,
        "error_count": error_count,
        "candidates_checked": len(candidates),
    }


def get_graph_statistics() -> dict:
    """Get comprehensive graph statistics."""
    db = mongo_memory._get_db()

    total_entities = db[mongo_memory.ENTITIES_COL].count_documents({})
    total_relations = db[mongo_memory.RELATIONS_COL].count_documents({})
    total_documents = db[mongo_memory.DOCUMENTS_COL].count_documents({})

    # Entity type distribution
    entity_types = db[mongo_memory.ENTITIES_COL].aggregate([
        {"$group": {"_id": "$entity_type", "count": {"$sum": 1}}}
    ])
    entity_type_dist = {r["_id"]: r["count"] for r in entity_types}

    # Predicate distribution
    predicates = db[mongo_memory.RELATIONS_COL].aggregate([
        {"$group": {"_id": "$predicate", "count": {"$sum": 1}}}
    ])
    predicate_dist = {r["_id"]: r["count"] for r in predicates}

    # Entities with aliases
    aliased = db[mongo_memory.ENTITIES_COL].count_documents({"canonical_id": {"$ne": None}})
    canonical = db[mongo_memory.ENTITIES_COL].count_documents({"canonical_id": None})

    # Recent activity
    recent_rels = list(db[mongo_memory.RELATIONS_COL].find(
        {},
        sort=[("created_at", -1)],
        limit=5,
    ))
    recent_entities = list(db[mongo_memory.ENTITIES_COL].find(
        {},
        sort=[("created_at", -1)],
        limit=5,
    ))

    return {
        "totals": {
            "entities": total_entities,
            "relations": total_relations,
            "documents": total_documents,
        },
        "entity_types": entity_type_dist,
        "predicates": predicate_dist,
        "entity_resolution": {
            "canonical_entities": canonical,
            "aliased_entities": aliased,
        },
        "recent_relations": [mongo_memory._doc_to_dict(r) for r in recent_rels],
        "recent_entities": [mongo_memory._doc_to_dict(e) for e in recent_entities],
    }


if __name__ == "__main__":
    mongo_memory.init()

    print("=== Graph Statistics ===")
    stats = get_graph_statistics()
    print(json.dumps(stats, indent=2, default=str))

    print("\n=== Duplicate Candidates ===")
    dups = find_duplicate_entities()
    print(f"Found {len(dups)} potential duplicates")
    for d in dups[:5]:
        print(f"  {d['entity_a']['name']} <-> {d['entity_b']['name']} ({d['similarity']:.2f})")

    print("\n=== Communities ===")
    communities = detect_communities()
    print(f"Found {len(communities)} communities")
    for c in communities[:3]:
        print(f"  Community of {c['size']}: {[e['name'] for e in c['entities']]}")

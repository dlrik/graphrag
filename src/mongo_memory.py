"""mongo_memory.py — MongoDB-backed unified memory layer.

Collections:
- documents: raw ingested documents with full-text search
- entities: canonical entity nodes with graph metadata
- relations: entity-relation-entity triplets (graph edges)
- chunks: text chunks with embeddings stored separately (ChromaDB for vectors)

This provides the MongoDB "Unified Memory" layer from the architecture diagram.
"""
import os, sys, uuid, hashlib
from datetime import datetime as dt
from typing import Optional

from pymongo import MongoClient, TEXT, ASCENDING
from pymongo.errors import DuplicateKeyError, OperationFailure, CollectionInvalid

# MongoDB connection
MONGO_URI = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017")
DB_NAME = "graphrag"

# Database and collection names
DOCUMENTS_COL = "documents"
ENTITIES_COL = "entities"
RELATIONS_COL = "relations"

# Global client (lazily initialized)
_client = None
_db = None


def _get_client():
    global _client
    if _client is None:
        _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    return _client


def _get_db():
    global _db
    if _db is None:
        _db = _get_client()[DB_NAME]
    return _db


def init():
    """Initialize MongoDB collections and indexes."""
    db = _get_db()

    # documents collection: raw source documents
    try:
        db.create_collection(DOCUMENTS_COL)
    except CollectionInvalid:
        pass
    docs = db[DOCUMENTS_COL]
    docs.create_index([("content", TEXT)], name="text_index", default_language="english")
    docs.create_index("doc_id", unique=True)
    docs.create_index("source")
    docs.create_index("created_at")

    # entities collection: canonical entity nodes
    try:
        db.create_collection(ENTITIES_COL)
    except CollectionInvalid:
        pass
    ents = db[ENTITIES_COL]
    ents.create_index("entity_id", unique=True)
    ents.create_index("name_lower", unique=True)  # case-insensitive dedup
    ents.create_index("canonical_id")  # for entity resolution linking
    ents.create_index("entity_type")
    ents.create_index([("name", TEXT)])

    # relations collection: graph edges
    try:
        db.create_collection(RELATIONS_COL)
    except CollectionInvalid:
        pass
    rels = db[RELATIONS_COL]
    rels.create_index("relation_id", unique=True)
    rels.create_index("subject_id")
    rels.create_index("object_id")
    rels.create_index("predicate")
    # Compound index for graph traversal
    rels.create_index([("subject_id", ASCENDING), ("predicate", ASCENDING)])
    rels.create_index([("object_id", ASCENDING), ("predicate", ASCENDING)])

    print("[mongo_memory] Collections initialized")
    return db


# ---------------------------------------------------------------------------
# Document storage
# ---------------------------------------------------------------------------

def store_document(content: str, source: str, metadata: dict = None) -> str:
    """Store a raw document. Returns doc_id."""
    db = _get_db()
    doc_id = hashlib.md5((content + source).encode()).hexdigest()[:16]

    doc = {
        "doc_id": doc_id,
        "content": content,
        "source": source,
        "metadata": metadata or {},
        "created_at": ts(),
    }

    try:
        db[DOCUMENTS_COL].update_one(
            {"doc_id": doc_id},
            {"$set": doc},
            upsert=True,
        )
    except DuplicateKeyError:
        pass

    return doc_id


def get_document(doc_id: str) -> Optional[dict]:
    """Retrieve a document by ID."""
    return db[DOCUMENTS_COL].find_one({"doc_id": doc_id})


def search_documents(keyword: str, limit: int = 20) -> list[dict]:
    """Full-text search on documents."""
    db = _get_db()
    cursor = db[DOCUMENTS_COL].find(
        {"$text": {"$search": keyword}},
        {"score": {"$meta": "textScore"}},
    ).sort([("score", {"$meta": "textScore"})]).limit(limit)

    results = []
    for doc in cursor:
        results.append({
            "doc_id": doc["doc_id"],
            "content": doc["content"],
            "source": doc["source"],
            "score": doc.get("score"),
        })
    return results


# ---------------------------------------------------------------------------
# Entity storage
# ---------------------------------------------------------------------------

def store_entity(name: str, entity_type: str, properties: dict = None,
                 canonical_id: str = None, source: str = "") -> str:
    """Store an entity node. Returns entity_id.

    Args:
        name: Entity name (e.g., "Abi", "Abi Aryan")
        entity_type: Type from entity classification (PERSON, ORGANIZATION, etc.)
        properties: Optional dict of additional properties
        canonical_id: ID of the canonical form (for aliases/duplicates)
        source: Source of this entity (url, file, etc.)
    """
    db = _get_db()
    entity_id = hashlib.md5(name.lower().encode()).hexdigest()[:16]

    entity = {
        "entity_id": entity_id,
        "name": name,
        "name_lower": name.lower().strip(),
        "entity_type": entity_type.upper(),
        "properties": properties or {},
        "canonical_id": canonical_id,  # link to master entity if this is an alias
        "source": source,
        "created_at": ts(),
        "updated_at": ts(),
    }

    try:
        db[ENTITIES_COL].update_one(
            {"name_lower": name.lower().strip()},
            {"$set": entity},
            upsert=True,
        )
    except DuplicateKeyError:
        # Entity already exists, update it
        db[ENTITIES_COL].update_one(
            {"name_lower": name.lower().strip()},
            {"$set": {**entity, "updated_at": ts()}},
        )

    return entity_id


def get_entity(name: str = None, entity_id: str = None) -> Optional[dict]:
    """Get an entity by name or ID."""
    db = _get_db()
    if entity_id:
        return db[ENTITIES_COL].find_one({"entity_id": entity_id})
    if name:
        return db[ENTITIES_COL].find_one({"name_lower": name.lower().strip()})
    return None


def get_canonical_entity(name: str) -> dict:
    """Resolve an entity to its canonical form.

    Returns the canonical entity (following alias links), or the entity itself
    if no canonical exists.
    """
    db = _get_db()
    ent = get_entity(name)
    if not ent:
        return None

    # Follow canonical links
    visited = set()
    while ent.get("canonical_id") and ent["entity_id"] not in visited:
        visited.add(ent["entity_id"])
        ent = db[ENTITIES_COL].find_one({"entity_id": ent["canonical_id"]})
        if not ent:
            break

    return ent


def resolve_entities(names: list[str]) -> dict[str, str]:
    """Resolve multiple names to their canonical entity_ids.

    Returns a dict mapping each input name to its canonical entity_id.
    """
    db = _get_db()
    results = {}
    for name in names:
        name_lower = name.lower().strip()
        ent = db[ENTITIES_COL].find_one({"name_lower": name_lower})
        if ent:
            # Follow canonical chain
            while ent.get("canonical_id"):
                canonical = db[ENTITIES_COL].find_one({"entity_id": ent["canonical_id"]})
                if canonical:
                    ent = canonical
                else:
                    break
            results[name] = ent["entity_id"]
        else:
            results[name] = None
    return results


def merge_entities(primary_name: str, *alias_names: str) -> str:
    """Merge multiple entities into a single canonical entity.

    Args:
        primary_name: The canonical name (will be kept as-is)
        alias_names: Other names that will point to the primary

    Returns:
        The canonical entity_id
    """
    db = _get_db()

    # Ensure primary exists
    primary_id = store_entity(primary_name, "CONCEPT", source="entity_resolution")

    # Update all aliases to point to primary
    for alias in alias_names:
        alias_lower = alias.lower().strip()
        existing = db[ENTITIES_COL].find_one({"name_lower": alias_lower})
        if existing:
            # Update to point to primary
            db[ENTITIES_COL].update_one(
                {"name_lower": alias_lower},
                {"$set": {
                    "canonical_id": primary_id,
                    "updated_at": ts(),
                }},
            )
        else:
            # Create alias entry
            entity_id = hashlib.md5(alias_lower.encode()).hexdigest()[:16]
            db[ENTITIES_COL].update_one(
                {"entity_id": entity_id},
                {"$set": {
                    "entity_id": entity_id,
                    "name": alias,
                    "name_lower": alias_lower,
                    "entity_type": "ALIAS",
                    "canonical_id": primary_id,
                    "created_at": ts(),
                    "updated_at": ts(),
                }},
                upsert=True,
            )

    return primary_id


def entity_search(query: str, limit: int = 10) -> list[dict]:
    """Search entities by text match."""
    db = _get_db()
    cursor = db[ENTITIES_COL].find(
        {"$text": {"$search": query}},
        {"score": {"$meta": "textScore"}},
    ).sort([("score", {"$meta": "textScore"})]).limit(limit)

    return [_doc_to_dict(doc) for doc in cursor]


def _doc_to_dict(doc) -> dict:
    """Remove MongoDB _id from a document for JSON serialization."""
    if doc and "_id" in doc:
        doc = dict(doc)
        del doc["_id"]
    return doc


# ---------------------------------------------------------------------------
# Relation storage
# ---------------------------------------------------------------------------

def store_relation(subject: str, predicate: str, object: str,
                   weight: float = 1.0, source: str = "",
                   subject_id: str = None, object_id: str = None) -> str:
    """Store a relation triplet.

    Args:
        subject: Subject entity name
        predicate: Relation type (MENTIONS, CONNECTED_TO, HAS, etc.)
        object: Object entity name
        weight: Edge weight (0.0 to 1.0)
        source: Source of this relation
        subject_id: Pre-resolved subject entity_id (optional)
        object_id: Pre-resolved object entity_id (optional)
    """
    db = _get_db()

    # Resolve entity IDs
    if subject_id is None:
        subject_id = hashlib.md5(subject.lower().strip().encode()).hexdigest()[:16]
    if object_id is None:
        object_id = hashlib.md5(object.lower().strip().encode()).hexdigest()[:16]

    relation_id = hashlib.md5(
        (subject.lower() + predicate + object.lower()).encode()
    ).hexdigest()[:16]

    relation = {
        "relation_id": relation_id,
        "subject": subject,
        "subject_id": subject_id,
        "predicate": predicate.upper(),
        "object": object,
        "object_id": object_id,
        "weight": weight,
        "source": source,
        "created_at": ts(),
    }

    try:
        db[RELATIONS_COL].update_one(
            {"relation_id": relation_id},
            {"$set": relation},
            upsert=True,
        )
    except DuplicateKeyError:
        pass

    return relation_id


def get_relations(subject_id: str = None, object_id: str = None,
                  predicate: str = None, limit: int = 100) -> list[dict]:
    """Get relations matching criteria."""
    db = _get_db()
    query = {}
    if subject_id:
        query["subject_id"] = subject_id
    if object_id:
        query["object_id"] = object_id
    if predicate:
        query["predicate"] = predicate.upper()

    cursor = db[RELATIONS_COL].find(query).limit(limit)
    return [_doc_to_dict(r) for r in cursor]


def traverse_hops(start_id: str, hops: int = 2, predicate: str = None) -> list[dict]:
    """Multi-hop graph traversal.

    Args:
        start_id: Starting entity_id
        hops: Number of hops (1-3)
        predicate: Optional predicate filter

    Returns:
        List of {entity, relation, depth} dicts
    """
    db = _get_db()
    results = []
    visited = {start_id}
    current_frontier = {start_id}

    for depth in range(1, hops + 1):
        next_frontier = set()
        query = {"subject_id": {"$in": list(current_frontier)}}
        if predicate:
            query["predicate"] = predicate.upper()

        for rel in db[RELATIONS_COL].find(query):
            if rel["object_id"] not in visited:
                results.append({
                    "entity": rel["object"],
                    "entity_id": rel["object_id"],
                    "relation": rel["predicate"],
                    "depth": depth,
                    "source_entity": rel["subject"],
                })
                visited.add(rel["object_id"])
                next_frontier.add(rel["object_id"])

            # Also follow reverse relations (incoming edges)
            if rel["subject_id"] not in visited:
                results.append({
                    "entity": rel["subject"],
                    "entity_id": rel["subject_id"],
                    "relation": rel["predicate"],
                    "depth": depth,
                    "source_entity": rel["object"],
                })
                visited.add(rel["subject_id"])
                next_frontier.add(rel["subject_id"])

        current_frontier = next_frontier
        if not current_frontier:
            break

    return results


def graph_neighbors(entity_id: str, max_depth: int = 1) -> list[dict]:
    """Get all entities connected to a given entity within max_depth hops."""
    db = _get_db()
    results = []
    visited = {entity_id}
    current_frontier = {entity_id}

    for _ in range(max_depth):
        next_frontier = set()
        for rel in db[RELATIONS_COL].find({
            "$or": [
                {"subject_id": {"$in": list(current_frontier)}},
                {"object_id": {"$in": list(current_frontier)}},
            ]
        }):
            results.append(_doc_to_dict(rel))
            if rel["subject_id"] not in visited:
                visited.add(rel["subject_id"])
                next_frontier.add(rel["subject_id"])
            if rel["object_id"] not in visited:
                visited.add(rel["object_id"])
                next_frontier.add(rel["object_id"])

        current_frontier = next_frontier
        if not current_frontier:
            break

    return results


def ts():
    return dt.now().strftime("%Y-%m-%dT%H:%M:%S")


if __name__ == "__main__":
    init()
    # Quick test
    doc_id = store_document("Example content about Doug and his projects.", source="test")
    print(f"stored doc: {doc_id}")

    e1 = store_entity("Doug", "PERSON", source="test")
    e2 = store_entity("Claude Code", "TECHNOLOGY", source="test")
    print(f"entities: {e1}, {e2}")

    r1 = store_relation("Doug", "USES", "Claude Code", source="test")
    print(f"relation: {r1}")

    neighbors = graph_neighbors(e1)
    print(f"Doug neighbors: {neighbors}")

"""ingestion.py — Unified ingestion pipeline for URLs, files, and conversations.
ETL: Extract → Clean → Chunk → Graph Extract → Embed → Store to memory-core.
"""
import os, re, hashlib, uuid
from typing import Optional

from . import graph_extractor
from . import client

# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_from_url(url: str) -> str:
    """Fetch and extract text from a URL via Jina Reader (lightweight) or Firecrawl."""
    import httpx

    # Try Jina Reader first (simpler)
    try:
        r = httpx.get(f"https://r.jina.ai/{url}", timeout=20, headers={
            "Accept": "text/plain",
        })
        if r.status_code == 200 and len(r.text) > 200:
            return r.text
    except Exception as e:
        print(f"[ingest] Jina fetch failed for {url}: {e}")

    # Fall back to Firecrawl if available
    firecrawl_key_path = os.path.expanduser("~/.openclaw/secrets/firecrawl.key")
    if os.path.exists(firecrawl_key_path):
        try:
            key = open(firecrawl_key_path).read().strip()
            r = httpx.post(
                "https://api.firecrawl.dev/v0/scrape",
                headers={"Authorization": f"Bearer {key}"},
                json={"url": url, "formats": ["markdown"]},
                timeout=30,
            )
            if r.status_code == 200:
                data = r.json()
                return data.get("data", {}).get("markdown", "") or data.get("content", "")
        except Exception as e:
            print(f"[ingest] Firecrawl failed for {url}: {e}")

    return ""


def extract_from_file(path: str) -> str:
    """Extract text from a local file."""
    if not os.path.exists(path):
        return ""

    ext = os.path.splitext(path)[1].lower()

    if ext in (".txt", ".md", ".csv", ".json", ".py", ".js", ".ts", ".html", ".xml"):
        with open(path) as f:
            return f.read()

    if ext == ".pdf":
        # Basic PDF text extraction
        try:
            import subprocess
            result = subprocess.run(["pdftotext", path, "-"], capture_output=True, text=True, timeout=30)
            return result.stdout if result.returncode == 0 else ""
        except Exception:
            return ""

    # Unknown extension - try as plain text anyway
    try:
        with open(path) as f:
            return f.read()
    except Exception:
        return ""


def extract_from_conversation(messages: list[dict]) -> str:
    """Convert conversation message list to readable text.

    Args:
        messages: list of {"role": "user|assistant", "content": "..."} dicts
    """
    lines = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if content:
            lines.append(f"[{role.upper()}] {content}")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Text cleaning and chunking
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Normalize whitespace, remove boilerplate."""
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove leading/trailing whitespace on each line
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    # Strip leading/trailing whitespace
    return text.strip()


CHUNK_SIZE = 1000  # characters
CHUNK_OVERLAP = 100


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    if len(text) <= size:
        return [text] if text.strip() else []

    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end]
        # Don't split mid-sentence if possible
        if end < len(text):
            # Try to find a sentence boundary near the end
            period = chunk.rfind(". ")
            newline = chunk.rfind("\n")
            split = max(period, newline)
            if split > start + size // 2:
                chunk = chunk[:split + 1]
                end = start + split + 1

        chunks.append(chunk.strip())
        start = end - overlap
        if start >= len(text) - overlap:
            break

    return [c for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# Knowledge Graph storage
# ---------------------------------------------------------------------------

def _entity_id(entity_name: str) -> str:
    """Generate a stable ID for an entity name."""
    return hashlib.md5(entity_name.lower().encode()).hexdigest()[:16]


def _store_entities(entities: list[dict], source: str, chunk_id: str) -> list[str]:
    """Store extracted entities as facts and graph edges."""
    stored = []
    entity_ids = {}

    for ent in entities:
        name = ent.get("name", "").strip()
        ent_type = ent.get("type", "Concept").upper()

        if not name:
            continue

        eid = _entity_id(name)
        entity_ids[name] = eid

        # Store as a fact in memory-core
        try:
            fid = client.add_fact(
                entity=name,
                category="entity",
                content=f"[{ent_type}] {name}",
                importance=5,
                source=f"graphrag:{source}",
            )
            stored.append(f"fact:{fid}")
        except Exception as e:
            print(f"[ingest] add_fact error for {name}: {e}")

        # Also store graph node via entity_graph
        try:
            client.graph_connect(
                subject=name,
                predicate="IS_A",
                object=ent_type,
                weight=0.5,
                source=f"graphrag:{source}",
            )
        except Exception as e:
            print(f"[ingest] graph_connect error for {name}: {e}")

    return stored, entity_ids


def _store_relations(relations: list[dict], entity_ids: dict, source: str, chunk_id: str):
    """Store extracted relationships as graph edges."""
    stored = []

    for rel in relations:
        subject = rel.get("subject", "").strip()
        predicate = rel.get("predicate", "").strip().upper()
        obj = rel.get("object", "").strip()

        if not subject or not predicate or not obj:
            continue

        # Normalize predicate
        if not predicate:
            predicate = "RELATED_TO"

        try:
            eid = client.graph_connect(
                subject=subject,
                predicate=predicate,
                object=obj,
                weight=1.0,
                source=f"graphrag:{source}",
            )
            stored.append(f"edge:{eid}")
        except Exception as e:
            print(f"[ingest] graph_connect error: {subject} --{predicate}--> {obj}: {e}")

    return stored


# ---------------------------------------------------------------------------
# Main ingestion pipeline
# ---------------------------------------------------------------------------

def ingest_url(url: str) -> dict:
    """Ingest a URL: fetch → clean → chunk → extract → store."""
    result = {
        "url": url,
        "chunks_processed": 0,
        "entities_stored": 0,
        "relations_stored": 0,
        "errors": [],
    }

    text = extract_from_url(url)
    if not text:
        result["errors"].append("Failed to extract text from URL")
        return result

    return _ingest_text(text, source=f"url:{url}", entity=None)


def ingest_file(path: str, entity: Optional[str] = None) -> dict:
    """Ingest a local file: read → clean → chunk → extract → store."""
    result = {
        "file": path,
        "chunks_processed": 0,
        "entities_stored": 0,
        "relations_stored": 0,
        "errors": [],
    }

    text = extract_from_file(path)
    if not text:
        result["errors"].append("Failed to extract text from file")
        return result

    return _ingest_text(text, source=f"file:{path}", entity=entity)


def ingest_conversation(messages: list[dict], session_id: Optional[str] = None) -> dict:
    """Ingest a conversation: convert → clean → chunk → extract → store."""
    result = {
        "session_id": session_id,
        "chunks_processed": 0,
        "entities_stored": 0,
        "relations_stored": 0,
        "errors": [],
    }

    text = extract_from_conversation(messages)
    if not text:
        result["errors"].append("No text extracted from conversation")
        return result

    return _ingest_text(text, source=f"conversation:{session_id or 'unknown'}", entity=None)


def _ingest_text(text: str, source: str, entity: Optional[str]) -> dict:
    """Core ingestion logic: clean → chunk → graph extract → store."""
    result = {
        "source": source,
        "chunks_processed": 0,
        "entities_stored": 0,
        "relations_stored": 0,
        "errors": [],
    }

    # Clean
    text = clean_text(text)
    if not text:
        result["errors"].append("Text empty after cleaning")
        return result

    # Chunk
    chunks = chunk_text(text)
    if not chunks:
        result["errors"].append("No chunks generated")
        return result

    result["chunks_processed"] = len(chunks)

    # Process each chunk
    all_entity_ids = {}

    for i, chunk in enumerate(chunks):
        chunk_id = f"{hashlib.md5(source.encode()).hexdigest()[:8]}_{i}"

        # Store vector (embedding via Voyage/memory-core)
        try:
            client.vec_add(chunk, entity=entity, source=source)
        except Exception as e:
            result["errors"].append(f"vec_add chunk {i}: {e}")

        # Graph extraction via LLM
        extraction = graph_extractor.extract(chunk)
        entities = extraction.get("entities", [])
        relations = extraction.get("relations", [])

        # Deduplicate entities within this chunk
        seen_names = set()
        unique_entities = []
        for e in entities:
            name = e.get("name", "").strip().lower()
            if name and name not in seen_names:
                seen_names.add(name)
                unique_entities.append(e)

        # Store entities
        stored_entities, entity_ids = _store_entities(unique_entities, source, chunk_id)
        all_entity_ids.update(entity_ids)
        result["entities_stored"] += len(stored_entities)

        # Store relations (using original names, not deduped)
        stored_relations = _store_relations(relations, entity_ids, source, chunk_id)
        result["relations_stored"] += len(stored_relations)

    result["total_entities"] = len(all_entity_ids)
    return result


if __name__ == "__main__":
    # Quick test
    r = ingest_url("https://example.com")
    print(f"ingest result: {r}")

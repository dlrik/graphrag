"""graph_extractor.py — LLM-based entity and relationship extraction.

Extracts entities and relations from text chunks using an LLM.
Uses MiniMax-M2.7 via the Anthropic Messages API.

Results are cached in MongoDB (extraction_cache collection, 60-day TTL)
so repeated or similar text is answered instantly without an LLM call.
"""
import json, os, re, hashlib
from typing import Optional
from datetime import datetime as dt

import httpx

OPENCLAW_CONFIG_PATH = os.path.expanduser("~/.openclaw/openclaw.json")

# MiniMax API (Anthropic-compatible Messages endpoint)
MINIMAX_API_BASE = "https://api.minimax.io/anthropic/v1/messages"
MINIMAX_MODEL = "MiniMax-M2.7"

# Extraction cache settings
EXTRACTION_CACHE_COL = "extraction_cache"
EXTRACTION_CACHE_TTL_DAYS = 60


def _get_cache_db():
    from src import mongo_memory
    db = mongo_memory._get_db()
    try:
        db.create_collection(EXTRACTION_CACHE_COL)
    except Exception:
        pass
    return db[EXTRACTION_CACHE_COL]


def _init_extraction_cache():
    col = _get_cache_db()
    try:
        col.create_index("text_hash", unique=True)
    except Exception:
        pass
    try:
        col.create_index("created_at", expireAfterSeconds=EXTRACTION_CACHE_TTL_DAYS * 86400)
    except Exception:
        pass
    col.create_index("hit_count")


def _text_hash(text: str) -> str:
    """Generate a content-stable hash for the input text."""
    normalized = text.encode("utf-8", errors="replace").decode("utf-8")
    return hashlib.md5(normalized.encode()).hexdigest()[:24]


def _get_cached_extraction(text_hash: str) -> Optional[dict]:
    col = _get_cache_db()
    cached = col.find_one({"text_hash": text_hash})
    if cached:
        col.update_one(
            {"text_hash": text_hash},
            {"$inc": {"hit_count": 1}, "$set": {"last_hit_at": dt.now().isoformat()}}
        )
        result = cached.get("result", {})
        result["_cached"] = True
        result["_cache_hit"] = True
        return result
    return None


def _cache_extraction(text_hash: str, text: str, result: dict, model: str = MINIMAX_MODEL):
    col = _get_cache_db()
    try:
        col.update_one(
            {"text_hash": text_hash},
            {"$set": {
                "text_hash": text_hash,
                "text_preview": text[:200],
                "result": result,
                "model": model,
                "entity_count": len(result.get("entities", [])),
                "relation_count": len(result.get("relations", [])),
                "hit_count": 0,
                "created_at": dt.now().isoformat(),
                "last_hit_at": dt.now().isoformat(),
            }},
            upsert=True,
        )
    except Exception as e:
        print(f"[graph_extractor] cache write failed: {e}")


def _load_config():
    try:
        with open(OPENCLAW_CONFIG_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def _get_minimax_key() -> str:
    """Get MiniMax API key from CCR config or openclaw config."""
    import json, os

    # Try CCR config
    ccr_config = os.path.expanduser("~/.claude-code-router/config.json")
    if os.path.exists(ccr_config):
        try:
            with open(ccr_config) as f:
                config = json.load(f)
            for provider in config.get("Providers", []):
                if provider.get("name") == "minimax":
                    return provider.get("api_key", "")
        except Exception:
            pass

    # Try openclaw config
    openclaw = os.path.expanduser("~/.openclaw/openclaw.json")
    if os.path.exists(openclaw):
        try:
            with open(openclaw) as f:
                config = json.load(f)
            models = config.get("models", {})
            providers = models.get("providers", {})
            for name, prov in providers.items():
                if name == "minimax":
                    return prov.get("apiKey", "")
        except Exception:
            pass

    return ""


EXTRACT_MARKER = "__TEXT_INPUT__"

# Ontology: allowed entity and relation types for schema-driven extraction
# Grok best practice: enforce a predefined ontology for production consistency
ENTITY_EXTRACTION_PROMPT = """You are a knowledge graph extraction engine. Given a text passage, extract structured entities and relationships following this ontology.

## Entity Types (use exactly these UPPER_SNAKE_CASE values)
- PERSON: Individual people (by name)
- ORGANIZATION: Companies, teams, projects, products
- LOCATION: Places, addresses, geographic features
- CONCEPT: Ideas, theories, methodologies, abstract concepts
- TASK: Actionable items, goals, deliverables
- DOCUMENT: Files, articles, reports, records
- TECHNOLOGY: Software, libraries, frameworks, tools
- EVENT: Conferences, releases, incidents, meetings

## Relationship Predicates (use exactly these UPPERCASE values)
- USES: Entity employs or depends on another (software uses library, person uses tool)
- HAS: Entity possesses or includes another (project has task, org has team)
- CONNECTED_TO: Related entities with no specific direction
- MENTIONS: Entity references or discusses another
- CREATED_BY: Entity was authored or built by another
- PART_OF: Entity is subordinate to another (team part_of org, task part_of project)
- FOLLOWS: Entity comes after or is successor to another
- LOCATED_IN: Entity resides or exists at another location

## Output Schema
Return ONLY a JSON object (no explanations, no markdown fences):
{"entities": [{"name": "...", "type": "PERSON|ORGANIZATION|..."}], "relations": [{"subject": "...", "predicate": "USES|HAS|...", "object": "..."}]}

## Rules
1. Extract only entities explicitly mentioned in the text
2. Use the most complete/formal name for entity.name (e.g., "Claude Code" not "Claude", "MongoDB" not "Mongo")
3. Relations should reflect direct, literal statements — not inferred causation
4. One entity per unique name — do not list the same entity twice
5. Be conservative: when in doubt, skip the entity or relation
6. Include the predicate LOCATED_IN for geographic relationships

Text:
""" + EXTRACT_MARKER + """

Respond ONLY with the JSON object:"""


def extract(text: str, model: Optional[str] = None, use_cache: bool = True) -> dict:
    """Extract entities and relationships from text using MiniMax.

    Args:
        text: Input text chunk (typically 500-2000 words)
        model: Ignored (MiniMax is used by default)
        use_cache: Whether to use / write the extraction cache (default True)

    Returns:
        dict with "entities" and "relations" keys
    """
    if use_cache:
        txt_hash = _text_hash(text)
        cached = _get_cached_extraction(txt_hash)
        if cached:
            return cached

    api_key = _get_minimax_key()
    if not api_key:
        return {"entities": [], "relations": [], "error": "No MiniMax API key found"}

    prompt = ENTITY_EXTRACTION_PROMPT.replace(EXTRACT_MARKER, text[:4000])
    result = None

    try:
        with httpx.Client(timeout=60) as client:
            response = client.post(
                MINIMAX_API_BASE,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "x-api-key": api_key,
                },
                json={
                    "model": MINIMAX_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 2048,
                },
            )
            response.raise_for_status()
            data = response.json()

        # Extract text content from MiniMax content blocks
        content = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                content = block.get("text", "")
                break

        if not content:
            raise ValueError("No text content in MiniMax response")

        # Try direct parse first
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try stripping markdown fences
        if result is None:
            stripped = re.sub(r"^```json\s*", "", content)
            stripped = re.sub(r"^```\s*", "", stripped)
            stripped = re.sub(r"\s*```$", "", stripped).strip()
            try:
                result = json.loads(stripped)
            except json.JSONDecodeError:
                pass

        # Try brace extraction last
        if result is None:
            first = content.find("{")
            last = content.rfind("}")
            if first != -1 and last > first:
                try:
                    result = json.loads(content[first:last + 1])
                except json.JSONDecodeError:
                    pass

    except Exception as e:
        print(f"[graph_extractor] extraction error: {e}")

    if result is None:
        result = {"entities": [], "relations": [], "error": f"Could not parse JSON from: {content[:200] if content else 'empty'}"}

    if use_cache:
        txt_hash = _text_hash(text)
        _cache_extraction(txt_hash, text, result)

    return result


def extract_batch(texts: list[str], model: Optional[str] = None, use_cache: bool = True) -> list[dict]:
    """Extract from multiple texts in sequence. Returns list of extraction results."""
    return [extract(t, model=model, use_cache=use_cache) for t in texts]


if __name__ == "__main__":
    from src import mongo_memory
    mongo_memory.init()
    _init_extraction_cache()

    test_text = """
    Doug built a memory system using FastAPI and ChromaDB. The system uses Voyage AI
    for embeddings and stores knowledge graphs in SQLite. It integrates with Claude Code
    via an MCP server. Last week he added entity resolution to handle duplicates.
    Sarah and John are working at Acme Corp. She lives in Seattle.
    """
    result = extract(test_text)
    print(json.dumps(result, indent=2))

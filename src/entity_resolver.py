"""entity_resolver.py — Entity resolution and normalization.

Merges duplicate entities (e.g., "Abi" = "Abi Aryan") into canonical forms.
Uses LLM to determine if two entities refer to the same real-world entity.
"""
import json
import re
import sys
import os
import hashlib
from datetime import datetime as dt
from typing import Optional

# Add parent to path for absolute imports within package
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Import via absolute module path
from src import mongo_memory
from src import llm_cache

ENTITY_RESOLUTION_CACHE_COL = "entity_resolution_cache"
ENTITY_RESOLUTION_CACHE_TTL = 30  # days


ENTITY_RESOLUTION_PROMPT = """You are an entity resolution engine. Given two entity names, determine if they refer to the SAME real-world entity.

Respond with ONLY a JSON object:
:START_JSON:
  "same_entity": true/false,
  "canonical_name": "the canonical form of this entity",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
:END_JSON:

Rules:
- same_entity=true ONLY if they are clearly the same person/entity
- Names with minor variations (Abi vs Abi Aryan, Dr. Smith vs Smith) → same_entity=true
- Completely different names → same_entity=false
- Be conservative with high confidence (0.9+) requirements
- canonical_name should be the most complete/formal version

Entity A: __NAME_A__
Entity B: __NAME_B__

JSON:"""


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


def _get_er_cache_db():
    """Get the entity resolution cache collection."""
    db = mongo_memory._get_db()
    try:
        db.create_collection(ENTITY_RESOLUTION_CACHE_COL)
    except Exception:
        pass
    col = db[ENTITY_RESOLUTION_CACHE_COL]
    try:
        col.create_index("cache_key", unique=True)
    except Exception:
        pass
    try:
        col.create_index("created_at", expireAfterSeconds=ENTITY_RESOLUTION_CACHE_TTL * 86400)
    except Exception:
        pass
    return col


def _er_cache_key(name_a: str, name_b: str) -> str:
    """Generate a cache key for an entity resolution pair (order-independent)."""
    # Sort alphabetically so 'Abi' vs 'Abi Aryan' and 'Abi Aryan' vs 'Abi' get same key
    a, b = sorted([name_a.lower().strip(), name_b.lower().strip()])
    return hashlib.md5(f"er:{a}:{b}".encode()).hexdigest()[:24]


def _cache_er_result(cache_key: str, name_a: str, name_b: str, result: dict):
    """Cache an entity resolution result."""
    col = _get_er_cache_db()
    try:
        col.update_one(
            {"cache_key": cache_key},
            {"$set": {
                "cache_key": cache_key,
                "name_a": name_a,
                "name_b": name_b,
                "result": result,
                "created_at": dt.now().isoformat(),
            }},
            upsert=True,
        )
    except Exception as e:
        print(f"[entity_resolver] cache write failed: {e}")


def resolve_pair(name_a: str, name_b: str, use_cache: bool = True) -> dict:
    """Use MiniMax LLM to determine if two entity names refer to the same entity.

    Returns dict with same_entity, canonical_name, confidence, reasoning.
    """
    import httpx

    # Check cache first
    if use_cache:
        cache_key = _er_cache_key(name_a, name_b)
        col = _get_er_cache_db()
        cached = col.find_one({"cache_key": cache_key})
        if cached:
            result = cached.get("result", {})
            result["_cached"] = True
            return result

    api_key = _get_minimax_key()
    if not api_key:
        return {
            "same_entity": False,
            "canonical_name": name_a,
            "confidence": 0.0,
            "reasoning": "No MiniMax API key found",
        }

    prompt = ENTITY_RESOLUTION_PROMPT.replace("__NAME_A__", name_a).replace("__NAME_B__", name_b)

    try:
        with httpx.Client(timeout=30) as client:
            response = client.post(
                "https://api.minimax.io/anthropic/v1/messages",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "x-api-key": api_key,
                },
                json={
                    "model": "MiniMax-M2.7",
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt + "\n\nRespond ONLY with the JSON object, no explanations or extra text.",
                        }
                    ],
                    "max_tokens": 512,
                },
            )
            response.raise_for_status()
            data = response.json()

        # Extract text content - MiniMax returns content blocks, find the text type
        content = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                content = block.get("text", "")
                break
            elif block.get("thinking"):
                # Strip thinking blocks - they pollute JSON parsing
                content = ""

        if not content:
            result = {"same_entity": False, "canonical_name": name_a, "confidence": 0.0, "reasoning": "No text content in MiniMax response"}
            if use_cache:
                _cache_er_result(cache_key, name_a, name_b, result)
            return result

        # Parse JSON - try direct parse first, then extract JSON from text if needed
        import re as re_module

        result = None
        # First try direct parse
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON object from the text
        if result is None:
            first_brace = content.find("{")
            last_brace = content.rfind("}")
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                json_str = content[first_brace:last_brace + 1]
                try:
                    result = json.loads(json_str)
                except json.JSONDecodeError:
                    pass

        # Clean markdown code fences and try again
        if result is None:
            cleaned = re_module.sub(r"^```json\s*", "", content)
            cleaned = re_module.sub(r"^```\s*", "", cleaned)
            cleaned = re_module.sub(r"\s*```$", "", cleaned).strip()
            try:
                result = json.loads(cleaned)
            except json.JSONDecodeError as e:
                result = {"same_entity": False, "canonical_name": name_a, "confidence": 0.0, "reasoning": f"JSON parse error: {e}, content: {content[:200]}"}

        if use_cache:
            _cache_er_result(cache_key, name_a, name_b, result)
        return result

    except Exception as e:
        result = {"same_entity": False, "canonical_name": name_a, "confidence": 0.0, "reasoning": f"Error: {str(e)}"}
        if use_cache:
            try:
                _cache_er_result(cache_key, name_a, name_b, result)
            except Exception:
                pass
        return result


def resolve_pair_batch(pairs: list[tuple[str, str]]) -> list[dict]:
    """Resolve multiple entity pairs in a single LLM call.

    Takes a list of (name_a, name_b) tuples and returns a list of result dicts
    in the same order. Skips pairs already in cache.

    Returns list of dicts with same keys as resolve_pair: same_entity, canonical_name, etc.
    """
    import httpx

    if not pairs:
        return []

    # Filter out already-cached pairs
    uncached_pairs = []
    cached_results = []
    for name_a, name_b in pairs:
        cache_key = _er_cache_key(name_a, name_b)
        col = _get_er_cache_db()
        cached = col.find_one({"cache_key": cache_key})
        if cached:
            result = dict(cached.get("result", {}))
            result["_cached"] = True
            cached_results.append((name_a, name_b, result))
        else:
            uncached_pairs.append((name_a, name_b))

    if not uncached_pairs:
        # All were cached — return in original order
        results_by_pair = {(_sort_pair(a, b)): r for a, b, r in cached_results}
        return [results_by_pair.get(_sort_pair(a, b), {}) for a, b in pairs]

    # Build batch prompt
    pairs_text = "\n".join(
        f'[{i}] Entity A: "{a}" | Entity B: "{b}"'
        for i, (a, b) in enumerate(uncached_pairs)
    )

    BATCH_PROMPT = f"""You are an entity resolution engine. For each pair of entities below, determine if they refer to the SAME real-world entity.

Respond with a JSON array (one object per pair, in the same order):

[
  {{"index": 0, "same_entity": true/false, "canonical_name": "...", "confidence": 0.0-1.0, "reasoning": "..."}},
  {{"index": 1, "same_entity": true/false, "canonical_name": "...", "confidence": 0.0-1.0, "reasoning": "..."}}
]

Rules:
- same_entity=true ONLY for clear matches (Abi vs Abi Aryan, Dr. Smith vs Smith)
- Completely different names → same_entity=false
- canonical_name = the most complete/formal version
- Be conservative — prefer false positives over false negatives

Pairs:
{pairs_text}

JSON:"""

    api_key = _get_minimax_key()
    if not api_key:
        return [{"same_entity": False, "canonical_name": a, "confidence": 0.0, "reasoning": "No API key"} for a, b in pairs]

    results_by_index = {}

    try:
        with httpx.Client(timeout=60) as client:
            response = client.post(
                "https://api.minimax.io/anthropic/v1/messages",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "x-api-key": api_key,
                },
                json={
                    "model": "MiniMax-M2.7",
                    "messages": [{"role": "user", "content": BATCH_PROMPT}],
                    "max_tokens": 2048,
                },
            )
            response.raise_for_status()
            data = response.json()

        content = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                content = block.get("text", "")
                break

        if not content:
            raise ValueError("No text content")

        # Try to parse JSON array
        import re as re_module
        first_bracket = content.find("[")
        last_bracket = content.rfind("]")
        if first_bracket != -1 and last_bracket > first_bracket:
            try:
                parsed = json.loads(content[first_bracket:last_bracket + 1])
                for item in parsed:
                    idx = item.get("index")
                    if idx is not None and idx < len(uncached_pairs):
                        result = {k: v for k, v in item.items() if k != "index"}
                        results_by_index[idx] = result
            except json.JSONDecodeError:
                pass

        if len(results_by_index) != len(uncached_pairs):
            # Fallback: extract individual JSON objects
            for match in re_module.finditer(r'\{[^{}]*"index"\s*:\s*(\d+)[^{}]*\}', content):
                try:
                    obj = json.loads(match.group(0))
                    idx = obj.get("index")
                    if idx is not None and idx not in results_by_index:
                        result = {k: v for k, v in obj.items() if k != "index"}
                        results_by_index[idx] = result
                except Exception:
                    pass

    except Exception as e:
        print(f"[entity_resolver] batch resolve error: {e}")

    # Build final results in original pair order
    final_results = {}
    # Add cached results
    for i, (a, b, result) in enumerate(cached_results):
        final_results[-(i + 1)] = result  # negative to sort before uncached

    # Add parsed/uncached results
    for i, (a, b) in enumerate(uncached_pairs):
        if i in results_by_index:
            result = results_by_index[i]
            # Cache it
            cache_key = _er_cache_key(a, b)
            try:
                _cache_er_result(cache_key, a, b, result)
            except Exception:
                pass
            final_results[i] = result
        else:
            final_results[i] = {"same_entity": False, "canonical_name": a, "confidence": 0.0, "reasoning": "Parse error in batch response"}

    return [final_results.get(i, {"same_entity": False, "canonical_name": pairs[i][0], "confidence": 0.0, "reasoning": "Unknown"}) for i in range(len(pairs))]


def _sort_pair(a: str, b: str) -> tuple:
    """Sort a pair for cache key lookup."""
    return tuple(sorted([a.lower().strip(), b.lower().strip()]))


def normalize_name(name: str) -> str:
    """Basic string normalization for entity names.

    Lowercase, strip whitespace, remove common prefixes/suffixes.
    """
    name = name.lower().strip()
    # Remove leading titles
    name = re.sub(r"^(dr\.?|mr\.?|ms\.?|mrs\.?|prof\.?)\s+", "", name)
    return name


def find_candidates(entity_name: str, threshold: float = 0.7) -> list[dict]:
    """Find potential duplicate entities in the database.

    Uses:
    1. Exact match on normalized name
    2. Fuzzy string similarity
    3. Prefix/suffix overlap

    Returns list of candidate entities with similarity scores.
    """
    from . import mongo_memory

    normalized = normalize_name(entity_name)
    candidates = []

    # Step 1: exact normalized match
    exact = mongo_memory.get_canonical_entity(entity_name)
    if exact:
        return [{
            "entity": _strip_mongo_id(dict(exact)),
            "normalized_name": normalized,
            "similarity": 1.0,
        }]

    # Step 2: look for entities with name_lower sharing prefix (filter before comparing)
    db = mongo_memory._get_db()
    prefix = normalized.split()[0][:3] if normalized else ""  # speed: prefix filter
    all_ents = db[mongo_memory.ENTITIES_COL].find(
        {"name_lower": {"$regex": f"^{re.escape(prefix)}"}}
    )

    best_score = 0.0
    best_entity = None
    best_normalized = ""

    for ent in all_ents:
        ent_name = ent.get("name", "")
        ent_normalized = normalize_name(ent_name)

        # Compute similarity
        score = _string_similarity(normalized, ent_normalized)

        if score > best_score:
            best_score = score
            best_entity = ent
            best_normalized = ent_normalized

        if score >= threshold:
            candidates.append({
                "entity": _strip_mongo_id(ent),
                "normalized_name": ent_normalized,
                "similarity": score,
            })

    if not candidates and best_entity and best_score >= 0.5:
        candidates.append({
            "entity": _strip_mongo_id(best_entity),
            "normalized_name": best_normalized,
            "similarity": best_score,
        })

    return sorted(candidates, key=lambda x: x["similarity"], reverse=True)


def _string_similarity(s1: str, s2: str) -> float:
    """Simple string similarity based on common substring length ratio."""
    if not s1 or not s2:
        return 0.0

    # Longest common substring ratio
    len_lcs = _lcs_length(s1, s2)
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 0.0

    return len_lcs / max_len


def _lcs_length(s1: str, s2: str) -> int:
    """Length of longest common substring."""
    m, n = len(s1), len(s2)
    # Space-optimized DP
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    max_len = 0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                curr[j] = prev[j-1] + 1
                max_len = max(max_len, curr[j])
            else:
                curr[j] = 0
        prev, curr = curr, prev

    return max_len


def resolve_entity(entity_name: str, entity_type: str = "CONCEPT",
                   source: str = "") -> str:
    """Main entry point: resolve an entity name to its canonical entity_id.

    1. Normalize the name
    2. Search for candidates in the database
    3. For close matches, use LLM to determine true duplicates
    4. Merge confirmed duplicates into canonical form
    5. Store and return the canonical entity_id
    """
    from . import mongo_memory

    # Check if entity already exists
    existing = mongo_memory.get_canonical_entity(entity_name)
    if existing:
        return existing["entity_id"]

    # Find candidates
    candidates = find_candidates(entity_name)

    canonical_id = None
    canonical_name = entity_name

    for candidate in candidates:
        if candidate["similarity"] >= 0.9:
            # Near-certain match — merge without LLM
            cand_ent = candidate["entity"]
            canonical_name = cand_ent["name"]
            canonical_id = cand_ent["entity_id"]
            break
        elif candidate["similarity"] >= 0.6:
            # Potential match — check with LLM
            cand_ent = candidate["entity"]
            result = resolve_pair(entity_name, cand_ent["name"])

            if result.get("same_entity"):
                canonical_name = result.get("canonical_name", entity_name)
                canonical_id = cand_ent["entity_id"]

                # Merge: update the canonical entity name if needed
                if canonical_name != entity_name:
                    # entity_name becomes an alias
                    mongo_memory.merge_entities(canonical_name, entity_name)
                else:
                    # candidate name becomes alias
                    mongo_memory.merge_entities(canonical_name, cand_ent["name"])
                break

    # If no merge happened, store as new entity
    if canonical_id is None:
        canonical_id = mongo_memory.store_entity(
            entity_name, entity_type, source=source
        )

    return canonical_id


def _strip_mongo_id(doc: dict) -> dict:
    """Remove MongoDB _id from document for JSON serialization."""
    if doc and "_id" in doc:
        doc = dict(doc)
        del doc["_id"]
    return doc


if __name__ == "__main__":
    import json

    # Initialize MongoDB
    mongo_memory.init()

    # Test entity resolution
    e1 = mongo_memory.store_entity("Abi", "PERSON", source="test")
    print(f"Created entity 'Abi': {e1}")

    # Later, encounter "Abi Aryan" — should resolve to same entity
    result = resolve_pair("Abi", "Abi Aryan")
    print(f"resolve_pair(Abi, Abi Aryan): {json.dumps(result, indent=2)}")

    if result.get("same_entity"):
        merged = mongo_memory.merge_entities("Abi Aryan", "Abi")
        print(f"Merged into canonical: {merged}")

    # Check what entities exist
    db = mongo_memory._get_db()
    all_ents = list(db[mongo_memory.ENTITIES_COL].find({}))
    for ent in all_ents:
        print(f"  Entity: {ent['name']} ({ent['entity_type']}) - canonical_id={ent.get('canonical_id', 'SELF')}")

"""entity_resolver.py — Entity resolution and normalization.

Merges duplicate entities (e.g., "Abi" = "Abi Aryan") into canonical forms.
Uses LLM to determine if two entities refer to the same real-world entity.
"""
import json
import re
import sys
import os
from typing import Optional

# Add parent to path for absolute imports within package
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Import via absolute module path
from src import mongo_memory


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


def resolve_pair(name_a: str, name_b: str) -> dict:
    """Use MiniMax LLM to determine if two entity names refer to the same entity.

    Returns dict with same_entity, canonical_name, confidence, reasoning.
    """
    import httpx

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
            raise ValueError("No text content in MiniMax response")

        # Parse JSON - try direct parse first, then extract JSON from text if needed
        import re as re_module

        # First try direct parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON object from the text
        # Look for first { and last } to extract the JSON object
        first_brace = content.find("{")
        last_brace = content.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            json_str = content[first_brace:last_brace + 1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

        # Clean markdown code fences and try again
        content = re_module.sub(r"^```json\s*", "", content)
        content = re_module.sub(r"^```\s*", "", content)
        content = re_module.sub(r"\s*```$", "", content)

        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            return {
                "same_entity": False,
                "canonical_name": name_a,
                "confidence": 0.0,
                "reasoning": f"JSON parse error: {e}, content: {content[:200]}",
            }

    except Exception as e:
        return {
            "same_entity": False,
            "canonical_name": name_a,
            "confidence": 0.0,
            "reasoning": f"Error: {str(e)}",
        }


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

    # Step 2: look for entities with name_lower containing the normalized name
    db = mongo_memory._get_db()
    all_ents = db[mongo_memory.ENTITIES_COL].find({})

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

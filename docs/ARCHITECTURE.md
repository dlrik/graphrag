# Agentic GraphRAG — Architecture Review

**Date:** 2026-04-23
**Reviewer:** Senior Software Architect
**Version:** Based on source code inspection of all Python modules

---

## 1. System Overview

Agentic GraphRAG is a unified memory system for Claude Code that combines three retrieval paradigms into a single queryable layer:

1. **Vector search** (ChromaDB via memory-core HTTP API) — semantic similarity on text chunks
2. **Knowledge graph** (MongoDB) — entity nodes + relation edges with multi-hop traversal
3. **Full-text search** (MongoDB TEXT index) — keyword/broad document retrieval

**Primary entry points:**
- CLI (`graphrag.py`) — `ask`, `ingest`, `neighbors`, `stats`
- Web UI (`src/server.py` at port 8080) — chat interface, raw query view, MongoDB management
- MCP server (`src/mcp_server.py`) — 11 tools exposed to Claude Code harness

**External dependencies:**
- MongoDB 7.0 (self-managed at `localhost:27017`)
- memory-core HTTP API (at `localhost:8765`) — ChromaDB-backed vector store
- MiniMax-M2.7 LLM (via `api.minimax.io/anthropic/v1/messages`)

---

## 2. Data Model

### MongoDB Collections

#### `documents` — Raw ingested content
| Field | Type | Notes |
|-------|------|-------|
| `doc_id` | string | MD5(content+source)[:16], unique |
| `content` | string | Full raw text |
| `source` | string | e.g. `file:/path/to/doc.pdf`, `url:https://...` |
| `metadata` | dict | Source-specific metadata |
| `created_at` | string | ISO timestamp |

**Indexes:** TEXT on `content` (english), `doc_id` (unique), `source`, `created_at`

#### `entities` — Canonical entity nodes
| Field | Type | Notes |
|-------|------|-------|
| `entity_id` | string | MD5(lowercase(name))[:16] |
| `name` | string | Original casing preserved |
| `name_lower` | string | Lowercase, unique via collation (strength=2) |
| `entity_type` | string | UPPER_SNAKE_CASE (PERSON, ORGANIZATION, etc.) |
| `canonical_id` | string | Points to master entity if this is an alias |
| `properties` | dict | Additional metadata |
| `source` | string | Ingestion source tag |
| `created_at` | string | ISO timestamp |
| `updated_at` | string | ISO timestamp |

**Indexes:** `entity_id` (unique), `name_lower` (unique, case-insensitive collation), `canonical_id`, `entity_type`, TEXT on `name`

#### `relations` — Graph edges (triplets)
| Field | Type | Notes |
|-------|------|-------|
| `relation_id` | string | MD5(lower(subject)+predicate+lower(object))[:16] |
| `subject` | string | Entity name |
| `subject_id` | string | MD5 of lower(subject) |
| `predicate` | string | UPPERCASE (MENTIONS, CONNECTED_TO, etc.) |
| `object` | string | Entity name |
| `object_id` | string | MD5 of lower(object) |
| `weight` | float | Edge weight (0.0–1.0) |
| `source` | string | e.g. `graphrag:file:...` |
| `created_at` | string | ISO timestamp |

**Indexes:** `relation_id` (unique), `subject_id`, `object_id`, `predicate`, compound `[(subject_id, predicate)]`, compound `[(object_id, predicate)]`

### memory-core (ChromaDB) Data

The `src/client.py` calls memory-core HTTP API at port 8765 for:
- **Facts** — key-value knowledge items with entity/category/content/importance
- **Vectors** — text chunk embeddings stored in ChromaDB
- **Graph** — subject-predicate-object edges in memory-core's own graph store

### LLM Cache Collections (all in MongoDB `graphrag` DB)

| Collection | Key | TTL | Purpose |
|-----------|-----|-----|---------|
| `llm_cache` | `query_hash` (MD5 of normalized query) | 30 days | Classification query cache |
| `extraction_cache` | `text_hash` (MD5 of chunk text) | 60 days | Graph extraction results |
| `rag_cache` | `cache_key` (MD5 of `rag:{question}:{context[:200]}`) | 30 days | Synthesized RAG answers |

---

## 3. Ingestion Pipeline

### 3A. `ingestion.py` — memory-core only (legacy path)

Flow: `extract_from_url/file` → `clean_text` → `chunk_text` → `graph_extractor.extract` → `_store_entities` + `_store_relations` via `src/client.py` (memory-core HTTP API) + `client.vec_add` (ChromaDB vector)

```
URL/File
  └─► extract_from_url / extract_from_file (httpx → Jina Reader / Firecrawl)
  └─► clean_text (whitespace normalization)
  └─► chunk_text (1000 char chunks, 100 char overlap, sentence-boundary aware)
  └─► graph_extractor.extract (MiniMax LLM → entities + relations)
        [CACHE: text_hash → extraction_cache, 60-day TTL]
  └─► _store_entities → client.add_fact + client.graph_connect (memory-core)
  └─► _store_relations → client.graph_connect (memory-core)
  └─► client.vec_add → ChromaDB vector storage (memory-core)
```

### 3B. `mongo_ingestion.py` — MongoDB unified memory path

Flow: `ingest_file_mongo` / `ingest_url_mongo` / `ingest_files_mongo` → `_ingest_text_mongo`

```
URL/File
  └─► extract_from_url / extract_from_file (via base_ingestion)
  └─► clean_text
  └─► store_document → MongoDB documents collection
  └─► chunk_text (via base_ingestion)
  └─► Parallel extraction: ThreadPoolExecutor(max_workers=5)
        for each chunk:
          └─► graph_extractor.extract (MiniMax LLM)
                [CACHE: text_hash → extraction_cache, 60-day TTL]
          └─► _store_entities_mongo → entity_resolver.resolve_entity → MongoDB entities
                [entity_resolver calls LLM for similarity < 0.9 candidates]
          └─► _store_relations_mongo → MongoDB relations
```

**Parallelization:** Chunk extraction is parallel (ThreadPoolExecutor, 5 workers). Entity resolution (`entity_resolver.resolve_entity`) runs **sequentially per entity per chunk** — this is a bottleneck.

### 3C. Entity Resolution (`entity_resolver.py`)

For each entity name:
1. Check if canonical entity already exists via `mongo_memory.get_canonical_entity`
2. Find candidates via `find_candidates` (exact match + fuzzy prefix filter)
3. If similarity ≥ 0.9 → merge without LLM
4. If similarity ≥ 0.6 and < 0.9 → call `resolve_pair` (MiniMax LLM) to determine if same entity
5. If same → `merge_entities` (canonical_id linking)
6. If no candidate matched → `store_entity` as new canonical

**Caching:** No cache on entity resolution calls. Each resolution may trigger an LLM call.

### 3D. Document Parser (`document_parser.py`)

Format-specific parsers: PDF (pdfplumber), Excel (openpyxl), CSV (pandas), DOCX (python-docx), TXT/MD (raw text). Returns structured `{"text": ..., "chunks": [...], "metadata": {...}, "format": ...}`.

Note: `ingestion.py` uses `parse_document` but `mongo_ingestion.py` does **not** use `document_parser.chunk_text` — it uses `base_ingestion.chunk_text` which is a simpler 1000-char splitter. The document parser's format-specific chunking (e.g., PDF by page) is effectively ignored in the MongoDB ingestion path.

---

## 4. Query Pipeline

### 4A. `query_router.hybrid_search` — Primary query path

```
User Question
  └─► llm_cache.cached_classify_query
        └─► [CACHE HIT] return cached classification
        └─► [CACHE MISS] → classify_query → MiniMax → cache result
  └─► query_type (SIMPLE|MULTI_HOP|BROAD|HYBRID)
  └─► Always: memory-core client.vec_search → vector_results (top_k results)
  └─► If detected_entities or MULTI_HOP/HYBRID:
        for each entity → mongo_memory.get_canonical_entity → mongo_memory.traverse_hops → graph_results
  └─► If BROAD/HYBRID:
        mongo_memory.search_documents → document_results
  └─► Synthesis: f"Found {total} results via {query_type} search"
        [NO LLM call in hybrid_search itself]
```

### 4B. `rag_synthesize.rag_with_query` — Full RAG with LLM synthesis

```
User Question
  └─► _detect_self_aware_query → inject system stats as context block
  └─► hybrid_search (vector + graph + document results)
  └─► Build context_blocks from all results (truncated to 800 chars each)
  └─► rag_synthesize:
        cache_key = MD5(f"rag:{question}:{context[:200]}")
        [CACHE HIT] return cached answer
        [CACHE MISS] → MiniMax prompt with context → natural language answer
        [CACHE: rag:{question}:{context[:200]} → rag_cache, 30-day TTL]
  └─► Return: search results + answer + sources + cached flag
```

**Context truncation:** Document results pass `content[:2000]` characters (line 239 of query_router.py). But in `rag_synthesize`, each context block is truncated to 800 chars (line 104 of rag_synthesize.py).

---

## 5. Module Dependency Graph

```
graphrag.py (CLI)
  ├─► mongo_memory.init()
  ├─► mongo_ingestion.ingest_file_mongo / ingest_url_mongo / ingest_files_mongo
  ├─► query_router.hybrid_search
  ├─► mongo_memory.get_canonical_entity + graph_neighbors
  ├─► graph_quality.get_graph_statistics / find_duplicate_entities / detect_communities
  ├─► llm_cache.get_cache_stats / invalidate_cache

src/
  ├── ingestion.py (base extraction/chunking, memory-core storage)
  │     ├─► graph_extractor.extract
  │     ├─► client (memory-core HTTP API)
  │     └─► document_parser.parse_document
  │
  ├── mongo_ingestion.py (MongoDB storage, entity resolution)
  │     ├─► ingestion (clean_text, chunk_text, extract_from_*)
  │     ├─► graph_extractor.extract
  │     ├─► entity_resolver.resolve_entity
  │     └─► mongo_memory (store_document, store_entity, store_relation)
  │
  ├── mongo_memory.py (MongoDB CRUD, graph traversal)
  │     └─► (direct MongoDB driver, no external deps)
  │
  ├── entity_resolver.py (LLM-based duplicate detection)
  │     ├─► mongo_memory (get_canonical_entity, find_candidates, merge_entities)
  │     └─► MiniMax API (httpx)
  │
  ├── graph_extractor.py (LLM entity/relation extraction)
  │     ├─► mongo_memory (extraction_cache collection)
  │     └─► MiniMax API (httpx)
  │
  ├── query_router.py (query classification + hybrid search)
  │     ├─► llm_cache.cached_classify_query
  │     ├─► client.vec_search (memory-core)
  │     ├─► mongo_memory (get_canonical_entity, traverse_hops, search_documents)
  │     └─► MiniMax API (httpx)
  │
  ├── rag_synthesize.py (LLM-powered answer synthesis)
  │     ├─► llm_cache (rag_cache get/set)
  │     ├─► query_router.hybrid_search
  │     ├─► graph_quality.get_graph_statistics (self-aware queries)
  │     └─► MiniMax API (httpx)
  │
  ├── llm_cache.py (multi-collection cache management)
  │     └─► mongo_memory._get_db()
  │
  ├── graph_quality.py (maintenance, stats, community detection)
  │     └─► mongo_memory (direct MongoDB access)
  │
  ├── client.py (memory-core HTTP API client)
  │     └─► httpx (HTTP client to localhost:8765)
  │
  ├── server.py (Flask web UI)
  │     ├─► mongo_memory, graph_quality, llm_cache
  │     ├─► query_router.hybrid_search
  │     └─► rag_synthesize.rag_with_query
  │
  ├── mcp_server.py (FastMCP server, 11 tools)
  │     ├─► ingestion / mongo_ingestion (for ingest tools)
  │     ├─► mongo_memory (for graph traversal)
  │     ├─► client (memory-core for nl_query_memory)
  │     └─► query_router (for hybrid_search_memory, classify_query)
  │
  ├── document_parser.py (multi-format parsing)
  │     └─► pdfplumber, openpyxl, pandas, python-docx (format-specific)
  │
  ├── observability.py (structured logging)
  └── benchmark_queries.py (latency benchmarking)
```

---

## 6. Storage Architecture

### What lives where and why

| Data | Storage | Rationale |
|------|---------|-----------|
| Raw ingested text | MongoDB `documents` | Full-text search, source tracking, TTL-free |
| Entity nodes | MongoDB `entities` | Canonical dedup, case-insensitive lookup, canonical_id linking |
| Relation edges | MongoDB `relations` | Graph traversal with subject_id/object_id indexes |
| Text chunk vectors | memory-core (ChromaDB) | Semantic similarity search |
| Knowledge graph (memory-core) | memory-core `/graph` endpoint | Separate graph store used by `ingestion.py` path only |
| Facts (memory-core) | memory-core `/facts` endpoint | Key-value knowledge used by `ingestion.py` path only |
| Classification cache | MongoDB `llm_cache` | 30-day TTL, persists across restarts |
| Extraction cache | MongoDB `extraction_cache` | 60-day TTL, chunk text → entities/relations |
| RAG synthesis cache | MongoDB `rag_cache` | 30-day TTL, question + first 200 chars of context |
| Self-aware stats | Injected at query time | No storage needed |

**Key architectural tension:** `ingestion.py` writes to memory-core (facts + graph + vectors) while `mongo_ingestion.py` writes to MongoDB (entities + relations + documents). These are two separate storage paths with partial duplication. The memory-core path is legacy; MongoDB is the primary active path.

---

## 7. LLM Integration Points

| Module | Endpoint | Model | Purpose | Cache |
|--------|----------|-------|---------|-------|
| `graph_extractor.extract` | `POST /anthropic/v1/messages` | MiniMax-M2.7 | Entity/relation extraction from chunks | `extraction_cache` (60d) by `text_hash` |
| `query_router.classify_query` | `POST /anthropic/v1/messages` | MiniMax-M2.7 | Query type classification + entity detection | `llm_cache` (30d) by `query_hash` |
| `entity_resolver.resolve_pair` | `POST /anthropic/v1/messages` | MiniMax-M2.7 | Determine if two entity names refer to same entity | **NONE** |
| `rag_synthesize.rag_synthesize` | `POST /anthropic/v1/messages` | MiniMax-M2.7 | Generate natural language answer from context | `rag_cache` (30d) by `rag:{question}:{context[:200]}` |

**API key retrieval:** All modules duplicate the same `_get_minimax_key()` logic — reads from `~/.claude-code-router/config.json` (CCR Providers array) then `~/.openclaw/openclaw.json` (openclaw models.providers). This is duplicated 5 times across: `graph_extractor.py`, `entity_resolver.py`, `query_router.py`, `rag_synthesize.py`.

**Prompts:**
- Extraction: JSON output with entities/relations from text
- Classification: JSON with query_type (SIMPLE/MULTI_HOP/BROAD/HYBRID), detected_entities, reasoning, suggested_hops
- Entity resolution: JSON with same_entity, canonical_name, confidence, reasoning
- RAG synthesis: Freeform natural language answer from context blocks

---

## 8. Identified Architectural Issues

### Issue 1: Entity Resolution is Sequential and Unbounded (SEVERITY: HIGH)

**Location:** `mongo_ingestion.py` lines 202–226, `_ingest_text_mongo`

**Problem:** After parallel chunk extraction (5 workers), entity storage runs sequentially per entity per chunk. Each `entity_resolver.resolve_entity` call may itself trigger multiple LLM calls (`resolve_pair`) and MongoDB queries (`find_candidates` does a regex prefix scan). For a file with 50 chunks each containing 10 entities, this could mean 500 sequential resolution operations, each with potentially expensive fuzzy matching.

**Fix:** Batch entity resolution — collect all unique entity names across all chunks, resolve in a single pass using `resolve_entities` (which already uses `$in` for bulk lookup), then distribute resolved IDs back to chunks.

---

### Issue 2: RAG Context Truncation Loses Graph Information (SEVERITY: HIGH)

**Location:** `rag_synthesize.py` lines 104, 236–250 and `query_router.py` line 239

**Problem:** `rag_with_query` passes graph traversal results to synthesis using:
```python
context_blocks.append({
    "text": item.get("text", "") or item.get("name", ""),
    "source": item.get("entity_id", "graph"),
})
```
This means graph results contribute only the entity **name** — no relationship context, no predicate information, no multi-hop paths. Meanwhile document results contribute up to 2000 chars and vector results contribute up to 300 chars.

**Fix:** Serialize graph traversal results as structured text including predicate, depth, and connected entities. E.g., `"Doug --USES--> Claude Code (depth 2 via memory)"`.

---

### Issue 3: No Cache on Entity Resolution (SEVERITY: HIGH)

**Location:** `entity_resolver.py`

**Problem:** Every `resolve_pair` call hits MiniMax. For large batch ingests with many similar entity names (e.g., multiple mentions of "Sarah", "John", "Acme Corp" across chunks), each will trigger a fresh LLM call even for identical pairs.

**Fix:** Add a cache collection for entity resolution results, keyed by sorted pair names (e.g., MD5 of `sorted([name_a, name_b]).join("|")`). TTL of 60 days like extraction cache.

---

### Issue 4: Silent Exception Swallowing in Critical Paths (SEVERITY: HIGH)

**Locations:**
- `mongo_ingestion.py` line 73–74: `except Exception as e: print(...)` — relation storage failure silently continues
- `mongo_ingestion.py` line 84: `except Exception as e: print(...)` — document storage failure silently continues
- `entity_resolver.py` lines 166–172: resolve_pair failure returns `same_entity=False` with zero confidence — this is treated as "different entities" and causes incorrect merges
- `graph_extractor.py` line 230: `except Exception as e` — extraction error only prints, doesn't raise
- `query_router.py` line 229: traverse_hops error only prints — entity silently contributes no graph results

**Problem:** In all these cases, errors are suppressed and processing continues. For entity resolution specifically, an LLM failure returns `same_entity=False` which can cause an incorrect entity merge (treating two different entities as the same when confidence is 0.0).

**Fix:** Add error collection to ingestion results dict, propagate failures for critical operations (entity storage, relation storage), and for entity resolution add an `error` field rather than returning a misleading `same_entity=False`.

---

### Issue 5: Double Chunking Inconsistency (SEVERITY: MEDIUM)

**Location:** `mongo_ingestion.py` uses `base_ingestion.chunk_text` (simple 1000-char splitter), while `document_parser.py` has format-aware chunking (PDF by page, Excel by sheet, etc.)

**Problem:** When ingesting a PDF via `mongo_ingestion.ingest_file_mongo`, the document parser's page-aware chunks are discarded — the raw text goes to `chunk_text` which splits by character count, potentially cutting sentences and table cells mid-paragraph.

**Fix:** `mongo_ingestion._ingest_text_mongo` should use document parser's chunks when available (passed via metadata from `parse_document` output), falling back to character-based chunking only for raw text sources.

---

### Issue 6: Extraction Cache Hash Collision Risk (SEVERITY: MEDIUM)

**Location:** `graph_extractor.py` line 49–52

**Problem:** `text_hash` is MD5 of the raw text. The normalization `text.encode("utf-8", errors="replace").decode("utf-8")` handles encoding errors but not semantic equivalence. Two texts that mean the same thing but differ by a space, newline, or word order will have different hashes and won't share cache entries.

While MD5 collisions are computationally infeasible for crafted inputs, the practical issue is that semantically similar texts (e.g., "Sarah works at Acme" vs "Sarah is employed by Acme") won't share cache entries.

**Fix:** Consider n-gram based fingerprinting for semantic similarity rather than exact hash, or accept current behavior as intentional (exact match cache for identical chunks).

---

### Issue 7: MongoDB Regex Prefix Index Usage (SEVERITY: MEDIUM)

**Location:** `entity_resolver.py` line 214

**Problem:** `find_candidates` uses `{"name_lower": {"$regex": f"^{re.escape(prefix)}"}}` where `prefix = normalized.split()[0][:3]`. This requires a regex scan, not an index lookup. The index on `name_lower` is a B-tree; the regex `^prefix` can use the index for the prefix portion, but the case-insensitive collation adds overhead.

**Fix:** Consider prefix-based B-tree range scan instead of regex: `{"name_lower": {"$gte": prefix, "$lt": prefix[:-1] + chr(ord(prefix[-1]) + 1)}}` for the first 3 characters. Or maintain a separate `name_prefix` field with just the first 3 characters for efficient prefix matching.

---

### Issue 8: Code Duplication — API Key Retrieval (SEVERITY: MEDIUM)

**Location:** `_get_minimax_key()` duplicated in `graph_extractor.py`, `entity_resolver.py`, `query_router.py`, `rag_synthesize.py`, `mcp_server.py`

**Problem:** Identical 50-line function copied 5 times. If CCR config format changes, all copies need updating.

**Fix:** Extract to `src/config.py` with a single `_get_minimax_key()` function, imported by all modules.

---

### Issue 9: Ingestion Path Divergence — `ingestion.py` vs `mongo_ingestion.py` (SEVERITY: MEDIUM)

**Location:** Two parallel ingestion systems

**Problem:** `ingestion.py` stores to memory-core (facts, graph, vectors). `mongo_ingestion.py` stores to MongoDB (entities, relations, documents). They share `graph_extractor.extract` and `clean_text`/`chunk_text` from base `ingestion` module, but have diverged in storage backend. The `ingestion.py` path appears legacy — no active CLI command uses it (all `graphrag.py` CLI commands use `mongo_ingestion`). But the MCP server's `use_mongo=False` flag can still activate the legacy path.

**Fix:** Deprecate `ingestion.py` path, remove `use_mongo=False` from MCP server tools, consolidate all ingestion through `mongo_ingestion.py`.

---

### Issue 10: RAG Cache Key Collision Risk (SEVERITY: LOW)

**Location:** `rag_synthesize.py` line 112

**Problem:** Cache key is `MD5(f"rag:{question}:{context[:200]}")`. Two semantically different queries with similar first 200 chars of context will collide. Conversely, the same question with different contexts beyond the first 200 chars will be treated as cache hits even though the full context differs.

**Fix:** Use a more robust cache key that captures the full context hash, or distinguish cache entries by the full context. Consider `MD5(f"rag:{question}:{sha256(context.encode()).hexdigest()[:32]}")`.

---

### Issue 11: Traverse Hops N+1 Pattern in Query Router (SEVERITY: LOW)

**Location:** `query_router.py` lines 216–231

**Problem:** For each detected entity (up to 5), a separate `traverse_hops` call is made. Each traverses from a single start entity. If many entities share a common ancestor, they're traversed independently even though the traversal could be shared.

**Fix:** Batch entity resolution and shared frontier traversal, or accept for current scale.

---

## 9. Scalability Analysis — 10,000 Files

If 10,000 files were ingested:

| Component | Current Behavior | Problem at Scale |
|-----------|-----------------|------------------|
| Chunk extraction | Parallel (5 workers) | ~200k chunks (assuming 20 chunks/file avg). At ~1s LLM call/chunk = ~11 hours wall time. Acceptable with more workers. |
| Entity resolution | Sequential per entity | ~2M entity resolution calls (200k chunks × 10 entities/chunk). Must batch. |
| MongoDB indexes | B-tree + TEXT | TEXT index scan on millions of documents will be slow. Need partition strategy. |
| ChromaDB vectors | memory-core external | Depends on memory-core configuration |
| Cache collections | TTL indexes | OK, but `extraction_cache` at 60-day TTL with 200k entries is fine |
| `find_duplicate_entities` | O(n²) pairwise comparison | Will be catastrophic at 100k+ entities. Uses length-based blocking but still scans all entities. |
| `detect_communities` | BFS over full adjacency map | Loads all relations into memory. Will OOM at millions of edges. |

---

## 10. ASCII Architecture Diagram

```
                                    ┌─────────────────────────────────────────────────────────────┐
                                    │                      USER                                   │
                                    │   (CLI / Web UI :8080 / Claude Code MCP harness)            │
                                    └──────────────────────────┬──────────────────────────────────┘
                                                               │
                            ┌──────────────────────────────────┼───────────────────────────────┐
                            │           graphrag.py (CLI)       │                               │
                            │         src/server.py (Flask)     │                               │
                            │        src/mcp_server.py         │                               │
                            └─────────────────┬────────────────┼───────────────────────────────┘
                                              │                │
                      ┌───────────────────────┼────────────────┼───────────────────────────────┐
                      │                       ▼                ▼                                │
                      │  ┌─────────────────────────────────┐   ┌──────────────────────────┐  │
                      │  │        query_router.py           │   │       mcp_server.py       │  │
                      │  │   (classify + hybrid search)     │   │     (11 MCP tools)        │  │
                      │  └──────────────┬───────────────────┘   └──────────────────────────┘  │
                      │                 │                                                    │
          ┌───────────┼─────────────────┼───────────────────────────────────────────────────┤
          │           │                 │                                                    │
          ▼           ▼                 ▼                                                    ▼
  ┌──────────────┐ ┌────────────┐ ┌──────────────────┐                           ┌─────────────────┐
  │ llm_cache.py │ │ client.py  │ │  rag_synthesize  │                           │ observability.py │
  │ (MongoDB)    │ │ (memory-   │ │    .py           │                           │   (logs/)       │
  │              │ │  core API) │ │                  │                           │                 │
  └──────┬───────┘ └──────┬─────┘ └────────┬─────────┘                           └─────────────────┘
         │                │                │
         │         ┌──────┴─────┐          │
         │         │            │          │
         ▼         ▼            ▼          ▼
  ┌────────────────────────────────────────────────────────────────────────────────────────────┐
  │                           MINIMAX-M2.7 (api.minimax.io)                                    │
  └────────────────────────────────────────────────────────────────────────────────────────────┘

  ┌────────────────────────────────────────────────────────────────────────────────────────────┐
  │                                   MongoDB (localhost:27017)                                  │
  │  ┌──────────────┐  ┌────────────────┐  ┌──────────────┐  ┌──────────┐ ┌────────────┐      │
  │  │  documents   │  │    entities    │  │  relations   │  │ llm_cache│ │extraction │      │
  │  │  (TEXT idx)  │  │ (name_lower)   │  │ (subject_id) │  │          │ │  _cache   │      │
  │  │              │  │                │  │  (object_id) │  │          │ │           │      │
  │  └──────────────┘  └────────────────┘  └──────────────┘  └──────────┘ └────────────┘      │
  └────────────────────────────────────────┬────────────────────────────────────────────────────┘
                                           │
                                           ▼
  ┌────────────────────────────────────────────────────────────────────────────────────────────┐
  │                        memory-core (localhost:8765)                                        │
  │  ┌────────────────┐    ┌────────────────┐    ┌────────────────┐                           │
  │  │   ChromaDB     │    │   /graph       │    │   /facts       │                           │
  │  │   (vectors)    │    │   (edges)      │    │   (k/v)        │                           │
  │  └────────────────┘    └────────────────┘    └────────────────┘                           │
  └────────────────────────────────────────────────────────────────────────────────────────────┘

  Ingestion Flow (mongo_ingestion path):
  ┌────────┐   ┌────────────┐   ┌─────────┐   ┌──────────┐   ┌────────────────┐   ┌──────────┐
  │File/URL│──►│document_   │──►│clean_   │──►│chunk_    │──►│graph_extractor │──►│ entity_  │
  │        │   │parser.py   │   │text     │   │text      │   │   (MiniMax)    │   │resolver  │
  └────────┘   └────────────┘   └─────────┘   └──────────┘   └────────────────┘   │(MiniMax) │
                                                                          │         └──────────┘
                                                                          ▼
                                                             ┌────────────────────────┐
                                                             │     MongoDB            │
                                                             │  entities + relations  │
                                                             └────────────────────────┘
```

---

## 11. Top 10 Architectural Issues Summary

| # | Issue | Severity | Fix Priority |
|---|-------|----------|-------------|
| 1 | Entity resolution is sequential and unbounded — O(n) LLM calls per ingest | HIGH | P0 — Batch resolve all entities before storing |
| 2 | RAG synthesis receives only entity names from graph results, no relational context | HIGH | P0 — Serialize graph traversal as structured text with predicates |
| 3 | No cache on entity resolution — repeated pairs cause redundant LLM calls | HIGH | P1 — Add entity_resolution_cache collection |
| 4 | Silent exception swallowing causes data loss and incorrect merges | HIGH | P1 — Add error collection to results, return errors rather than suppressing |
| 5 | Double chunking inconsistency — document parser's format-aware chunks discarded | MEDIUM | P2 — Pass document parser chunks through to MongoDB ingestion |
| 6 | Extraction cache uses exact hash — no semantic sharing of similar texts | MEDIUM | P2 — Accept or implement n-gram fingerprinting |
| 7 | MongoDB regex prefix scan in find_candidates not using index efficiently | MEDIUM | P2 — Replace with B-tree range scan on prefix field |
| 8 | _get_minimax_key() duplicated 5 times — maintenance hazard | MEDIUM | P2 — Extract to src/config.py |
| 9 | Two ingestion paths (memory-core vs MongoDB) — drift and maintenance burden | MEDIUM | P3 — Deprecate ingestion.py path |
| 10 | RAG cache key collision from truncated context + weak hash | LOW | P3 — Use full context hash or different key strategy |

---

## 12. Top 5 Recommended Improvements

### 1. Batch Entity Resolution (Estimated Impact: 10x ingestion speed improvement)

**What:** Collect all unique entity names across all chunks before storing. Use MongoDB `$in` query to fetch all existing entities in one call. Resolve all candidate pairs in batch. Store all entities in one pass.

**Why:** Currently each entity triggers sequential resolution. For a 50-chunk file with 10 entities/chunk = 500 sequential operations. Batch would reduce to ~10 MongoDB queries + N LLM calls for unresolved candidates only.

### 2. Structured Graph Context in RAG (Estimated Impact: Major answer quality improvement for multi-hop queries)

**What:** Instead of passing only entity names from graph traversal, construct structured text: `"Entity: Doug | Predicate: USES | Object: Claude Code | Depth: 1 via MongoDB"`. Include multi-hop paths as formatted sequences.

**Why:** Current approach passes graph as empty context for SIMPLE queries and only entity names for multi-hop — the relation context that makes graph traversal valuable is lost before synthesis.

### 3. Entity Resolution Cache (Estimated Impact: 5x reduction in LLM calls during batch ingestion)

**What:** Add `entity_resolution_cache` collection in MongoDB. Key = MD5 of sorted pair `name_a|name_b`. TTL = 60 days. Cache both LLM resolutions and high-similarity merges.

**Why:** During batch ingestion of documents about the same people/orgs, entity pairs like ("Sarah", "Sarah Smith") appear repeatedly. Each triggers an LLM call. Caching eliminates redundant calls.

### 4. Error Propagation and Reporting (Estimated Impact: Debugging and data integrity)

**What:** `mongo_ingestion._ingest_text_mongo` result dict already has `errors` list but it's not surfaced to caller in many cases. Make errors fatal for entity/relation storage (at minimum log and increment error counter), and surface in CLI output.

**Why:** Currently a failed entity store silently drops that entity. Over time this leads to incomplete graphs with no indication of data loss.

### 5. Single API Key Configuration (Estimated Impact: Maintainability)

**What:** Create `src/config.py` with a singleton `_get_minimax_key()` that reads from CCR config then openclaw config. All 5 modules import from it.

**Why:** When MiniMax API key location changes (new config format, new env var), only one file needs updating. Currently 5 copies of the same 50-line function would all need updating.

---

## 13. MongoDB Index Strategy Analysis

| Query | Index Used | Effective? |
|-------|-----------|------------|
| `get_canonical_entity(name)` | `name_lower` (unique collation) | YES — direct lookup |
| `get_entity(name=name)` | `name_lower` (unique) | YES — direct lookup |
| `resolve_entities(names)` | `name_lower` with `$in` | YES — single query for all names |
| `find_candidates(prefix)` | `name_lower` with `$regex: ^prefix` | PARTIAL — regex can use index for prefix, but collation adds overhead |
| `traverse_hops(start_id, hops)` | `subject_id` compound, `object_id` compound | YES — `$in` queries on frontier sets |
| `graph_neighbors(entity_id)` | `subject_id`, `object_id` with `$or` | YES — efficient frontier expansion |
| `search_documents(keyword)` | TEXT index on `content` | YES — full-text search |
| `store_entity` (upsert) | `name_lower` (unique) | YES — prevents duplicates |
| `store_relation` (upsert) | `relation_id` (unique) | YES — prevents duplicate edges |

**Missing index:** No index on `entities.entity_type` for queries like "find all PERSON entities". Could be useful for type-specific queries.

**Unused index:** TEXT index on `entities.name` — used only in `entity_search` which is not called from any active query path. Could be removed.
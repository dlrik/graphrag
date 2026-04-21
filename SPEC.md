# Agentic GraphRAG — Project Specification

## Overview

A personal unified memory system for Claude Code, combining vector search, knowledge graph extraction, and graph traversal into a single coherent layer. Built on top of the existing memory-core infrastructure (ChromaDB + Voyage AI + FastAPI).

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         AGENTIC GRAPHRAG                                  │
├──────────────────┬──────────────────────────────────────────────────────┤
│  DATA PIPELINE   │              MEMORY PIPELINE                          │
│                  │                                                        │
│  URIs            │  Clean → Chunk → Graph Extract → Embed → Store       │
│  Notes           │  (open-source     (Voyage AI)    (Knowledge Graph)    │
│  Docs            │   extractor)                                         │
│  Conversations   │                                                        │
│         ↓         │         ┌────────────────────────────────────────┐  │
│         ETL       │         │      UNIFIED MEMORY (memory-core)        │  │
│         ↓         │         │  ┌─────────┐ ┌──────────┐ ┌──────────┐ │  │
│  Raw Docs        │         │  │ Knowledge│ │  Vector  │ │   Text   │ │  │
│  (MongoDB or     │         │  │  Graph  │ │  Index   │ │   Index  │ │  │
│   filesystem)   │         │  │ (graph) │ │(ChromaDB)│ │  (SQLite)│ │  │
│                  │         │  └─────────┘ └──────────┘ └──────────┘ │  │
└──────────────────┴─────────┴────────────────────────────────────────┘  │
│                                    │                                      │
│                        MCP SERVER (FastMCP)                              │
│              nl_query │ query │ deep_search │ ingest                     │
│                                    │                                      │
│                     CLAUDE CODE HARNESS                                   │
│              assistant-memory │ assistant-learn                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Data Model

### Knowledge Graph Object
```json
{
  "id": "uuid",
  "type": "entity" | "relation" | "document",
  "name": "entity name",
  "label": "Person | Task | Episode | Preference | Document | ...",
  "properties": {},
  "relations": [
    {"target": "uuid", "type": "MENTIONS | CONNECTED_TO | HAS | ..."}
  ],
  "vector": [0.123, ...],
  "metadata": {
    "source": "uri | file | conversation",
    "chunk_id": "uuid",
    "created_at": "ISO8601"
  }
}
```

### Node Types
- **Person** — people mentioned or involved
- **Task** — actionable items, goals
- **Episode** — events, sessions, time-bounded activities
- **Preference** — user preferences, stated likes/dislikes
- **Document** — source documents, files, URLs

### Relation Types
- **MENTIONS** — entity references a node
- **CONNECTED_TO** — general relationship
- **HAS** — ownership/composition
- **RELATED_TO** — topic similarity
- **FOLLOWED_BY** — temporal ordering

## Components

### 1. Ingestion Pipeline

**Sources:**
- `ingest_url` — fetch URL via Firecrawl/Jina → extract text
- `ingest_file` — read local file (txt, md, pdf, html)
- `ingest_conversation` — process Claude Code conversation history

**ETL Steps:**
1. **Extract** — pull raw text from source
2. **Clean** — normalize whitespace, remove boilerplate
3. **Chunk** — split into manageable pieces (overlap for context)
4. **Graph Extract** — use LLM to identify entities and relationships
5. **Normalize** — merge duplicate entities ("Abi" = "Abi Aryan")
6. **Embed** — generate vector via Voyage AI
7. **Store** — write to unified memory layer

### 2. Graph Extraction (LLM-based)

Prompt-driven entity/relation extraction from text chunks:
- Identify named entities (people, organizations, locations)
- Extract relations between entities
- Classify entity types
- Normalize names via entity resolution

### 3. Unified Memory Layer (extends existing memory-core)

Extends the running memory-core at `localhost:8765` with:

**Text Index** — keyword search via SQLite FTS
**Vector Index** — semantic search via ChromaDB
**Graph Index** — multi-hop traversal via entity_graph.py

New endpoints needed:
- `POST /kg/entity` — add entity node
- `POST /kg/relation` — add edge
- `GET /kg/traverse?start_id=&hops=2` — graph traversal
- `GET /kg/entity/{id}` — fetch entity with relations

### 4. MCP Server (Prefect FastMCP)

Exposed tools for Claude Code harness:

| Tool | Description |
|------|-------------|
| `nl_query_memory` | Natural language query → semantic + keyword search |
| `query_memory` | Structured query against memory |
| `deep_search_memory` | Progressive graph expansion (2-3 hops) |
| `ingest_url` | Fetch and ingest URL |
| `ingest_file` | Ingest local file |
| `ingest_conversation` | Ingest conversation history |

### 5. Agent Skills

**assistant-memory** — semantic retrieval skill
- Reads query → calls `nl_query_memory` → returns context

**assistant-learn** — memory write skill
- After significant insights → calls appropriate ingest tool

## Phase 1 Scope

### Goal: Minimal viable GraphRAG ingestion + retrieval

**Must have:**
- [ ] Check existing memory-core is running
- [ ] Extend memory-core with graph store (add entity_graph endpoints)
- [ ] Basic ingestion: URL → text → chunk → entity extraction → store
- [ ] Graph traversal endpoint
- [ ] MCP server with core tools
- [ ] Working end-to-end test: ingest URL → query memory

**Out of scope for Phase 1:**
- Conversation ingestion (Phase 2)
- Entity resolution/normalization (Phase 2)
- Multiple hops deep search (Phase 2)
- File-based document ingestion (Phase 2)
- Performance optimization

## Tech Stack

- **Python 3.12+**
- **memory-core** — existing FastAPI + ChromaDB server at :8765
- **FastMCP** — MCP server framework
- **Voyage AI** — embeddings (already configured in memory-core)
- **LLM** — for graph extraction (via OpenRouter CCR or direct)
- **Firecrawl** — URL content extraction
- **SQLite** — extended for graph storage (or new collection if using different backend)

## File Structure

```
/home/doug/Agentic GraphRAG/
├── SPEC.md
├── src/
│   ├── __init__.py
│   ├── memory_core_extensions.py   # New graph store endpoints
│   ├── ingestion.py                 # URL/file/conversation ingestion
│   ├── graph_extractor.py           # LLM-based entity extraction
│   ├── mcp_server.py               # FastMCP server
│   └── client.py                   # Client for memory-core
├── skills/
│   ├── assistant-memory.ts         # Claude Code skill
│   └── assistant-learn.ts          # Claude Code skill
├── tests/
│   └── test_ingestion.py
└── data/
    └── .gitkeep
```

## Dependencies

```txt
fastmcp>=0.1.0
httpx>=0.27.0
python-dotenv>=1.0.0
```

(Existing memory-core dependencies already installed)

## Success Criteria

1. Can ingest a URL and store extracted entities/relations
2. Can query memory with natural language
3. Can traverse graph relationships
4. MCP tools are callable from Claude Code
5. assistant-memory skill returns relevant context

## References

- Original architecture diagram: `HGV3TIgWEAAZVmo.jpeg`
- Concept description: `AgenticGraphRAG`
- Existing memory-core: `/home/doug/memory_core/`

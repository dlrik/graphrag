# GraphRAG — Agentic Memory System

A unified memory system combining vector search, knowledge graphs, and LLM-powered Q&A on your documents. Built with MongoDB, ChromaDB, and MiniMax-M2.7.

## Quick Start

```bash
cd /home/doug/Agentic GraphRAG
source .venv/bin/activate

# Start MongoDB (if not running)
~/mongodb-linux-x86_64-ubuntu2204-7.0.5/bin/mongod --dbpath ~/mongodb-data --fork

# Start the web UI
python3 src/server.py
# → Open http://localhost:8080
```

## Architecture

```
┌─────────────┐     ┌──────────────────────────────────────────┐
│   YOU       │     │            GraphRAG                       │
│             │     │                                          │
│  Web UI     │◄────│  ┌──────────┐  ┌──────────┐  ┌────────┐  │
│  localhost  │     │  │ MongoDB  │  │ ChromaDB │  │ MiniMax│  │
│  :8080      │     │  │ entities │  │ vectors  │  │   LLM  │  │
│             │     │  │ documents│  │          │  │        │  │
│  Claude Code│     │  │ full-text│  │          │  │        │  │
│  (Harness)  │     │  └──────────┘  └──────────┘  └────────┘  │
└─────────────┘     └──────────────────────────────────────────┘
```

## Web UI — Chat with your Documents

Open http://localhost:8080 and click **💬 Chat** to ask questions in natural language. The LLM reads your indexed documents and responds conversationally.

The **🔍 Query** tab shows raw search results (vector, graph, and document hits).

## Claude Code as Harness

Claude Code can act as the intelligent harness that orchestrates your GraphRAG tools. This requires Claude Code CLI installed and the MCP server configured.

### Setup

```bash
# Register the MCP server (from project directory)
cd /home/doug/Agentic\ GraphRAG
claude mcp add-json -s project graphrag '{"type":"stdio","command":"/home/doug/Agentic GraphRAG/.venv/bin/python3","args":["-m","src.mcp_server"],"env":{}}'

# Verify connection
claude mcp list
# → graphrag ✓ Connected
```

### Available Tools

Once registered, Claude Code automatically discovers and can use these tools:

| Tool | Description |
|------|-------------|
| `nl_query_memory` | Natural language query against unified memory |
| `query_memory` | Keyword search across memory |
| `deep_search_memory` | Multi-hop graph traversal from entities |
| `ingest_file` | Ingest a file (PDF, Excel, CSV, DOCX, TXT, MD) |
| `ingest_url` | Ingest content from a URL |
| `ingest_conversation` | Ingest a conversation (list of messages) |
| `ingest_image` | Ingest an image with vision model (GPT-4o) |
| `graph_traverse` | Explore entity relationships |
| `hybrid_search_memory` | Combined vector + graph + full-text search |
| `classify_query` | Detect query type (SIMPLE/MULTI_HOP/BROAD/HYBRID) |
| `record_query_feedback` | Record feedback for the feedback loop |

### Usage

```bash
# Start a Claude Code session in the project directory
cd /home/doug/Agentic GraphRAG
claude

# Now ask questions naturally:
# "what's in my memory about Q1 financials?"
# "show me everything about Doug's projects"
# "ingest the meeting notes from last Tuesday"
# "who are the people in my knowledge graph?"
```

Claude Code will detect available MCP tools and use them to answer your questions, ingest new content, and explore your memory graph.

## CLI Commands

```bash
# Query the unified memory
python3 graphrag.py ask "what is GraphRAG"

# Ingest a file
python3 graphrag.py ingest /path/to/file.pdf

# Explore graph neighbors
python3 graphrag.py neighbors "Doug" --hops 2

# View statistics
python3 graphrag.py stats

# Find duplicate entities
python3 graphrag.py duplicates

# Detect communities
python3 graphrag.py communities

# Cache management
python3 graphrag.py cache
```

## Ingesting Documents

Files are parsed with structure awareness:

- **PDF** — page-level chunking
- **Excel** — per-sheet chunking
- **CSV** — column headers preserved
- **DOCX** — paragraph-level chunking
- **TXT/MD** — paragraph-level chunking

```bash
python3 graphrag.py ingest /path/to/document.pdf
```

After ingestion:
1. Text is extracted and chunked
2. Entities and relations are extracted via MiniMax LLM
3. Entities are deduplicated and merged via entity resolution
4. Chunks go to ChromaDB (vector search)
5. Entities/relations go to MongoDB (graph)
6. Raw text goes to MongoDB (full-text search)

## Self-Aware Queries

The Chat tab automatically answers questions about itself:

- "how many documents are indexed?" → "8 documents are indexed"
- "how many entities are in the graph?"
- "what have been ingested?"

System stats are injected into context when these query patterns are detected.

## Dependencies

- MongoDB 7.0 (self-hosted at `~/mongodb-data`)
- Python 3.12+ with `.venv`
- MiniMax-M2.7 API (configured in CCR or openclaw)
- ChromaDB (via memory-core, 932 chunks indexed)

## Files

| File | Purpose |
|------|---------|
| `src/server.py` | Flask web UI + MongoDB management |
| `src/rag_synthesize.py` | LLM-powered RAG synthesis |
| `src/document_parser.py` | Multi-format document parsing |
| `src/query_router.py` | Query classification + hybrid retrieval |
| `src/mongo_memory.py` | MongoDB-backed memory layer |
| `src/entity_resolver.py` | LLM-powered duplicate detection |
| `src/graph_extractor.py` | Entity/relation extraction |
| `src/llm_cache.py` | LLM response caching |
| `src/mcp_server.py` | FastMCP server (11 tools) |
| `src/ingestion.py` | File ingestion pipeline |
| `graphrag.py` | CLI interface |
| `.mcp.json` | Claude Code MCP server config |
| `ui/index.html` | Light-themed web interface |

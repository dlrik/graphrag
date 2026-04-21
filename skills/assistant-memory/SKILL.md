---
name: assistant-memory
description: >
  GraphRAG semantic retrieval skill. Use when Doug asks memory-related queries like
  "search my memory", "what do you know about", "find things related to X",
  "query my knowledge graph", or similar semantic search requests.
  This skill provides natural language access to the unified memory layer.
triggers:
  - "search memory"
  - "query memory"
  - "what do you know about"
  - "find things related to"
  - "memory search"
  - "my knowledge graph"
  - "semantic search"
  - "hybrid search"
---

# assistant-memory — Natural Language Memory Retrieval (Phase 2)

## Memory System Architecture

Doug has a unified memory system combining:
- **MongoDB** (graphrag database) — Knowledge Graph + Documents + Entities
- **ChromaDB** (memory-core at :8765) — Vector search via Voyage AI
- **SQLite** (memory-core at :8765) — Structured facts

## Query Routing

This skill uses intelligent query routing (Phase 2 enhancement):

| Query Type | Strategy |
|------------|----------|
| SIMPLE | Direct vector + fact search |
| MULTI_HOP | Graph traversal from detected entities |
| BROAD | Full-text document search |
| HYBRID | Combines vector + graph + document |

## Quick Reference

### Natural Language Query (with query routing)
```
assistant_memory_nl_query(query="what does Doug know about AI", top_k=5)
```

### Hybrid Search (Phase 2 - combines all paths)
```
assistant_memory_hybrid_search(query="how does GraphRAG relate to MongoDB", top_k=5, max_hops=2)
```

### Structured Fact Query
```
assistant_memory_query(entity="doug", category="preference", limit=20)
```

### Graph Traversal
```
assistant_memory_graph_traverse(entity="Claude", hops=2)
```

### Deep Search (multi-hop with query classification)
```
assistant_memory_deep_search(query="what projects is Doug working on", hops=2, top_k=3)
```

## Implementation

The tools are accessed via HTTP API to memory-core and MongoDB:

```bash
# Natural language / semantic search (ChromaDB vector)
curl -s "http://localhost:8765/vec/search?query=${QUERY}&top_k=${TOP_K}"

# Keyword facts search (memory-core SQLite)
curl -s "http://localhost:8765/facts?query=${QUERY}&limit=20"

# Graph neighbors (MongoDB or memory-core)
curl -s "http://localhost:8765/graph/neighbors?entity=${ENTITY}"

# MongoDB direct traversal (multi-hop)
python3 -c "
import sys; sys.path.insert(0, '/home/doug/Agentic GraphRAG')
from src import mongo_memory, query_router
result = query_router.hybrid_search('your query', top_k=5)
print(result)
"
```

## Query Classification

Queries are automatically classified as:
- **SIMPLE** (1 hop): "what is MongoDB", "who is Doug"
- **MULTI_HOP** (2-3 hops): "how does X relate to Y", "who are Sarah's colleagues"
- **BROAD** (aggregated): "show me everything", "what have I learned"
- **HYBRID** (combined): Complex queries needing multiple strategies

## Return Format

- **hybrid_search**: Returns `query_type`, `vector_results`, `graph_results`, `document_results`, `synthesis`
- **query_memory**: Returns list of matching facts with entity, content, category
- **graph_traverse**: Returns entity + traversal results with depth
- **deep_search**: Returns initial results + MongoDB multi-hop expansion

## Examples

**"what does Doug know about the memory system"**
```bash
curl -s "http://localhost:8765/vec/search?query=Doug memory system&top_k=5"
```

**"how does GraphRAG connect to MongoDB"**
```bash
python3 -c "
from src import query_router
result = query_router.hybrid_search('how does GraphRAG connect to MongoDB', top_k=5)
for g in result['graph_results']:
    print(g)
"
```

**"who are Doug's connections in the graph"**
```bash
python3 -c "
from src import mongo_memory
ent = mongo_memory.get_canonical_entity('Doug')
if ent:
    trav = mongo_memory.traverse_hops(ent['entity_id'], hops=2)
    for t in trav:
        print(t)
"
```

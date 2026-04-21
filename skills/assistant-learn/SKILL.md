---
name: assistant-learn
description: >
  GraphRAG ingestion skill. Use when Doug wants to store new information in memory,
  ingest documents, URLs, conversations, or images. Also handles graph feedback loops.
  Triggers: "ingest this", "add to memory", "remember this document",
  "index this URL", "learn from this file", "analyze this image/diagram".
---

# assistant-learn — Memory Ingestion (Phase 2)

## Ingestion Pipeline (Phase 2)

Sources flow through Extract → Clean → Chunk → Graph Extract (MiniMax) → Entity Resolution → Store:

1. **Entities** → MongoDB entities collection (with canonical resolution)
2. **Relations** → MongoDB relations collection (triplets)
3. **Documents** → MongoDB documents collection + ChromaDB vector store
4. **Feedback** → Graph updates from successful retrievals

## Ingestion Sources

| Source | Tool | Notes |
|--------|------|-------|
| URL | `ingest_url` | Jina Reader primary, Firecrawl fallback |
| Local file | `ingest_file` | txt, md, pdf, json, py, js, html |
| Conversation | `ingest_conversation` | List of {role, content} dicts |
| Image/Diagram | `ingest_image` | MiniMax vision for entity extraction |

## Quick Reference

### Ingest a URL (MongoDB unified memory)
```
assistant_memory_ingest_url(url="https://example.com/article", use_mongo=True)
```

### Ingest a File
```
assistant_memory_ingest_file(path="/home/doug/docs/notes.md", entity="doug", use_mongo=True)
```

### Ingest Conversation
```
assistant_memory_ingest_conversation(messages=[{"role": "user", "content": "..."}], session_id="abc123", use_mongo=True)
```

### Ingest Image/Diagram (Phase 2)
```
assistant_memory_ingest_image(path="/path/to/diagram.jpeg", description="architecture diagram")
```

### Record Query Feedback (Phase 2 - feedback loop)
```
assistant_memory_record_feedback(query="what is GraphRAG", result_entities=["GraphRAG"], result_relations=[{"subject": "GraphRAG", "predicate": "USES", "object": "MongoDB"}])
```

## Implementation

Call via MongoDB-backed ingestion module:

```bash
cd /home/doug/Agentic\ GraphRAG
source .venv/bin/activate
python3 -c "
from src import mongo_ingestion
result = mongo_ingestion.ingest_url_mongo('https://example.com')
print(result)
"
```

## Entity Resolution (Phase 2)

Duplicate entities are automatically resolved:
- "Abi" + "Abi Aryan" → merged to canonical "Abi Aryan"
- Uses MiniMax to determine if two names refer to the same entity
- Confidence scoring (0.85+ triggers merge)

## Feedback Loops (Steps 12-13)

After successful retrievals, record results back to graph:

```python
from src import mongo_ingestion
mongo_ingestion.record_query_feedback(
    query="what is GraphRAG",
    result_entities=["GraphRAG", "MongoDB"],
    result_relations=[{"subject": "GraphRAG", "predicate": "USES", "object": "MongoDB"}]
)
```

## Node Types

- PERSON, ORGANIZATION, LOCATION, CONCEPT
- TASK, DOCUMENT, TECHNOLOGY, PROJECT
- PREFERENCE, DECISION, GOAL

## Relation Types

- MENTIONS, CONNECTED_TO, HAS, RELATED_TO
- FOLLOWS, WORKS_AT, LOCATED_IN, INTERESTED_IN
- USES, BUILT, INTEGRATES_WITH, CONNECTED_VIA

# Indexing — Agentic GraphRAG

MongoDB indexes for the graphrag database. Run after any fresh `init()` to ensure all performance indexes are in place.

## Collections & Indexes

### entities
| Index | Key | Purpose |
|-------|-----|---------|
| `entity_id_1` | `{entity_id: 1}` | Direct lookup by entity_id |
| `name_lower_1` | `{name_lower: 1}` | Case-insensitive entity lookup (unique) |
| `canonical_id_1` | `{canonical_id: 1}` | Alias resolution (follow canonical chain) |
| `entity_type_1` | `{entity_type: 1}` | Filter entities by type |
| `name_text` | text index on `name` | Full-text entity search |
| `entity_type_name_lower` | `{entity_type, name_lower}` | Type-filtered + name lookup |
| `canonical_id_entity_type` | `{canonical_id, entity_type}` | Canonical chain + type filtering |

### relations
| Index | Key | Purpose |
|-------|-----|---------|
| `relation_id_1` | `{relation_id: 1}` | Direct lookup by relation_id |
| `subject_id_1` | `{subject_id: 1}` | Graph traversal from subject |
| `object_id_1` | `{object_id: 1}` | Graph traversal from object |
| `predicate_1` | `{predicate: 1}` | Filter by relation type |
| `subject_id_1_predicate_1` | `{subject_id, predicate}` | Traversal + predicate filter |
| `object_id_1_predicate_1` | `{object_id, predicate}` | Traversal + predicate filter |
| `predicate_created_at` | `{predicate, created_at}` | Temporal queries by type |
| `subject_predicate_object` | `{subject_id, predicate, object_id}` | Full triplet lookup |

### documents
| Index | Key | Purpose |
|-------|-----|---------|
| `text_index` | text index on `content` | Full-text document search |
| `doc_id_1` | `{doc_id: 1}` | Direct document lookup (unique) |
| `source_1` | `{source: 1}` | Filter by source (url, file, etc.) |
| `created_at_1` | `{created_at: 1}` | Temporal queries |
| `source_created_at` | `{source, created_at}` | Source-scoped temporal queries |

## Creating Indexes

```bash
cd /home/doug/Agentic\ GraphRAG
source .venv/bin/activate
python3 -c "
from src import mongo_memory
mongo_memory.init()

from pymongo import ASCENDING
db = mongo_memory._get_db()

indexes = [
    (mongo_memory.ENTITIES_COL, [('entity_type', ASCENDING), ('name_lower', ASCENDING)], 'entity_type_name_lower'),
    (mongo_memory.RELATIONS_COL, [('predicate', ASCENDING), ('created_at', ASCENDING)], 'predicate_created_at'),
    (mongo_memory.RELATIONS_COL, [('subject_id', ASCENDING), ('predicate', ASCENDING), ('object_id', ASCENDING)], 'subject_predicate_object'),
    (mongo_memory.ENTITIES_COL, [('canonical_id', ASCENDING), ('entity_type', ASCENDING)], 'canonical_id_entity_type'),
    (mongo_memory.DOCUMENTS_COL, [('source', ASCENDING), ('created_at', ASCENDING)], 'source_created_at'),
]

for col, fields, name in indexes:
    try:
        db[col].create_index(fields, name=name)
        print(f'Created: {col}.{name}')
    except Exception as e:
        print(f'Skip/Error {col}.{name}: {e}')
"
```

## Benchmarking

```bash
source .venv/bin/activate
python3 src/benchmark_queries.py
```

See `benchmark_queries.py` for query-level latency measurements.
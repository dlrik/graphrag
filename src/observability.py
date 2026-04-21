"""observability.py — Logging, metrics, and tracing for Agentic GraphRAG.

Provides structured logging for retrieval paths, confidence scores, token usage,
and query performance monitoring.
"""
import sys, os, json, time
from datetime import datetime as dt
from typing import Optional
from functools import wraps

# Logging configuration
LOG_DIR = "/home/doug/Agentic GraphRAG/logs"
LOG_FILE = os.path.join(LOG_DIR, "graphrag_observability.log")
os.makedirs(LOG_DIR, exist_ok=True)


def _ensure_log_file():
    """Ensure log file exists."""
    if not os.path.exists(LOG_FILE):
        open(LOG_FILE, "w").write("# Agentic GraphRAG Observability Log\n")
        open(LOG_FILE + ".metrics", "w").write("# Metrics Log\n")


def _log_entry(level: str, event: str, data: dict):
    """Write a structured log entry."""
    _ensure_log_file()
    entry = {
        "timestamp": dt.now().isoformat(),
        "level": level,
        "event": event,
        **data,
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def log_info(event: str, **data):
    _log_entry("INFO", event, data)


def log_warn(event: str, **data):
    _log_entry("WARN", event, data)


def log_error(event: str, **data):
    _log_entry("ERROR", event, data)


def log_query(query: str, query_type: str, classification: dict,
              vector_count: int, graph_count: int, doc_count: int,
              latency_ms: float, tokens_used: int = 0):
    """Log a query execution with full metadata."""
    log_info("QUERY_EXECUTED",
        query=query[:100],
        query_type=query_type,
        detected_entities=classification.get("detected_entities", []),
        suggested_hops=classification.get("suggested_hops", 1),
        vector_results=vector_count,
        graph_results=graph_count,
        document_results=doc_count,
        total_results=vector_count + graph_count + doc_count,
        latency_ms=latency_ms,
        tokens_used=tokens_used,
    )


def log_ingestion(source: str, chunks: int, entities: int, relations: int,
                  errors: list, latency_ms: float):
    """Log an ingestion operation."""
    log_info("INGESTION_COMPLETED",
        source=source[:100],
        chunks_processed=chunks,
        entities_stored=entities,
        relations_stored=relations,
        error_count=len(errors),
        errors=errors[:5] if errors else [],
        latency_ms=latency_ms,
    )


def log_entity_resolution(name_a: str, name_b: str, same_entity: bool,
                          confidence: float, canonical_name: str, latency_ms: float):
    """Log entity resolution decision."""
    log_info("ENTITY_RESOLVED",
        name_a=name_a,
        name_b=name_b,
        same_entity=same_entity,
        confidence=confidence,
        canonical_name=canonical_name,
        latency_ms=latency_ms,
    )


def log_graph_traversal(entity: str, hops: int, results_count: int,
                         latency_ms: float):
    """Log a graph traversal operation."""
    log_info("GRAPH_TRAVERSAL",
        entity=entity,
        hops=hops,
        results_count=results_count,
        latency_ms=latency_ms,
    )


def log_feedback_loop(query: str, entities_stored: int, relations_stored: int):
    """Log feedback loop recording."""
    log_info("FEEDBACK_RECORDED",
        query=query[:100],
        entities_stored=entities_stored,
        relations_stored=relations_stored,
    )


class QueryLogger:
    """Context manager for tracking query performance."""

    def __init__(self, query: str):
        self.query = query
        self.start_time = time.time()
        self.vector_count = 0
        self.graph_count = 0
        self.doc_count = 0
        self.query_type = "UNKNOWN"
        self.tokens_used = 0
        self.errors = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        latency_ms = (time.time() - self.start_time) * 1000
        if exc_type:
            log_error("QUERY_FAILED", {
                "query": self.query[:100],
                "error": str(exc_val),
                "latency_ms": latency_ms,
            })
            return False

        log_query(
            query=self.query,
            query_type=self.query_type,
            classification={"detected_entities": [], "suggested_hops": 1},
            vector_count=self.vector_count,
            graph_count=self.graph_count,
            doc_count=self.doc_count,
            latency_ms=latency_ms,
            tokens_used=self.tokens_used,
        )
        return False

    def set_results(self, query_type: str, vector_count: int, graph_count: int, doc_count: int):
        self.query_type = query_type
        self.vector_count = vector_count
        self.graph_count = graph_count
        self.doc_count = doc_count

    def add_tokens(self, count: int):
        self.tokens_used += count


def get_recent_logs(lines: int = 50) -> list[dict]:
    """Get recent log entries."""
    _ensure_log_file()
    entries = []
    with open(LOG_FILE) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                try:
                    entries.append(json.loads(line))
                except Exception:
                    pass
    return entries[-lines:]


def get_metrics_summary() -> dict:
    """Get summary metrics from the metrics log."""
    _ensure_log_file()

    total_queries = 0
    total_ingestions = 0
    total_entities = 0
    avg_latency = 0.0

    entries = get_recent_logs(lines=500)

    latencies = []
    for entry in entries:
        if entry.get("event") == "QUERY_EXECUTED":
            total_queries += 1
            latencies.append(entry.get("latency_ms", 0))
        elif entry.get("event") == "INGESTION_COMPLETED":
            total_ingestions += 1
            total_entities += entry.get("entities_stored", 0)
            latencies.append(entry.get("latency_ms", 0))

    if latencies:
        avg_latency = sum(latencies) / len(latencies)

    return {
        "total_queries": total_queries,
        "total_ingestions": total_ingestions,
        "total_entities_stored": total_entities,
        "avg_latency_ms": round(avg_latency, 2),
        "log_entries": len(entries),
    }


if __name__ == "__main__":
    # Test logging
    log_info("TEST_EVENT", {"key": "value"})
    log_entity_resolution("Abi", "Abi Aryan", True, 0.85, "Abi Aryan", 150.0)

    summary = get_metrics_summary()
    print("Metrics summary:", json.dumps(summary, indent=2))

    recent = get_recent_logs(lines=5)
    print("Recent logs:", json.dumps(recent, indent=2))

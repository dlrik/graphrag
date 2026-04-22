r"""server.py — GraphRAG Web UI + MongoDB management.

Run: cd /home/doug/Agentic GraphRAG && source .venv/bin/activate && python3 src/server.py
Then open: http://localhost:8080
"""
import sys, os, json, subprocess, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template_string, jsonify, request
from src import mongo_memory, graph_quality, llm_cache
from src.query_router import hybrid_search
from src.rag_synthesize import rag_with_query

app = Flask(__name__)

MONGO_BIN = os.path.expanduser("~/mongodb-linux-x86_64-ubuntu2204-7.0.5/bin/mongod")
MONGO_DATA = os.path.expanduser("~/mongodb-data")
MONGO_PORT = 27017

# ---------------------------------------------------------------------------
# MongoDB process management
# ---------------------------------------------------------------------------

def mongo_is_running() -> bool:
    """Check if mongod process is alive."""
    result = subprocess.run(["pgrep", "-f", "mongod"], capture_output=True)
    return result.returncode == 0


def mongo_start() -> dict:
    """Start MongoDB daemon."""
    if mongo_is_running():
        return {"status": "already_running"}

    os.makedirs(MONGO_DATA, exist_ok=True)
    log_path = os.path.join(MONGO_DATA, "mongod.log")
    pid_path = os.path.join(MONGO_DATA, "mongod.pid")

    cmd = [
        MONGO_BIN,
        "--dbpath", MONGO_DATA,
        "--port", str(MONGO_PORT),
        "--logpath", log_path,
        "--pidfilepath", pid_path,
        "--fork",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return {"status": "started", "log": log_path}
    return {"status": "failed", "error": result.stderr[:200]}


def mongo_stop() -> dict:
    """Stop MongoDB daemon."""
    if not mongo_is_running():
        return {"status": "already_stopped"}

    # Find PID and kill gracefully
    result = subprocess.run(["pgrep", "-f", "mongod"], capture_output=True, text=True)
    pids = [int(p) for p in result.stdout.strip().split("\n") if p]

    for pid in pids:
        try:
            subprocess.run(["kill", str(pid)], timeout=10)
        except Exception:
            subprocess.run(["kill", "-9", str(pid)], timeout=5)

    return {"status": "stopped", "pids_killed": pids}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the GraphRAG UI."""
    try:
        # UI is at project_root/ui/index.html
        ui_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ui", "index.html")
        with open(ui_path, "r") as f:
            html = f.read()
        return render_template_string(html)
    except FileNotFoundError:
        return "UI not found. Run from project root.", 404


@app.route("/api/status")
def api_status():
    """Return MongoDB and system status."""
    mongo_ok = mongo_is_running()
    try:
        mongo_memory.init()
        stats = graph_quality.get_graph_statistics()
        cache_stats = llm_cache.get_cache_stats()
        mongo_docs = stats["totals"]["documents"]
    except Exception as e:
        mongo_docs = 0
        stats = {}
        cache_stats = {}

    return jsonify({
        "mongo": {
            "running": mongo_ok,
            "port": MONGO_PORT,
        },
        "graphrag": {
            "entities": stats.get("totals", {}).get("entities", 0),
            "relations": stats.get("totals", {}).get("relations", 0),
            "documents": stats.get("totals", {}).get("documents", 0),
            "cached_queries": cache_stats.get("total_cached", 0),
        },
    })


@app.route("/api/mongo/start", methods=["POST"])
def api_mongo_start():
    return jsonify(mongo_start())


@app.route("/api/mongo/stop", methods=["POST"])
def api_mongo_stop():
    return jsonify(mongo_stop())


@app.route("/api/stats")
def api_stats():
    """Full graph statistics."""
    return jsonify(graph_quality.get_graph_statistics())


@app.route("/api/ask")
def api_ask():
    """Run a query against the unified memory (chunk-based results)."""
    query = request.args.get("q", "")
    if not query:
        return jsonify({"error": "no query"})
    result = hybrid_search(query, top_k=5, max_hops=2)
    return jsonify(result)


@app.route("/api/rag")
def api_rag():
    """Run RAG query with LLM synthesis (natural language answer)."""
    query = request.args.get("q", "")
    if not query:
        return jsonify({"error": "no query"})
    result = rag_with_query(query, top_k=5, max_hops=2)
    return jsonify(result)


@app.route("/api/neighbors/<entity_name>")
def api_neighbors(entity_name):
    """Get neighbors of an entity."""
    ent = mongo_memory.get_canonical_entity(entity_name)
    if not ent:
        ent = mongo_memory.get_entity(name=entity_name)
    if not ent:
        return jsonify({"error": "entity not found"})

    neighbors = mongo_memory.graph_neighbors(ent["entity_id"], max_depth=2)
    return jsonify({
        "entity": ent["name"],
        "entity_id": ent["entity_id"],
        "entity_type": ent.get("entity_type"),
        "neighbors": neighbors,
        "count": len(neighbors),
    })


@app.route("/api/communities")
def api_communities():
    return jsonify({"communities": graph_quality.detect_communities()})


@app.route("/api/duplicates")
def api_duplicates():
    return jsonify({"duplicates": graph_quality.find_duplicate_entities()})


if __name__ == "__main__":
    print("GraphRAG UI starting at http://localhost:8080")
    print("MongoDB management: POST /api/mongo/start and /api/mongo/stop")
    app.run(host="0.0.0.0", port=8080, debug=False)
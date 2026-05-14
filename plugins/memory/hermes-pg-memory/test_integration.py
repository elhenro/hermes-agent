"""Integration tests for the PG memory provider.

Run with PG + Ollama up:
    python plugins/memory/hermes-pg-memory/test_integration.py

The plugin directory uses hyphens (invalid Python module name), so we
boot the package via the Hermes plugin loader, then pull submodules
out of sys.modules.

NOTE: All bootstrapping is inside __main__ — the Hermes plugin loader
exec's every *.py in this dir at plugin-load time, so top-level work
would cause recursion.
"""

import os
import sys
from pathlib import Path


DSN = os.environ.get(
    "HERMES_PG_DSN",
    "postgresql://hermes:hermes@localhost:5432/hermes",
)


def _bootstrap():
    """Set up sys.path + load plugin so submodules are in sys.modules."""
    this_dir = Path(__file__).resolve().parent
    hermes_root = this_dir.parent.parent.parent
    sys.path.insert(0, str(hermes_root))

    from plugins.memory import load_memory_provider
    provider = load_memory_provider("hermes-pg-memory")
    if provider is None:
        print("FATAL: could not load hermes-pg-memory provider")
        sys.exit(1)

    pkg = "plugins.memory.hermes-pg-memory"
    return (
        sys.modules[f"{pkg}.schema"].PgConnection,
        sys.modules[f"{pkg}.embed"].Embedder,
        sys.modules[f"{pkg}.store"].MemoryStore,
    )


def _new_store(PgConnection, Embedder, MemoryStore):
    db = PgConnection(DSN)
    db.connect()
    db.ensure_schema()
    emb = Embedder()
    return db, MemoryStore(db, emb)


def test_connection(PgConnection, Embedder, MemoryStore):
    db = PgConnection(DSN)
    db.connect()
    db.ensure_schema()
    row = db.fetch_one("SELECT 1 AS ok")
    assert row and row["ok"] == 1
    db.close()
    print("ok  connection")


def test_embedder(PgConnection, Embedder, MemoryStore):
    emb = Embedder()
    assert emb.is_available(), "Ollama not running"
    vec = emb.embed("test query")
    assert vec is not None, "embed returned None"
    assert len(vec) in (384, 768), f"unexpected dim: {len(vec)}"
    print(f"ok  embedder ({len(vec)} dims)")


def test_memory_crud(PgConnection, Embedder, MemoryStore):
    db, store = _new_store(PgConnection, Embedder, MemoryStore)
    try:
        mid = store.store_memory(
            "test memory content for crud",
            metadata={"test": True},
            source="test",
        )
        assert mid is not None
        results = store.search_memories("test memory", limit=5)
        assert any("test memory content" in r["content"] for r in results)
        stats = store.get_stats()
        assert stats["memories"] >= 1
        store.delete_memory(mid)
        print(f"ok  memory crud (stats: {stats})")
    finally:
        db.close()


def test_research(PgConnection, Embedder, MemoryStore):
    db, store = _new_store(PgConnection, Embedder, MemoryStore)
    try:
        rid = store.store_research(
            query="test query for research",
            content="This is some research content about AI agents and pgvector.",
            url="https://example.com",
            title="Test Research",
            tags=["ai", "test"],
            source="test",
        )
        assert rid is not None
        results = store.search_research("AI agents", limit=5)
        assert len(results) >= 1
        print(f"ok  research ({len(results)} results)")
    finally:
        db.close()


def test_entity(PgConnection, Embedder, MemoryStore):
    db, store = _new_store(PgConnection, Embedder, MemoryStore)
    try:
        eid = store.upsert_entity(
            "Hermes Agent (test)",
            "project",
            "Open-source AI agent framework",
        )
        assert eid
        eid2 = store.upsert_entity(
            "PostgreSQL (test)",
            "tool",
            "Relational database with pgvector",
        )
        assert eid2
        store.relate_entities(
            "Hermes Agent (test)", "PostgreSQL (test)", "uses"
        )
        results = store.search_entities("Hermes", limit=5)
        assert len(results) >= 1
        print(f"ok  entity ({len(results)} results)")
    finally:
        db.close()


def test_unified_search(PgConnection, Embedder, MemoryStore):
    db, store = _new_store(PgConnection, Embedder, MemoryStore)
    try:
        results = store.unified_search("AI agent", limit_per_source=2)
        assert "memories" in results
        assert "research" in results
        assert "entities" in results
        total = sum(len(v) for v in results.values())
        print(f"ok  unified search ({total} total)")
    finally:
        db.close()


if __name__ == "__main__":
    print("Running PG memory provider integration tests...")
    print(f"DSN: {DSN}")
    print()
    PgConn, Emb, Store = _bootstrap()
    test_connection(PgConn, Emb, Store)
    test_embedder(PgConn, Emb, Store)
    test_memory_crud(PgConn, Emb, Store)
    test_research(PgConn, Emb, Store)
    test_entity(PgConn, Emb, Store)
    test_unified_search(PgConn, Emb, Store)
    print()
    print("all tests passed")

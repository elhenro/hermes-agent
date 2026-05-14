"""Connection pool and schema management for Hermes PG memory provider."""

from __future__ import annotations

import logging
import re
from typing import Optional

import psycopg2
from psycopg2 import pool as pg_pool
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

DDL_STATEMENTS = [
    "CREATE EXTENSION IF NOT EXISTS vector",
    "CREATE EXTENSION IF NOT EXISTS pg_trgm",
    "CREATE EXTENSION IF NOT EXISTS pgcrypto",
    """
    CREATE TABLE IF NOT EXISTS memories (
        id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        content     TEXT NOT NULL,
        embedding   vector(768),
        metadata    JSONB DEFAULT '{}',
        trust_score FLOAT DEFAULT 0.5,
        source      TEXT DEFAULT 'turn',
        session_id  TEXT,
        platform    TEXT,
        created_at  TIMESTAMPTZ DEFAULT NOW(),
        accessed_at TIMESTAMPTZ DEFAULT NOW()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_memories_embedding ON memories USING hnsw (embedding vector_cosine_ops)",
    "CREATE INDEX IF NOT EXISTS idx_memories_metadata ON memories USING GIN (metadata)",
    "CREATE INDEX IF NOT EXISTS idx_memories_source ON memories (source)",
    "CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories (created_at DESC)",
    """
    CREATE TABLE IF NOT EXISTS sessions (
        id          TEXT PRIMARY KEY,
        title       TEXT,
        model       TEXT,
        provider    TEXT,
        platform    TEXT,
        turn_count  INT DEFAULT 0,
        token_in    BIGINT DEFAULT 0,
        token_out   BIGINT DEFAULT 0,
        cost        FLOAT DEFAULT 0.0,
        started_at  TIMESTAMPTZ DEFAULT NOW(),
        ended_at    TIMESTAMPTZ,
        tags        TEXT[] DEFAULT '{}'
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions (started_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_sessions_tags ON sessions USING GIN (tags)",
    """
    CREATE TABLE IF NOT EXISTS messages (
        id          BIGSERIAL PRIMARY KEY,
        session_id  TEXT REFERENCES sessions(id) ON DELETE CASCADE,
        turn_index  INT NOT NULL,
        role        TEXT NOT NULL,
        content     TEXT NOT NULL,
        embedding   vector(768),
        tool_name   TEXT,
        tokens      INT,
        created_at  TIMESTAMPTZ DEFAULT NOW()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages (session_id, turn_index)",
    "CREATE INDEX IF NOT EXISTS idx_messages_embedding ON messages USING hnsw (embedding vector_cosine_ops)",
    """
    CREATE TABLE IF NOT EXISTS research (
        id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        query       TEXT NOT NULL,
        url         TEXT,
        title       TEXT,
        content     TEXT NOT NULL,
        summary     TEXT,
        embedding   vector(768),
        tags        TEXT[] DEFAULT '{}',
        source      TEXT,
        session_id  TEXT,
        accessed_at TIMESTAMPTZ DEFAULT NOW(),
        created_at  TIMESTAMPTZ DEFAULT NOW()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_research_embedding ON research USING hnsw (embedding vector_cosine_ops)",
    "CREATE INDEX IF NOT EXISTS idx_research_tags ON research USING GIN (tags)",
    "CREATE INDEX IF NOT EXISTS idx_research_created ON research (created_at DESC)",
    """
    CREATE TABLE IF NOT EXISTS research_chunks (
        id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        research_id     UUID REFERENCES research(id) ON DELETE CASCADE,
        chunk_index     INT NOT NULL,
        content         TEXT NOT NULL,
        embedding       vector(768),
        created_at      TIMESTAMPTZ DEFAULT NOW()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_research_chunks_embedding ON research_chunks USING hnsw (embedding vector_cosine_ops)",
    "CREATE INDEX IF NOT EXISTS idx_research_chunks_parent ON research_chunks (research_id)",
    """
    CREATE TABLE IF NOT EXISTS entities (
        id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        name        TEXT NOT NULL UNIQUE,
        type        TEXT,
        description TEXT,
        metadata    JSONB DEFAULT '{}',
        embedding   vector(768),
        created_at  TIMESTAMPTZ DEFAULT NOW(),
        updated_at  TIMESTAMPTZ DEFAULT NOW()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_entities_embedding ON entities USING hnsw (embedding vector_cosine_ops)",
    "CREATE INDEX IF NOT EXISTS idx_entities_type ON entities (type)",
    """
    CREATE TABLE IF NOT EXISTS entity_relations (
        id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        source_id     UUID REFERENCES entities(id) ON DELETE CASCADE,
        target_id     UUID REFERENCES entities(id) ON DELETE CASCADE,
        relation_type TEXT NOT NULL,
        strength      FLOAT DEFAULT 1.0,
        metadata      JSONB DEFAULT '{}',
        created_at    TIMESTAMPTZ DEFAULT NOW()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_relations_source ON entity_relations (source_id)",
    "CREATE INDEX IF NOT EXISTS idx_relations_target ON entity_relations (target_id)",
    """
    CREATE TABLE IF NOT EXISTS metrics (
        id          BIGSERIAL PRIMARY KEY,
        session_id  TEXT REFERENCES sessions(id) ON DELETE CASCADE,
        turn_index  INT,
        model       TEXT,
        provider    TEXT,
        tokens_in   INT DEFAULT 0,
        tokens_out  INT DEFAULT 0,
        cost        FLOAT DEFAULT 0.0,
        latency_ms  INT,
        tool_calls  INT DEFAULT 0,
        created_at  TIMESTAMPTZ DEFAULT NOW()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_metrics_session ON metrics (session_id)",
    "CREATE INDEX IF NOT EXISTS idx_metrics_created ON metrics (created_at)",
    "CREATE INDEX IF NOT EXISTS idx_metrics_model ON metrics (model)",
    """
    CREATE TABLE IF NOT EXISTS cron_results (
        id          BIGSERIAL PRIMARY KEY,
        job_id      TEXT NOT NULL,
        job_name    TEXT,
        status      TEXT NOT NULL,
        duration_ms INT,
        output_len  INT,
        error       TEXT,
        tokens_used INT,
        created_at  TIMESTAMPTZ DEFAULT NOW()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_cron_job ON cron_results (job_id, created_at DESC)",
    """
    CREATE TABLE IF NOT EXISTS preferences (
        id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        key         TEXT NOT NULL UNIQUE,
        value       JSONB NOT NULL,
        confidence  FLOAT DEFAULT 0.5,
        source      TEXT,
        updated_at  TIMESTAMPTZ DEFAULT NOW(),
        created_at  TIMESTAMPTZ DEFAULT NOW()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_memories_fts ON memories USING GIN (to_tsvector('english', content))",
    "CREATE INDEX IF NOT EXISTS idx_research_fts ON research USING GIN (to_tsvector('english', content || ' ' || COALESCE(title, '')))",
    "CREATE INDEX IF NOT EXISTS idx_messages_fts ON messages USING GIN (to_tsvector('english', content))",
]


class PgConnection:
    """Manages a psycopg2 connection pool to Postgres."""

    def __init__(self, dsn: str):
        self.dsn = dsn
        self._pool: Optional[pg_pool.ThreadedConnectionPool] = None

    def connect(self, minconn: int = 1, maxconn: int = 5) -> None:
        self._pool = pg_pool.ThreadedConnectionPool(minconn, maxconn, self.dsn)
        logger.info("Connected to Postgres: %s", self._mask_dsn(self.dsn))

    def get_conn(self):
        if self._pool is None:
            raise RuntimeError("Not connected to Postgres")
        return self._pool.getconn()

    def put_conn(self, conn) -> None:
        if self._pool is not None:
            self._pool.putconn(conn)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.closeall()
            self._pool = None
            logger.info("Disconnected from Postgres")

    def execute(self, query: str, params: tuple = None) -> None:
        conn = self.get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(query, params)
            conn.commit()
        finally:
            self.put_conn(conn)

    def fetch_one(self, query: str, params: tuple = None) -> Optional[dict]:
        conn = self.get_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                row = cur.fetchone()
                result = dict(row) if row else None
            conn.commit()
            return result
        finally:
            self.put_conn(conn)

    def fetch_all(self, query: str, params: tuple = None) -> list:
        conn = self.get_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
                result = [dict(r) for r in rows]
            conn.commit()
            return result
        finally:
            self.put_conn(conn)

    def ensure_schema(self) -> None:
        """Run DDL idempotently. Each stmt isolated so partial state OK."""
        conn = self.get_conn()
        try:
            for stmt in DDL_STATEMENTS:
                try:
                    with conn.cursor() as cur:
                        cur.execute(stmt)
                    conn.commit()
                except Exception as e:
                    conn.rollback()
                    logger.warning("DDL skipped: %s — %s", stmt[:80].strip(), e)
            logger.info("Schema ensured (version %d)", SCHEMA_VERSION)
        finally:
            self.put_conn(conn)

    @staticmethod
    def _mask_dsn(dsn: str) -> str:
        return re.sub(r'(postgresql://[^:]+:)([^@]+)(@)', r'\1****\3', dsn)

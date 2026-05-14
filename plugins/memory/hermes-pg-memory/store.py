"""CRUD operations for the PG memory plugin."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from .schema import PgConnection
from .embed import Embedder

logger = logging.getLogger(__name__)


def _vec_str(embedding: Optional[List[float]]) -> Optional[str]:
    if not embedding:
        return None
    return "[" + ",".join(str(v) for v in embedding) + "]"


def _iso(row: dict, *keys: str) -> None:
    for k in keys:
        if k in row and isinstance(row[k], datetime):
            row[k] = row[k].isoformat()


class MemoryStore:
    """High-level store operations backed by PG + embeddings."""

    def __init__(self, db: PgConnection, embedder: Embedder):
        self.db = db
        self.embedder = embedder

    # memories

    def store_memory(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        trust_score: float = 0.5,
        source: str = "turn",
        session_id: str = "",
        platform: str = "",
    ) -> Optional[str]:
        embedding_str = _vec_str(self.embedder.embed(content))
        meta_json = json.dumps(metadata or {})
        result = self.db.fetch_one(
            """INSERT INTO memories (content, embedding, metadata, trust_score, source, session_id, platform)
               VALUES (%s, %s::vector, %s::jsonb, %s, %s, %s, %s)
               RETURNING id""",
            (content, embedding_str, meta_json, trust_score,
             source, session_id or None, platform or None),
        )
        return str(result["id"]) if result else None

    def search_memories(
        self,
        query: str,
        limit: int = 5,
        min_trust: float = 0.0,
        source: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        embedding = self.embedder.embed(query)
        if embedding:
            embedding_str = _vec_str(embedding)
            if source:
                rows = self.db.fetch_all(
                    """SELECT id, content, metadata, trust_score, source, session_id,
                              platform, created_at,
                              1 - (embedding <=> %s::vector) AS similarity
                       FROM memories
                       WHERE trust_score >= %s AND source = %s
                       ORDER BY embedding <=> %s::vector
                       LIMIT %s""",
                    (embedding_str, min_trust, source, embedding_str, limit),
                )
            else:
                rows = self.db.fetch_all(
                    """SELECT id, content, metadata, trust_score, source, session_id,
                              platform, created_at,
                              1 - (embedding <=> %s::vector) AS similarity
                       FROM memories
                       WHERE trust_score >= %s
                       ORDER BY embedding <=> %s::vector
                       LIMIT %s""",
                    (embedding_str, min_trust, embedding_str, limit),
                )
        else:
            if source:
                rows = self.db.fetch_all(
                    """SELECT id, content, metadata, trust_score, source, session_id,
                              platform, created_at, 0 AS similarity
                       FROM memories
                       WHERE trust_score >= %s AND source = %s
                         AND to_tsvector('english', content) @@ plainto_tsquery('english', %s)
                       ORDER BY created_at DESC
                       LIMIT %s""",
                    (min_trust, source, query, limit),
                )
            else:
                rows = self.db.fetch_all(
                    """SELECT id, content, metadata, trust_score, source, session_id,
                              platform, created_at, 0 AS similarity
                       FROM memories
                       WHERE trust_score >= %s
                         AND to_tsvector('english', content) @@ plainto_tsquery('english', %s)
                       ORDER BY created_at DESC
                       LIMIT %s""",
                    (min_trust, query, limit),
                )
        for row in rows:
            if row.get("id") is not None:
                row["id"] = str(row["id"])
            _iso(row, "created_at", "accessed_at")
        return rows

    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        row = self.db.fetch_one(
            "UPDATE memories SET accessed_at = NOW() WHERE id = %s RETURNING *",
            (memory_id,),
        )
        if row:
            row["id"] = str(row["id"])
            _iso(row, "created_at", "accessed_at")
        return row

    def delete_memory(self, memory_id: str) -> bool:
        self.db.execute("DELETE FROM memories WHERE id = %s", (memory_id,))
        return True

    # sessions

    def upsert_session(
        self,
        session_id: str,
        title: str = "",
        model: str = "",
        provider: str = "",
        platform: str = "",
        turn_count: int = 0,
        token_in: int = 0,
        token_out: int = 0,
        cost: float = 0.0,
        ended: bool = False,
    ) -> None:
        if ended:
            self.db.execute(
                """INSERT INTO sessions (id, title, model, provider, platform, turn_count,
                                         token_in, token_out, cost, ended_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                   ON CONFLICT (id) DO UPDATE SET
                       turn_count = EXCLUDED.turn_count,
                       token_in = sessions.token_in + EXCLUDED.token_in,
                       token_out = sessions.token_out + EXCLUDED.token_out,
                       cost = sessions.cost + EXCLUDED.cost,
                       ended_at = NOW()""",
                (session_id, title or None, model or None, provider or None,
                 platform or None, turn_count, token_in, token_out, cost),
            )
        else:
            self.db.execute(
                """INSERT INTO sessions (id, title, model, provider, platform, turn_count,
                                         token_in, token_out, cost)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (id) DO UPDATE SET
                       turn_count = EXCLUDED.turn_count,
                       token_in = sessions.token_in + EXCLUDED.token_in,
                       token_out = sessions.token_out + EXCLUDED.token_out,
                       cost = sessions.cost + EXCLUDED.cost""",
                (session_id, title or None, model or None, provider or None,
                 platform or None, turn_count, token_in, token_out, cost),
            )

    # messages

    def store_message(
        self,
        session_id: str,
        turn_index: int,
        role: str,
        content: str,
        tool_name: str = "",
        tokens: int = 0,
    ) -> Optional[int]:
        embedding_str = _vec_str(self.embedder.embed(content))
        result = self.db.fetch_one(
            """INSERT INTO messages (session_id, turn_index, role, content, embedding, tool_name, tokens)
               VALUES (%s, %s, %s, %s, %s::vector, %s, %s)
               RETURNING id""",
            (session_id, turn_index, role, content, embedding_str,
             tool_name or None, tokens),
        )
        return result["id"] if result else None

    # research

    def store_research(
        self,
        query: str,
        content: str,
        url: str = "",
        title: str = "",
        summary: str = "",
        tags: Optional[List[str]] = None,
        source: str = "web_search",
        session_id: str = "",
    ) -> Optional[str]:
        embedding_str = _vec_str(self.embedder.embed(content[:4000]))
        tags_arr = "{" + ",".join(tags or []) + "}"
        result = self.db.fetch_one(
            """INSERT INTO research (query, url, title, content, summary, embedding, tags, source, session_id)
               VALUES (%s, %s, %s, %s, %s, %s::vector, %s, %s, %s)
               RETURNING id""",
            (query, url or None, title or None, content, summary or None,
             embedding_str, tags_arr, source, session_id or None),
        )
        return str(result["id"]) if result else None

    def search_research(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        embedding = self.embedder.embed(query)
        if embedding:
            embedding_str = _vec_str(embedding)
            rows = self.db.fetch_all(
                """SELECT id, query, url, title, content, summary, tags, source,
                          created_at, 1 - (embedding <=> %s::vector) AS similarity
                   FROM research
                   ORDER BY embedding <=> %s::vector
                   LIMIT %s""",
                (embedding_str, embedding_str, limit),
            )
        else:
            rows = self.db.fetch_all(
                """SELECT id, query, url, title, content, summary, tags, source,
                          created_at, 0 AS similarity
                   FROM research
                   WHERE to_tsvector('english', content || ' ' || COALESCE(title, ''))
                         @@ plainto_tsquery('english', %s)
                   ORDER BY created_at DESC
                   LIMIT %s""",
                (query, limit),
            )
        for row in rows:
            row["id"] = str(row["id"])
            _iso(row, "created_at")
        return rows

    # entities

    def upsert_entity(
        self,
        name: str,
        type_: str = "",
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        embed_input = f"{name}: {description}" if description else name
        embedding_str = _vec_str(self.embedder.embed(embed_input))
        meta_json = json.dumps(metadata or {})
        result = self.db.fetch_one(
            """INSERT INTO entities (name, type, description, metadata, embedding)
               VALUES (%s, %s, %s, %s::jsonb, %s::vector)
               ON CONFLICT (name) DO UPDATE SET
                   type = EXCLUDED.type,
                   description = EXCLUDED.description,
                   metadata = EXCLUDED.metadata,
                   embedding = EXCLUDED.embedding,
                   updated_at = NOW()
               RETURNING id""",
            (name, type_ or None, description or None, meta_json, embedding_str),
        )
        return str(result["id"]) if result else ""

    def relate_entities(
        self,
        source_name: str,
        target_name: str,
        relation_type: str,
        strength: float = 1.0,
    ) -> None:
        self.db.execute(
            """INSERT INTO entity_relations (source_id, target_id, relation_type, strength)
               SELECT s.id, t.id, %s, %s
               FROM entities s, entities t
               WHERE s.name = %s AND t.name = %s
               ON CONFLICT DO NOTHING""",
            (relation_type, strength, source_name, target_name),
        )

    def search_entities(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        embedding = self.embedder.embed(query)
        if embedding:
            embedding_str = _vec_str(embedding)
            rows = self.db.fetch_all(
                """SELECT id, name, type, description, metadata, updated_at,
                          1 - (embedding <=> %s::vector) AS similarity
                   FROM entities
                   ORDER BY embedding <=> %s::vector
                   LIMIT %s""",
                (embedding_str, embedding_str, limit),
            )
        else:
            rows = self.db.fetch_all(
                """SELECT id, name, type, description, metadata, updated_at, 0 AS similarity
                   FROM entities
                   WHERE to_tsvector('english', name || ' ' || COALESCE(description, ''))
                         @@ plainto_tsquery('english', %s)
                   LIMIT %s""",
                (query, limit),
            )
        for row in rows:
            row["id"] = str(row["id"])
            _iso(row, "updated_at")
        return rows

    # metrics

    def store_metric(
        self,
        session_id: str,
        turn_index: int,
        model: str = "",
        provider: str = "",
        tokens_in: int = 0,
        tokens_out: int = 0,
        cost: float = 0.0,
        latency_ms: int = 0,
        tool_calls: int = 0,
    ) -> None:
        self.db.execute(
            """INSERT INTO metrics (session_id, turn_index, model, provider,
                                    tokens_in, tokens_out, cost, latency_ms, tool_calls)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (session_id, turn_index, model or None, provider or None,
             tokens_in, tokens_out, cost, latency_ms, tool_calls),
        )

    def query_metrics(
        self,
        metric: str = "tokens_in",
        group_by: str = "model",
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        allowed_metrics = {"tokens_in", "tokens_out", "cost", "tool_calls", "latency_ms"}
        allowed_groups = {"model", "provider", "session_id", "DATE(created_at)"}
        if metric not in allowed_metrics:
            return [{"error": f"Invalid metric '{metric}'. Allowed: {sorted(allowed_metrics)}"}]
        if group_by not in allowed_groups:
            return [{"error": f"Invalid group_by '{group_by}'. Allowed: {sorted(allowed_groups)}"}]
        # whitelist-checked → safe to interpolate
        return self.db.fetch_all(
            f"SELECT {group_by} AS key, SUM({metric}) AS total, COUNT(*) AS n FROM metrics "
            f"GROUP BY {group_by} ORDER BY total DESC LIMIT %s",
            (limit,),
        )

    # cron

    def store_cron_result(
        self,
        job_id: str,
        status: str,
        job_name: str = "",
        duration_ms: int = 0,
        output_len: int = 0,
        error: str = "",
        tokens_used: int = 0,
    ) -> None:
        self.db.execute(
            """INSERT INTO cron_results (job_id, job_name, status, duration_ms,
                                         output_len, error, tokens_used)
               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            (job_id, job_name or None, status, duration_ms, output_len,
             error or None, tokens_used),
        )

    # preferences

    def set_preference(
        self,
        key: str,
        value: Any,
        confidence: float = 0.5,
        source: str = "",
    ) -> None:
        val_json = json.dumps(value)
        self.db.execute(
            """INSERT INTO preferences (key, value, confidence, source)
               VALUES (%s, %s::jsonb, %s, %s)
               ON CONFLICT (key) DO UPDATE SET
                   value = EXCLUDED.value,
                   confidence = EXCLUDED.confidence,
                   source = EXCLUDED.source,
                   updated_at = NOW()""",
            (key, val_json, confidence, source or None),
        )

    def get_preference(self, key: str) -> Optional[Any]:
        row = self.db.fetch_one("SELECT value FROM preferences WHERE key = %s", (key,))
        return row["value"] if row else None

    # raw query (sandboxed)

    def query_raw(self, sql: str, params: tuple = None, limit: int = 50) -> List[Dict[str, Any]]:
        sql_stripped = sql.strip().upper()
        if not sql_stripped.startswith("SELECT"):
            return [{"error": "Only SELECT queries are allowed"}]
        if "LIMIT" not in sql_stripped:
            sql = sql.rstrip(";") + f" LIMIT {limit}"
        return self.db.fetch_all(sql, params)

    # unified

    def unified_search(self, query: str, limit_per_source: int = 3) -> Dict[str, List[Dict]]:
        return {
            "memories": self.search_memories(query, limit=limit_per_source),
            "research": self.search_research(query, limit=limit_per_source),
            "entities": self.search_entities(query, limit=limit_per_source),
        }

    # stats

    def get_stats(self) -> Dict[str, Any]:
        counts = self.db.fetch_all(
            """SELECT 'memories' AS tbl, COUNT(*) AS n FROM memories
               UNION ALL SELECT 'messages', COUNT(*) FROM messages
               UNION ALL SELECT 'research', COUNT(*) FROM research
               UNION ALL SELECT 'entities', COUNT(*) FROM entities
               UNION ALL SELECT 'metrics', COUNT(*) FROM metrics
               UNION ALL SELECT 'cron_results', COUNT(*) FROM cron_results"""
        )
        stat_map = {r["tbl"]: int(r["n"]) for r in counts}
        stat_map["total"] = sum(stat_map.values())
        return stat_map

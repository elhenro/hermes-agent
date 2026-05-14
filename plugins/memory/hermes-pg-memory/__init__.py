"""Hermes PG Memory Provider — Postgres-backed memory with pgvector.

Connects to a local Postgres instance with pgvector + Ollama embeddings.
Exposes semantic memory, research, entity graph, and analytics as tools.

Config (in order of precedence):
  1. $HERMES_HOME/hermes-pg-memory.json (per-key override)
  2. Environment variables:
       HERMES_PG_DSN           postgresql://hermes:hermes@localhost:5432/hermes
       HERMES_PG_EMBED_MODEL   nomic-embed-text
       OLLAMA_BASE_URL         http://localhost:11434
       HERMES_PG_EMBED_DIMS    768
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

from .schema import PgConnection
from .store import MemoryStore
from .embed import Embedder

logger = logging.getLogger(__name__)


def _load_config() -> dict:
    """Load config from env vars + $HERMES_HOME/hermes-pg-memory.json."""
    from hermes_constants import get_hermes_home

    config = {
        "dsn": os.environ.get(
            "HERMES_PG_DSN",
            "postgresql://hermes:hermes@localhost:5432/hermes",
        ),
        "embedding_model": os.environ.get("HERMES_PG_EMBED_MODEL", "nomic-embed-text"),
        "ollama_base_url": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        "embedding_dims": int(os.environ.get("HERMES_PG_EMBED_DIMS", "768")),
    }

    config_path = get_hermes_home() / "hermes-pg-memory.json"
    if config_path.exists():
        try:
            file_cfg = json.loads(config_path.read_text(encoding="utf-8"))
            config.update({k: v for k, v in file_cfg.items() if v not in (None, "")})
        except Exception as e:
            logger.warning("Failed to load %s: %s", config_path, e)

    return config


# tool schemas

TOOL_MEMORY_STORE = {
    "name": "memory_store",
    "description": (
        "Store a fact, memory, or piece of information for future recall. "
        "Use when the user says something worth remembering, or when you "
        "discover something during research. Stored with vector embedding "
        "for semantic search."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The fact or memory to store. Be specific and self-contained."},
            "metadata": {"type": "object", "description": "Optional structured metadata (tags, category, etc.)"},
            "trust_score": {"type": "number", "description": "Confidence in this memory (0.0-1.0). Default 0.5", "default": 0.5},
            "source": {"type": "string", "description": "Source type: 'turn', 'explicit', 'research'", "default": "explicit"},
        },
        "required": ["content"],
    },
}

TOOL_MEMORY_SEARCH = {
    "name": "memory_search",
    "description": (
        "Search stored memories using semantic (vector) search. "
        "Returns memories ranked by relevance. Use to recall what "
        "you know about a topic, person, or project."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for — natural language query"},
            "limit": {"type": "integer", "description": "Max results (default 5)", "default": 5},
            "source": {"type": "string", "description": "Filter by source: 'turn', 'explicit', 'research'"},
        },
        "required": ["query"],
    },
}

TOOL_RESEARCH_STORE = {
    "name": "research_store",
    "description": (
        "Save a web research finding to the research database. "
        "Use after a web_search to persist the result with vector "
        "embedding for future retrieval. Include the original query "
        "so we can trace provenance."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query that produced this finding"},
            "content": {"type": "string", "description": "The full content or text of the finding"},
            "url": {"type": "string", "description": "Source URL"},
            "title": {"type": "string", "description": "Page title or source name"},
            "summary": {"type": "string", "description": "Brief summary of the finding"},
            "tags": {"type": "array", "items": {"type": "string"}, "description": "Tags for categorization"},
        },
        "required": ["query", "content"],
    },
}

TOOL_RESEARCH_SEARCH = {
    "name": "research_search",
    "description": (
        "Search past research findings using semantic search. "
        "Use when you need to recall something you found during "
        "a previous research session."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for"},
            "limit": {"type": "integer", "description": "Max results (default 5)", "default": 5},
        },
        "required": ["query"],
    },
}

TOOL_ENTITY_STORE = {
    "name": "entity_store",
    "description": (
        "Create or update an entity (person, project, concept, tool). "
        "Use to build a knowledge graph of named things and their "
        "relationships. Entities are searchable by semantic similarity."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Entity name (unique identifier)"},
            "type": {"type": "string", "description": "Entity type: person, project, concept, tool, framework, library"},
            "description": {"type": "string", "description": "Description of the entity"},
        },
        "required": ["name"],
    },
}

TOOL_ENTITY_RELATE = {
    "name": "entity_relate",
    "description": "Create a relationship between two existing entities.",
    "parameters": {
        "type": "object",
        "properties": {
            "source": {"type": "string", "description": "Source entity name"},
            "target": {"type": "string", "description": "Target entity name"},
            "relation": {"type": "string", "description": "Relation type: depends_on, built_with, knows, uses, part_of, related_to"},
        },
        "required": ["source", "target", "relation"],
    },
}

TOOL_KNOWLEDGE_TELL = {
    "name": "knowledge_tell",
    "description": (
        "Unified search across memories, research, and entities. "
        "Use when you want to know everything the system remembers "
        "about a topic."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for"},
            "limit": {"type": "integer", "description": "Results per category (default 3)", "default": 3},
        },
        "required": ["query"],
    },
}

TOOL_METRICS_QUERY = {
    "name": "metrics_query",
    "description": (
        "Query aggregated usage metrics. Use to answer questions "
        "about token usage, cost, model performance over time."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "metric": {
                "type": "string",
                "enum": ["tokens_in", "tokens_out", "cost", "tool_calls", "latency_ms"],
                "description": "Metric to aggregate",
            },
            "group_by": {
                "type": "string",
                "enum": ["model", "provider", "session_id"],
                "description": "How to group results",
            },
            "limit": {"type": "integer", "description": "Max results (default 10)", "default": 10},
        },
        "required": ["metric", "group_by"],
    },
}

TOOL_PG_STATS = {
    "name": "pg_stats",
    "description": "Get storage statistics for all PG-backed tables (counts per table + total).",
    "parameters": {"type": "object", "properties": {}},
}


# provider

class HermesPgMemoryProvider(MemoryProvider):
    """Postgres-backed memory with pgvector semantic search."""

    def __init__(self, config: dict):
        self._config = config
        self._dsn = config.get("dsn")
        self._embed_model = config.get("embedding_model", "nomic-embed-text")
        self._ollama_url = config.get("ollama_base_url", "http://localhost:11434")
        self._embed_dims = int(config.get("embedding_dims", 768))

        self._db: Optional[PgConnection] = None
        self._embedder: Optional[Embedder] = None
        self._store: Optional[MemoryStore] = None
        self._current_session_id: str = ""

    @property
    def name(self) -> str:
        return "hermes-pg-memory"

    def is_available(self) -> bool:
        """Check PG reachability + Ollama responding. Auto-starts Docker if PG down."""
        import time

        def _try_connect(timeout: int = 3) -> bool:
            try:
                import psycopg2
                import requests
                conn = psycopg2.connect(self._dsn, connect_timeout=timeout)
                conn.close()
                resp = requests.get(f"{self._ollama_url}/api/tags", timeout=5)
                return resp.status_code == 200
            except Exception:
                return False

        # Fast path — already running
        if _try_connect(3):
            return True

        # Slow path — try Docker auto-start
        try:
            import shutil
            import subprocess

            compose_file = os.path.expanduser("~/hermes-pg/docker-compose.yml")
            if not os.path.exists(compose_file):
                logger.debug("No docker-compose.yml found at %s", compose_file)
                return False

            # Resolve docker binary (Docker Desktop on macOS isn't always on PATH).
            docker_bin = shutil.which("docker") or (
                "/Applications/Docker.app/Contents/Resources/bin/docker"
                if os.path.exists("/Applications/Docker.app/Contents/Resources/bin/docker")
                else None
            )
            if not docker_bin:
                logger.debug("docker binary not found")
                return False

            # Daemon up? If not, bail before wasting 30s on compose.
            info = subprocess.run(
                [docker_bin, "info"], capture_output=True, timeout=5,
            )
            if info.returncode != 0:
                logger.debug("docker daemon not running: %s", info.stderr[:200])
                return False

            # Start PG container
            logger.info("PG unreachable — attempting Docker auto-start")
            result = subprocess.run(
                [docker_bin, "compose", "-f", compose_file, "up", "-d"],
                capture_output=True, timeout=30,
            )
            if result.returncode != 0:
                logger.debug("Docker compose up failed: %s", result.stderr.decode()[:200])
                return False

            # Wait up to 20s for PG to become healthy
            for _ in range(10):
                time.sleep(2)
                if _try_connect(2):
                    logger.info("PG auto-started via Docker")
                    return True

            logger.debug("PG container started but not accepting connections within 20s")
            return False
        except Exception as e:
            logger.debug("Docker auto-start failed: %s", e)
            return False

    def initialize(self, session_id: str, **kwargs) -> None:
        self._current_session_id = session_id or ""

        self._db = PgConnection(self._dsn)
        self._db.connect()
        self._db.ensure_schema()

        self._embedder = Embedder(
            model=self._embed_model,
            base_url=self._ollama_url,
            dims=self._embed_dims,
        )
        self._store = MemoryStore(self._db, self._embedder)

        # ensure session row exists so FK-bound writes don't fail
        if self._current_session_id:
            try:
                platform = kwargs.get("platform", "") or ""
                self._store.upsert_session(
                    session_id=self._current_session_id,
                    platform=platform,
                )
            except Exception as e:
                logger.debug("session upsert at init failed: %s", e)

        logger.info("PG memory provider initialized for session %s", session_id)

    def system_prompt_block(self) -> str:
        return (
            "You have access to a Postgres-backed memory system with pgvector semantic search. "
            "Use `memory_store` to save important facts, `memory_search` to recall them. "
            "Use `research_store` and `research_search` for web research findings. "
            "Use `entity_store` and `entity_relate` to build a knowledge graph. "
            "Use `knowledge_tell` for unified search across all data. "
            "Use `metrics_query` to check token usage and costs."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._store:
            return ""
        try:
            results = self._store.unified_search(query, limit_per_source=3)
            parts = []
            if results.get("memories"):
                parts.append("Relevant memories:")
                for m in results["memories"]:
                    parts.append(f"- {m['content']} (trust: {m.get('trust_score', 0.0):.1f})")
            if results.get("research"):
                parts.append("Relevant research:")
                for r in results["research"]:
                    title = r.get("title") or r.get("query") or ""
                    snippet = (r.get("content") or "")[:200]
                    parts.append(f"- {title}: {snippet}...")
            if results.get("entities"):
                parts.append("Relevant entities:")
                for e in results["entities"]:
                    parts.append(f"- {e['name']} ({e.get('type') or ''})")
            return "\n".join(parts) if parts else ""
        except Exception as e:
            logger.warning("Prefetch failed: %s", e)
            return ""

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
    ) -> None:
        if not self._store:
            return
        sid = session_id or self._current_session_id
        if not sid:
            return
        try:
            self._store.upsert_session(session_id=sid)
            self._store.store_memory(
                content=user_content,
                source="turn",
                session_id=sid,
                metadata={"role": "user"},
            )
            self._store.store_memory(
                content=assistant_content[:1000],
                source="turn",
                session_id=sid,
                metadata={"role": "assistant"},
            )
        except Exception as e:
            logger.warning("sync_turn failed: %s", e)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            TOOL_MEMORY_STORE,
            TOOL_MEMORY_SEARCH,
            TOOL_RESEARCH_STORE,
            TOOL_RESEARCH_SEARCH,
            TOOL_ENTITY_STORE,
            TOOL_ENTITY_RELATE,
            TOOL_KNOWLEDGE_TELL,
            TOOL_METRICS_QUERY,
            TOOL_PG_STATS,
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if not self._store:
            return tool_error("PG memory provider not initialized")
        try:
            result = self._dispatch(tool_name, args or {})
            return json.dumps(result, default=str)
        except KeyError as e:
            return tool_error(f"Missing required arg: {e}")
        except Exception as e:
            logger.error("Tool call '%s' failed: %s", tool_name, e)
            return tool_error(f"PG memory error: {e}")

    def _dispatch(self, tool_name: str, args: dict) -> Any:
        store = self._store
        sid = self._current_session_id
        dispatch = {
            "memory_store": lambda: store.store_memory(
                content=args["content"],
                metadata=args.get("metadata"),
                trust_score=args.get("trust_score", 0.5),
                source=args.get("source", "explicit"),
                session_id=sid,
            ),
            "memory_search": lambda: store.search_memories(
                query=args["query"],
                limit=args.get("limit", 5),
                source=args.get("source"),
            ),
            "research_store": lambda: store.store_research(
                query=args["query"],
                content=args["content"],
                url=args.get("url", ""),
                title=args.get("title", ""),
                summary=args.get("summary", ""),
                tags=args.get("tags"),
                session_id=sid,
            ),
            "research_search": lambda: store.search_research(
                query=args["query"],
                limit=args.get("limit", 5),
            ),
            "entity_store": lambda: store.upsert_entity(
                name=args["name"],
                type_=args.get("type", ""),
                description=args.get("description", ""),
            ),
            "entity_relate": lambda: store.relate_entities(
                source_name=args["source"],
                target_name=args["target"],
                relation_type=args["relation"],
            ),
            "knowledge_tell": lambda: store.unified_search(
                query=args["query"],
                limit_per_source=args.get("limit", 3),
            ),
            "metrics_query": lambda: store.query_metrics(
                metric=args["metric"],
                group_by=args["group_by"],
                limit=args.get("limit", 10),
            ),
            "pg_stats": lambda: store.get_stats(),
        }
        handler = dispatch.get(tool_name)
        if not handler:
            raise ValueError(f"Unknown tool: {tool_name}")
        return handler()

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if not self._store or not self._current_session_id:
            return
        try:
            turn_count = sum(1 for m in messages if m.get("role") == "user")
            self._store.db.execute(
                "UPDATE sessions SET ended_at = NOW(), turn_count = %s WHERE id = %s",
                (turn_count, self._current_session_id),
            )
        except Exception as e:
            logger.warning("on_session_end failed: %s", e)

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self._store or action != "add":
            return
        try:
            md = dict(metadata or {})
            sid = md.get("session_id") or self._current_session_id
            self._store.store_memory(
                content=content,
                metadata={"target": target, **md},
                source="memory_write",
                session_id=sid or "",
            )
        except Exception as e:
            logger.debug("on_memory_write mirror failed: %s", e)

    def on_delegation(self, task: str, result: str, **kwargs) -> None:
        if not self._store:
            return
        try:
            result_str = str(result)[:500] if result else ""
            self._store.store_memory(
                content=f"Subagent task: {task[:300]}\nResult: {result_str}",
                source="delegation",
                metadata={"subagent": True},
                session_id=self._current_session_id,
            )
        except Exception as e:
            logger.debug("on_delegation store failed: %s", e)

    def shutdown(self) -> None:
        if self._db:
            try:
                self._db.close()
            except Exception as e:
                logger.debug("shutdown close failed: %s", e)


# entry point

def register(ctx) -> None:
    """Register the PG memory provider with the Hermes plugin system."""
    ctx.register_memory_provider(HermesPgMemoryProvider(config=_load_config()))

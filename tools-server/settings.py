"""Environment-driven configuration for custom-tools server."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Brave Search (primary web_search provider)
    brave_api_key: str = Field(default="", alias="BRAVE_API_KEY")
    brave_cache_ttl_hours: int = Field(default=24, alias="BRAVE_CACHE_TTL_HOURS")

    # SearXNG (self-hosted meta-search; web_search fallback when Brave is
    # quota-exhausted/rate-limited). Empty → no fallback. Point at the JSON API
    # of a SearXNG instance on the LAN, e.g. http://192.168.1.11:8088
    searxng_url: str = Field(default="", alias="SEARXNG_URL")

    # Alternate web_search between Brave and SearXNG per query (halves Brave
    # quota, decorrelates the two backends so a bad window on one doesn't sink
    # every worker on a request). Query-hash picks the PRIMARY; each backend
    # still cross-falls-back to the other on failure. When off, or when
    # SEARXNG_URL is unset, web_search is Brave-primary with SearXNG only as a
    # failure fallback (the pre-2026-07-09 behavior). Flip + `up -d
    # --force-recreate custom-tools` to apply.
    web_search_alternate: bool = Field(default=True, alias="WEB_SEARCH_ALTERNATE")

    # web_fetch (page-opener). Bounds on a model-steerable HTTP client: an overall
    # wall-clock deadline for the whole call — redirects + reads + extraction — and
    # a cap on concurrent in-flight fetches. The deadline must stay UNDER Audrey's
    # 30s tool-dispatch ceiling (`graph.DEFAULT_DISPATCH_TIMEOUT_S`) so a slow or
    # redirect-chained page reports "deadline" instead of surfacing as a bare
    # dispatch timeout with an orphaned coroutine; the per-op `_FETCH_TIMEOUT_S`
    # resets on each hop and can't do that alone. The concurrency cap is acquired
    # INSIDE the deadline, so a saturated pool makes further callers wait it out and
    # fail as a clean timeout, bounding sockets + memory under a burst. Change
    # either + `up -d --force-recreate custom-tools` to apply.
    web_fetch_overall_deadline_s: float = Field(default=25.0, alias="WEB_FETCH_OVERALL_DEADLINE_S")
    web_fetch_max_concurrent: int = Field(default=8, alias="WEB_FETCH_MAX_CONCURRENT")

    # Audrey (for kb_search / kb_image_search proxying)
    audrey_url: str = Field(default="http://audrey-ai:8000", alias="AUDREY_URL")
    # Middle rung of the KB timeout ladder. Must sit BELOW Audrey's tool-dispatch
    # timeout (`graph.DEFAULT_DISPATCH_TIMEOUT_S`, 30s) and ABOVE the embed budget
    # inside /v1/kb/query (`TextEmbedder.query_timeout_s`, 24s). Previously 30.0 —
    # tied with its own caller, so the 502 raised at app.py `kb_search` could never
    # win the race and every slow KB query reached the model as a bare timeout.
    audrey_kb_timeout_seconds: float = Field(default=27.0, alias="AUDREY_KB_TIMEOUT_SECONDS")
    # Shared secret presented to Audrey's gated KB query routes (Phase 31), sent in
    # the `X-Audrey-Service-Token` header on the audrey client so kb_search /
    # kb_image_search act on behalf of the pipeline user. Empty → header omitted
    # (dev/local); Audrey then requires a user bearer instead. Must match Audrey's
    # `KB_SERVICE_TOKEN`. Change + `up -d --force-recreate custom-tools` to apply.
    kb_service_token: str = Field(default="", alias="KB_SERVICE_TOKEN")

    # Local storage
    data_dir: Path = Field(default=Path("/app/data"), alias="TOOLS_DATA_DIR")

    # Memory (Qdrant-backed, semantic search via nomic-embed-text)
    qdrant_url: str = Field(default="http://qdrant:6333", alias="QDRANT_URL")
    memory_collection: str = Field(default="kb_memory", alias="MEMORY_COLLECTION")
    memory_embed_model: str = Field(default="nomic-embed-text", alias="MEMORY_EMBED_MODEL")
    memory_embed_dim: int = Field(default=768, alias="MEMORY_EMBED_DIM")
    # Cosine similarity floor. `memory_search` drops anything below this.
    # Tighter than KB because memory false-positives poison the prompt as
    # "facts about the user" — a false hit actively misleads the model.
    memory_similarity_threshold: float = Field(default=0.5, alias="MEMORY_SIMILARITY_THRESHOLD")

    # Ollama URL for nomic-embed-text calls
    ollama_url: str = Field(default="http://ollama:11434", alias="OLLAMA_URL")
    ollama_embed_timeout_s: float = Field(default=10.0, alias="OLLAMA_EMBED_TIMEOUT_S")

    # Chat archive (per-user searchable conversation history).
    # SQLite is the source of truth; Qdrant indexes Q+A-pair chunks for
    # semantic search. Reuses the same embedder as durable memory so a
    # missing Ollama only breaks one subsystem at a time.
    chat_archive_collection: str = Field(default="kb_chat_archive", alias="CHAT_ARCHIVE_COLLECTION")
    chat_archive_chunk_max_chars: int = Field(default=2500, alias="CHAT_ARCHIVE_CHUNK_MAX_CHARS")
    chat_archive_chunk_overlap_chars: int = Field(default=100, alias="CHAT_ARCHIVE_CHUNK_OVERLAP_CHARS")
    chat_archive_search_threshold: float = Field(default=0.4, alias="CHAT_ARCHIVE_SEARCH_THRESHOLD")
    # 0 means "keep forever" / "no cap". Wired in from day one so a later
    # retention policy doesn't need a schema migration.
    chat_archive_retention_days: int = Field(default=0, alias="CHAT_ARCHIVE_RETENTION_DAYS")
    chat_archive_max_bytes: int = Field(default=0, alias="CHAT_ARCHIVE_MAX_BYTES")

    @model_validator(mode="after")
    def _check_chunk_overlap(self) -> Settings:
        """Overlap must be strictly less than the chunk cap.

        `_split_long`'s hard-split path steps by `max_chars - overlap`; an
        overlap >= max_chars makes that step <= 0 and crashes archive writes
        at runtime. Fail fast at boot with a clear message instead.
        """
        if self.chat_archive_chunk_overlap_chars >= self.chat_archive_chunk_max_chars:
            raise ValueError(
                "CHAT_ARCHIVE_CHUNK_OVERLAP_CHARS "
                f"({self.chat_archive_chunk_overlap_chars}) must be less than "
                "CHAT_ARCHIVE_CHUNK_MAX_CHARS "
                f"({self.chat_archive_chunk_max_chars})"
            )
        return self

    @property
    def chat_archive_db_path(self) -> Path:
        """SQLite source-of-truth for the chat archive."""
        return self.data_dir / "chat_archive.db"

    @property
    def memory_db_path(self) -> Path:
        """Legacy SQLite path. Used only for one-shot migration on startup.

        Once migrated, the file is renamed to `memory.db.migrated` and Qdrant
        becomes the authoritative store.
        """
        return self.data_dir / "memory.db"


settings = Settings()

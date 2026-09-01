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

    # One page of `get_file_text`. **Must stay under Audrey's
    # `agentic.react.max_tool_result_chars`** (raised 2000 → 6000 on
    # 2026-08-05). The artifact route ends a page on a line boundary precisely
    # so a transcript is not cut mid-sentence; a page bigger than the
    # dispatcher's cap is then cut mid-word on arrival regardless, which is the
    # failure the paging exists to prevent. 4000 leaves room for the JSON
    # envelope around the text.
    file_text_page_chars: int = Field(default=4000, alias="FILE_TEXT_PAGE_CHARS")

    # Audrey (for kb_search / kb_image_search proxying)
    audrey_url: str = Field(default="http://audrey-ai:8000", alias="AUDREY_URL")
    # Middle rung of the KB timeout ladder. Must sit BELOW Audrey's tool-dispatch
    # timeout (`graph.DEFAULT_DISPATCH_TIMEOUT_S`, 30s) and ABOVE the embed budget
    # inside /v1/kb/query (`TextEmbedder.query_timeout_s`, 24s). Previously 30.0 —
    # tied with its own caller, so the 502 raised at app.py `kb_search` could never
    # win the race and every slow KB query reached the model as a bare timeout.
    audrey_kb_timeout_seconds: float = Field(default=27.0, alias="AUDREY_KB_TIMEOUT_SECONDS")
    kb_service_token: str = Field(default="", alias="KB_SERVICE_TOKEN")

    # Local storage
    data_dir: Path = Field(default=Path("/app/data"), alias="TOOLS_DATA_DIR")

    # Memory (Qdrant-backed, semantic search via nomic-embed-text)
    qdrant_url: str = Field(default="http://qdrant:6333", alias="QDRANT_URL")
    capability_probe_timeout_s: float = Field(
        default=2.0, ge=0.1, alias="CAPABILITY_PROBE_TIMEOUT_S",
    )
    capability_retry_interval_s: float = Field(
        default=5.0, ge=0.1, alias="CAPABILITY_RETRY_INTERVAL_S",
    )
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
    # Memory auto-recall runs on the hot path of EVERY request, and Audrey
    # gives it a 5s deadline (`agentic.memory.timeout_s`). That is the tightest
    # outer budget in the system, so the memory embed needs its own rung BELOW
    # it — sharing the 10s general embed budget inverted the ladder, and the
    # outer always won: custom-tools kept embedding for another 5s after Audrey
    # had given up, and every stall reached the logs as a bare `timeout in
    # 5.00s` with no cause attached. Same failure shape as the KB ladder
    # (`test_kb_timeout_ladder.py`), same root cause on this box — the embedder
    # being evicted from VRAM by a local panel worker. A memory embed is one
    # short string; warm it is ~100ms, so anything past 4s is an eviction
    # stall, and on the hot path failing fast beats waiting.
    memory_embed_timeout_s: float = Field(default=4.0, alias="MEMORY_EMBED_TIMEOUT_S")
    # How long Ollama holds the embedder in VRAM after a call. Its default is 5
    # minutes — shorter than the gap between bursts of chat on a personal box,
    # so the embedder was cold on effectively every recall and a cold load
    # (4.18s measured 2026-08-10) does not fit the 4s budget above. It never
    # could: `nomic-embed-text` is 323 MB and residency is close to free, so
    # the right fix is to stop it being evicted rather than to widen the
    # budget. Warm, the same call takes 0.059s.
    # ⚠️ Occupies one `OLLAMA_MAX_LOADED_MODELS` slot (2 on this box).
    # Empty string sends no field and restores Ollama's default.
    embed_keep_alive: str = Field(default="24h", alias="EMBED_KEEP_ALIVE")

    # Chat archive (per-user searchable conversation history).
    # SQLite is the source of truth; Qdrant indexes Q+A-pair chunks for
    # semantic search. Reuses the same embedder as durable memory so a
    # missing Ollama only breaks one subsystem at a time.
    chat_archive_collection: str = Field(default="kb_chat_archive", alias="CHAT_ARCHIVE_COLLECTION")
    chat_archive_chunk_max_chars: int = Field(default=2500, alias="CHAT_ARCHIVE_CHUNK_MAX_CHARS")
    chat_archive_chunk_overlap_chars: int = Field(default=100, alias="CHAT_ARCHIVE_CHUNK_OVERLAP_CHARS")
    chat_archive_search_threshold: float = Field(default=0.4, alias="CHAT_ARCHIVE_SEARCH_THRESHOLD")
    # 0 retention means "keep forever". Maintenance still runs so failed
    # index writes and already-queued deletions recover after restart.
    chat_archive_retention_days: int = Field(default=0, alias="CHAT_ARCHIVE_RETENTION_DAYS")
    chat_archive_max_bytes: int = Field(default=0, alias="CHAT_ARCHIVE_MAX_BYTES")
    chat_archive_maintenance_interval_s: float = Field(
        default=300.0, ge=0.0, alias="CHAT_ARCHIVE_MAINTENANCE_INTERVAL_S",
    )
    chat_archive_repair_batch_size: int = Field(
        default=50, ge=1, le=500, alias="CHAT_ARCHIVE_REPAIR_BATCH_SIZE",
    )
    chat_archive_max_retry_attempts: int = Field(
        default=5, ge=1, le=100, alias="CHAT_ARCHIVE_MAX_RETRY_ATTEMPTS",
    )

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
        if self.chat_archive_max_bytes:
            raise ValueError(
                "CHAT_ARCHIVE_MAX_BYTES is not implemented; leave it at 0 "
                "instead of configuring an unenforced archive cap"
            )
        return self

    @property
    def chat_archive_db_path(self) -> Path:
        """SQLite source-of-truth for the chat archive."""
        return self.data_dir / "chat_archive.db"

    @property
    def memory_db_path(self) -> Path:
        """Legacy SQLite path. Used only for one-shot migration on startup.

        Only after every row migrates, the file is renamed to
        `memory.db.migrated` and Qdrant becomes the authoritative store.
        """
        return self.data_dir / "memory.db"


settings = Settings()

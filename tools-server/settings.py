"""Environment-driven configuration for custom-tools server."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Brave Search
    brave_api_key: str = Field(default="", alias="BRAVE_API_KEY")
    brave_cache_ttl_hours: int = Field(default=24, alias="BRAVE_CACHE_TTL_HOURS")

    # Audrey (for kb_search / kb_image_search proxying)
    audrey_url: str = Field(default="http://audrey-ai:8000", alias="AUDREY_URL")
    audrey_kb_timeout_seconds: float = Field(default=30.0, alias="AUDREY_KB_TIMEOUT_SECONDS")

    # Local storage
    data_dir: Path = Field(default=Path("/app/data"), alias="TOOLS_DATA_DIR")

    # Memory (Qdrant-backed, semantic; Phase 12)
    qdrant_url: str = Field(default="http://qdrant:6333", alias="QDRANT_URL")
    memory_collection: str = Field(default="kb_memory", alias="MEMORY_COLLECTION")
    memory_embed_model: str = Field(default="nomic-embed-text", alias="MEMORY_EMBED_MODEL")
    memory_embed_dim: int = Field(default=768, alias="MEMORY_EMBED_DIM")
    # Cosine similarity floor. `memory_search` drops anything below this.
    # Tighter than KB because memory false-positives poison the prompt as
    # "facts about the user" — a false hit actively misleads the model.
    memory_similarity_threshold: float = Field(default=0.4, alias="MEMORY_SIMILARITY_THRESHOLD")

    # Ollama URL for nomic-embed-text calls
    ollama_url: str = Field(default="http://ollama:11434", alias="OLLAMA_URL")
    ollama_embed_timeout_s: float = Field(default=10.0, alias="OLLAMA_EMBED_TIMEOUT_S")

    @property
    def memory_db_path(self) -> Path:
        """Legacy SQLite path. Used only for one-shot migration on startup.

        Once migrated, the file is renamed to `memory.db.migrated` and Qdrant
        becomes the authoritative store.
        """
        return self.data_dir / "memory.db"


settings = Settings()

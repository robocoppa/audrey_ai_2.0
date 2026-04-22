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

    @property
    def memory_db_path(self) -> Path:
        return self.data_dir / "memory.db"


settings = Settings()

"""Application configuration using pydantic-settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="FAULTTRACE_", extra="ignore")

    # Paths
    data_root: Path = Path("data")
    artifacts_root: Path = Path("artifacts")
    db_path: Path = Path("data/faulttrace.db")

    # API
    cors_origins: str = "http://localhost:3000"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False
    log_level: str = "INFO"

    # Database
    database_url: str = ""  # empty = use SQLite at db_path

    # Demo defaults
    demo_seed: int = 42
    demo_scales: str = "10,50,200,1000"

    @property
    def effective_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        return f"sqlite:///{self.db_path}"


@lru_cache
def get_settings() -> Settings:
    return Settings()

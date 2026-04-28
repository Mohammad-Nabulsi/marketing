from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_env: str = "dev"
    frontend_origin: str = "http://localhost:5173"
    storage_dir: str = "storage"

    def storage_path(self) -> Path:
        return Path(self.storage_dir).resolve()


settings = Settings()


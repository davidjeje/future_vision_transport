from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Segmentation API"
    app_version: str = "1.0.0"

    # mock = test sans vrai modèle
    # local = charge api/model/meilleurs_poids.pt
    model_mode: str = "local"

    max_upload_mb: int = 10

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
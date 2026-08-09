from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    app_name: str = "Segmentation API"
    app_version: str = "1.0.0"

    model_mode: str = "mock"

    mlflow_tracking_uri: str = (
        "sqlite:///../mlflow.db"
    )

    mlflow_experiment_name: str = (
        "future_vision_transport_segmentation_xpu"
    )

    max_upload_mb: int = 10


@lru_cache
def get_settings() -> Settings:
    return Settings()
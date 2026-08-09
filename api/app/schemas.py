from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_mode: str


class ModelInfoResponse(BaseModel):
    mode: str

    model_name: str | None = None
    alias: str | None = None
    version: str | None = None
    run_id: str | None = None
    model_uri: str | None = None
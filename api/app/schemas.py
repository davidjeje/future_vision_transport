from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_mode: str


class ModelInfoResponse(BaseModel):
    mode: str

    model_name: str | None = None
    architecture: str | None = None

    run_id: str | None = None

    miou: float | None = None
    dice: float | None = None

    nombre_classes: int | None = None
    largeur: int | None = None
    hauteur: int | None = None
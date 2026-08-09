from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response

from app.config import get_settings
from app.image_utils import (
    colorize_mask,
    decode_image,
    encode_png,
    image_to_tensor,
)
from app.model_service import SegmentationModelService
from app.schemas import HealthResponse, ModelInfoResponse


logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

settings = get_settings()
model_service = SegmentationModelService(settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle une seule fois au démarrage de l'application."""
    logger.info("Démarrage de l'API de segmentation.")

    model_service.load()

    logger.info("Modèle chargé. API prête.")

    yield

    logger.info("Arrêt de l'API.")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="API de prédiction pour un modèle de segmentation sémantique.",
    lifespan=lifespan,
)


@app.get("/")
def root():
    """Endpoint principal de l'API."""
    return {
        "message": settings.app_name,
        "docs": "/docs",
        "health": "/health",
        "model_info": "/model-info",
        "predict": "/predict",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
)
def health():
    """Vérifie que l'API fonctionne et que le modèle est chargé."""
    return HealthResponse(
        status="ok" if model_service.loaded else "degraded",
        model_loaded=model_service.loaded,
        model_mode=settings.model_mode,
    )


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
)
def model_info():
    """Retourne les informations sur le modèle actuellement chargé."""
    return ModelInfoResponse(
        **model_service.info.__dict__
    )


@app.post(
    "/predict",
    responses={
        200: {
            "content": {"image/png": {}},
            "description": "Masque de segmentation coloré au format PNG.",
        },
        400: {"description": "Image invalide."},
        413: {"description": "Image trop volumineuse."},
        415: {"description": "Format de fichier non supporté."},
        500: {"description": "Erreur pendant l'inférence."},
    },
)
async def predict(
    file: UploadFile = File(...)
):
    """
    Reçoit une image et retourne le masque de segmentation prédit
    au format PNG coloré.
    """

    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/webp",
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail="Formats acceptés : JPEG, PNG et WEBP.",
        )

    max_bytes = (
        settings.max_upload_mb
        * 1024
        * 1024
    )

    data = await file.read(max_bytes + 1)

    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Image trop volumineuse. "
                f"Maximum : {settings.max_upload_mb} MB."
            ),
        )

    if not data:
        raise HTTPException(
            status_code=400,
            detail="Le fichier envoyé est vide.",
        )

    try:
        image = decode_image(data)
    except Exception as exc:
        logger.exception("Impossible de décoder l'image.")
        raise HTTPException(
            status_code=400,
            detail="Image invalide.",
        ) from exc

    try:
        tensor = image_to_tensor(
            image,
            largeur=model_service.largeur,
            hauteur=model_service.hauteur,
        )

        logger.info(
            "Prédiction sur image %sx%s avec entrée modèle %sx%s.",
            image.width,
            image.height,
            model_service.largeur,
            model_service.hauteur,
        )

        prediction = model_service.predict(tensor)

        mask = colorize_mask(
            prediction=prediction,
            original_size=image.size,
        )

        png = encode_png(mask)

    except Exception as exc:
        logger.exception("Erreur pendant l'inférence.")
        raise HTTPException(
            status_code=500,
            detail=(
                "Erreur pendant la prédiction : "
                f"{type(exc).__name__}"
            ),
        ) from exc

    headers = {
        "X-Model-Mode": str(
            getattr(
                model_service.info,
                "mode",
                settings.model_mode,
            )
        )
    }

    model_name = getattr(model_service.info, "model_name", None)
    architecture = getattr(model_service.info, "architecture", None)
    run_id = getattr(model_service.info, "run_id", None)
    miou = getattr(model_service.info, "miou", None)
    dice = getattr(model_service.info, "dice", None)

    if model_name:
        headers["X-Model-Name"] = str(model_name)

    if architecture:
        headers["X-Model-Architecture"] = str(architecture)

    if run_id:
        headers["X-MLflow-Run-ID"] = str(run_id)

    if miou is not None:
        headers["X-Model-MIoU"] = str(miou)

    if dice is not None:
        headers["X-Model-Dice"] = str(dice)

    return Response(
        content=png,
        media_type="image/png",
        headers=headers,
    )

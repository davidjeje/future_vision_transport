from dataclasses import dataclass
import logging

import mlflow
import torch
from mlflow import MlflowClient

from app.config import Settings

from pathlib import Path
import sys

# dossier_parent
ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fonctions.segmentation import (
    construire_modele,
    extraire_logits,
)

logger = logging.getLogger(__name__)

@dataclass
class LoadedModelInfo:
    mode: str
    model_name: str | None = None
    architecture: str | None = None
    run_id: str | None = None
    miou: float | None = None
    dice: float | None = None


class SegmentationModelService:

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = None

        self.device = torch.device("cpu")

        self.largeur = 256
        self.hauteur = 128
        self.nombre_classes = 8

        self.info = LoadedModelInfo(
            mode=settings.model_mode
        )

    @property
    def loaded(self):
        if self.settings.model_mode == "mock":
            return True

        return self.model is not None

    def load(self):

        if self.settings.model_mode == "mock":
            logger.warning(
                "Mode mock : aucun modèle réel chargé."
            )
            return

        mlflow.set_tracking_uri(
            self.settings.mlflow_tracking_uri
        )

        client = MlflowClient()

        experiment = client.get_experiment_by_name(
            self.settings.mlflow_experiment_name
        )

        if experiment is None:
            raise RuntimeError(
                "Expérience MLflow introuvable."
            )

        runs = client.search_runs(
            experiment_ids=[
                experiment.experiment_id
            ],
            order_by=[
                "metrics.miou_validation DESC",
                "metrics.dice_validation DESC",
            ],
            max_results=1,
        )

        if not runs:
            raise RuntimeError(
                "Aucun run MLflow trouvé."
            )

        best_run = runs[0]

        run_id = best_run.info.run_id

        # Le notebook passe nom_modele comme run_name
        nom_modele = best_run.data.tags.get(
            "mlflow.runName"
        )

        self.largeur = int(
            best_run.data.params.get(
                "largeur_image",
                256,
            )
        )

        self.hauteur = int(
            best_run.data.params.get(
                "hauteur_image",
                128,
            )
        )

        self.nombre_classes = int(
            best_run.data.params.get(
                "nombre_classes",
                8,
            )
        )

        logger.info(
            "Meilleur run : %s - mIoU=%s",
            nom_modele,
            best_run.data.metrics.get(
                "miou_validation"
            ),
        )

        # Reconstruction de l'architecture
        #
        # IMPORTANT :
        # poids_preentraines=False car nous allons ensuite
        # charger le state_dict entraîné.
        self.model = construire_modele(
            nom_modele=nom_modele,
            nombre_classes=self.nombre_classes,
            poids_preentraines=False,
        )

        # Téléchargement de l'artifact MLflow
        chemin_poids = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="poids/meilleurs_poids.pt",
        )

        state_dict = torch.load(
            chemin_poids,
            map_location=self.device,
            weights_only=True,
        )

        self.model.load_state_dict(
            state_dict
        )

        self.model.to(
            self.device
        )

        self.model.eval()

        self.info = LoadedModelInfo(
            mode="mlflow",
            model_name=nom_modele,
            architecture=best_run.data.params.get(
                "architecture"
            ),
            run_id=run_id,
            miou=best_run.data.metrics.get(
                "miou_validation"
            ),
            dice=best_run.data.metrics.get(
                "dice_validation"
            ),
        )

    def predict(self, tensor):

        if self.settings.model_mode == "mock":
            return self._mock_predict(tensor)

        tensor = tensor.to(
            self.device
        )

        with torch.no_grad():

            output = self.model(
                tensor
            )

            logits = extraire_logits(
                output,
                (
                    self.hauteur,
                    self.largeur,
                ),
            )

            prediction = logits.argmax(
                dim=1
            )

        return prediction.cpu().numpy()

    @staticmethod
    def _mock_predict(tensor):

        image = tensor[0]

        gray = image.mean(
            dim=0
        )

        return (
            gray > gray.mean()
        ).long().unsqueeze(0).numpy()
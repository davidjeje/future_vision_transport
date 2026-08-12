from dataclasses import dataclass
import json
import logging
from pathlib import Path
import sys

import torch

from app.config import Settings


# ---------------------------------------------------------
# Chemins du projet
# ---------------------------------------------------------

# model_service.py :
# projet/api/app/model_service.py
#
# parents[0] -> app
# parents[1] -> api
# parents[2] -> racine du projet

API_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = Path(__file__).resolve().parents[2]

MODEL_DIR = API_DIR / "model"

CONFIG_PATH = MODEL_DIR / "model_config.json"
WEIGHTS_PATH = MODEL_DIR / "meilleurs_poids.pt"


# ---------------------------------------------------------
# Permet d'importer fonctions/segmentation.py
# ---------------------------------------------------------

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from fonctions.segmentation import (
    construire_modele,
    extraire_logits,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Informations exposées par /model-info
# ---------------------------------------------------------

@dataclass
class LoadedModelInfo:
    mode: str

    model_name: str | None = None
    architecture: str | None = None

    run_id: str | None = None

    miou: float | None = None
    dice: float | None = None

    nombre_classes: int | None = None
    largeur: int | None = None
    hauteur: int | None = None


# ---------------------------------------------------------
# Service du modèle
# ---------------------------------------------------------

class SegmentationModelService:

    def __init__(
        self,
        settings: Settings,
    ):
        self.settings = settings

        self.model = None

        # Heroku fera l'inférence sur CPU.
        self.device = torch.device("cpu")

        # Valeurs par défaut.
        # Elles seront remplacées par model_config.json.
        self.largeur = 256
        self.hauteur = 128
        self.nombre_classes = 8

        self.model_config = {}

        self.info = LoadedModelInfo(
            mode=settings.model_mode
        )

    # -----------------------------------------------------
    # État du modèle
    # -----------------------------------------------------

    @property
    def loaded(self) -> bool:

        if self.settings.model_mode == "mock":
            return True

        return self.model is not None

    # -----------------------------------------------------
    # Chargement du modèle
    # -----------------------------------------------------

    def load(self) -> None:

        # Mode de développement sans vrai modèle.
        if self.settings.model_mode == "mock":

            logger.warning(
                "Mode mock : aucun modèle réel chargé."
            )

            self.info = LoadedModelInfo(
                mode="mock"
            )

            return

        logger.info(
            "Chargement du modèle de segmentation local."
        )

        # -------------------------------------------------
        # Vérification des fichiers
        # -------------------------------------------------

        if not CONFIG_PATH.exists():

            raise FileNotFoundError(
                f"Fichier de configuration introuvable : "
                f"{CONFIG_PATH}"
            )

        if not WEIGHTS_PATH.exists():

            raise FileNotFoundError(
                f"Fichier de poids introuvable : "
                f"{WEIGHTS_PATH}"
            )

        # -------------------------------------------------
        # Lecture de model_config.json
        # -------------------------------------------------

        with CONFIG_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:

            self.model_config = json.load(
                file
            )

        # -------------------------------------------------
        # Paramètres nécessaires à l'inférence
        # -------------------------------------------------

        nom_modele = self.model_config[
            "nom_modele"
        ]

        self.nombre_classes = int(
            self.model_config.get(
                "nombre_classes",
                8,
            )
        )

        self.largeur = int(
            self.model_config.get(
                "largeur_image",
                256,
            )
        )

        self.hauteur = int(
            self.model_config.get(
                "hauteur_image",
                128,
            )
        )

        logger.info(
            "Modèle : %s",
            nom_modele,
        )

        logger.info(
            "Taille d'entrée : %sx%s",
            self.largeur,
            self.hauteur,
        )

        logger.info(
            "Nombre de classes : %s",
            self.nombre_classes,
        )

        # -------------------------------------------------
        # Reconstruction de l'architecture
        # -------------------------------------------------

        self.model = construire_modele(
            nom_modele=nom_modele,
            nombre_classes=self.nombre_classes,
            poids_preentraines=False,
        )

        # -------------------------------------------------
        # Chargement du state_dict
        # -------------------------------------------------

        logger.info(
            "Chargement des poids : %s",
            WEIGHTS_PATH,
        )

        state_dict = torch.load(
            WEIGHTS_PATH,
            map_location=self.device,
            weights_only=True,
        )

        self.model.load_state_dict(
            state_dict
        )

        # -------------------------------------------------
        # Passage en mode inférence
        # -------------------------------------------------

        self.model.to(
            self.device
        )

        self.model.eval()

        # -------------------------------------------------
        # Métadonnées
        # -------------------------------------------------

        metriques = self.model_config.get(
            "metriques_validation",
            {},
        )

        self.info = LoadedModelInfo(
            mode="local",
            model_name=nom_modele,
            architecture=self.model_config.get(
                "architecture"
            ),
            run_id=self.model_config.get(
                "run_id"
            ),
            miou=metriques.get(
                "miou"
            ),
            dice=metriques.get(
                "dice"
            ),
            nombre_classes=self.nombre_classes,
            largeur=self.largeur,
            hauteur=self.hauteur,
        )

        logger.info(
            "Modèle chargé avec succès."
        )

        logger.info(
            "mIoU validation : %s",
            self.info.miou,
        )

        logger.info(
            "Dice validation : %s",
            self.info.dice,
        )

    # -----------------------------------------------------
    # Prédiction
    # -----------------------------------------------------

    def predict(
        self,
        tensor: torch.Tensor,
    ):

        if self.settings.model_mode == "mock":
            return self._mock_predict(
                tensor
            )

        if self.model is None:

            raise RuntimeError(
                "Le modèle n'est pas chargé."
            )

        tensor = tensor.to(
            self.device
        )

        with torch.no_grad():

            output = self.model(
                tensor
            )

            # Compatible avec vos différents modèles,
            # notamment SegFormer.
            logits = extraire_logits(
                output,
                (
                    self.hauteur,
                    self.largeur,
                ),
            )

            # Pour chaque pixel :
            # classe ayant le logit le plus élevé.
            prediction = logits.argmax(
                dim=1
            )

        return (
            prediction
            .cpu()
            .numpy()
        )

    # -----------------------------------------------------
    # Faux modèle pour tests locaux
    # -----------------------------------------------------

    @staticmethod
    def _mock_predict(
        tensor: torch.Tensor,
    ):

        image = tensor[0]

        gray = image.mean(
            dim=0
        )

        return (
            gray > gray.mean()
        ).long().unsqueeze(0).numpy()
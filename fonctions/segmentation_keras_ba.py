"""Outils Keras 3 (backend PyTorch/XPU) pour la segmentation Cityscapes.

Migration du module PyTorch historique ``fonctions/segmentation.py``.
Le backend Keras est fixé à ``torch`` avant l'import de Keras. Lorsque
``torch.xpu`` est disponible, Keras 3 l'utilise automatiquement (ou via
``KERAS_TORCH_DEVICE=xpu``).

Principes importants
--------------------
- 8 catégories métier Cityscapes, comme dans le projet d'origine.
- Split train / validation / test AVANT toute augmentation.
- Les augmentations aléatoires s'appliquent uniquement au train afin d'éviter
  la fuite de données. Validation et test restent déterministes.
- Deux scénarios peuvent être comparés : train augmenté et train non augmenté,
  en gardant exactement les mêmes val/test.
- Optimisation bayésienne KerasTuner sur train/validation ; le test n'est jamais
  utilisé pour choisir les hyperparamètres. Les mêmes meilleurs hyperparamètres
  peuvent ensuite être réutilisés avec et sans data augmentation.
- ``keras.utils.PyDataset`` permet plusieurs workers. Sous Windows/Jupyter,
  ``use_multiprocessing=False`` est le choix sûr par défaut.
"""
from __future__ import annotations

import contextlib
import json
import os
import random
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

# IMPORTANT : doit être défini avant l'import de Keras.
os.environ.setdefault("KERAS_BACKEND", "torch")

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Sur une machine Intel XPU, on demande explicitement le périphérique à Keras.
if "KERAS_TORCH_DEVICE" not in os.environ:
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        os.environ["KERAS_TORCH_DEVICE"] = "xpu"
    elif torch.cuda.is_available():
        os.environ["KERAS_TORCH_DEVICE"] = "cuda"
    else:
        os.environ["KERAS_TORCH_DEVICE"] = "cpu"

import keras
from keras import layers, ops

try:
    import mlflow
except Exception:  # pragma: no cover - dépendance optionnelle
    mlflow = None

try:
    import keras_tuner as kt
except Exception:  # pragma: no cover - dépendance optionnelle
    kt = None


NOMS_CATEGORIES = [
    "vide",
    "plat",
    "construction",
    "objet",
    "nature",
    "ciel",
    "humain",
    "vehicule",
]

COULEURS_CATEGORIES = np.array(
    [
        [0, 0, 0],
        [128, 64, 128],
        [70, 70, 70],
        [153, 153, 153],
        [107, 142, 35],
        [70, 130, 180],
        [220, 20, 60],
        [0, 0, 142],
    ],
    dtype=np.uint8,
)

# Regroupement des labelIds Cityscapes en huit catégories métier.
CORRESPONDANCE_LABELS = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
    7: 1,
    8: 1,
    9: 1,
    10: 1,
    11: 2,
    12: 2,
    13: 2,
    14: 2,
    15: 2,
    16: 2,
    17: 3,
    18: 3,
    19: 3,
    20: 3,
    21: 4,
    22: 4,
    23: 5,
    24: 6,
    25: 6,
    26: 7,
    27: 7,
    28: 7,
    29: 7,
    30: 7,
    31: 7,
    32: 7,
    33: 7,
    -1: 0,
    255: 0,
}


@dataclass(frozen=True)
class DiagnosticEnvironnement:
    backend_keras: str
    version_keras: str
    version_torch: str
    xpu_disponible: bool
    cuda_disponible: bool
    peripherique_demande: str
    peripherique_keras_detecte: str
    nombre_cpu_logiques: int
    nom_xpu: str | None = None


def fixer_aleatoire(graine: int = 42) -> None:
    """Fixe les graines NumPy, Python, PyTorch et Keras."""
    random.seed(graine)
    np.random.seed(graine)
    torch.manual_seed(graine)
    keras.utils.set_random_seed(graine)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed_all(graine)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(graine)


def choisir_peripherique() -> str:
    """Retourne le périphérique d'entraînement préféré : xpu, cuda ou cpu."""
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def detecter_nombre_travailleurs(maximum: int = 6, reserve_cpu: int = 2) -> int:
    """Choisit un nombre prudent de workers pour le chargement des images.

    Avec 22 CPU logiques, la valeur par défaut est 6. C'est volontairement
    conservateur : le GPU Intel intégré partage la mémoire et la bande passante
    avec le CPU, et OpenCV peut lui-même lancer des threads.
    """
    n_cpu = os.cpu_count() or 1
    disponible = max(1, n_cpu - reserve_cpu)
    return max(1, min(maximum, disponible))


def configurer_threads_opencv(nombre_threads: int = 1) -> None:
    """Évite la sur-souscription OpenCV lorsque plusieurs workers sont actifs."""
    try:
        cv2.setNumThreads(max(1, int(nombre_threads)))
    except Exception:
        pass


def diagnostiquer_environnement() -> DiagnosticEnvironnement:
    """Teste le backend Keras et crée une petite couche pour voir son device réel."""
    backend = keras.backend.backend()
    xpu_disponible = bool(hasattr(torch, "xpu") and torch.xpu.is_available())
    cuda_disponible = bool(torch.cuda.is_available())
    peripherique_demande = os.environ.get("KERAS_TORCH_DEVICE", choisir_peripherique())
    nom_xpu = None
    if xpu_disponible:
        try:
            nom_xpu = torch.xpu.get_device_name(0)
        except Exception:
            try:
                nom_xpu = torch.xpu.get_device_properties(0).name
            except Exception:
                nom_xpu = "Intel XPU"

    peripherique_keras_detecte = "inconnu"
    if backend == "torch":
        couche = layers.Dense(2)
        _ = couche(np.zeros((1, 3), dtype=np.float32))
        try:
            peripherique_keras_detecte = str(couche.weights[0].value.device)
        except Exception:
            peripherique_keras_detecte = peripherique_demande

    return DiagnosticEnvironnement(
        backend_keras=backend,
        version_keras=keras.__version__,
        version_torch=torch.__version__,
        xpu_disponible=xpu_disponible,
        cuda_disponible=cuda_disponible,
        peripherique_demande=peripherique_demande,
        peripherique_keras_detecte=peripherique_keras_detecte,
        nombre_cpu_logiques=os.cpu_count() or 1,
        nom_xpu=nom_xpu,
    )


def verifier_backend_keras_torch(exiger_xpu: bool = False) -> DiagnosticEnvironnement:
    """Valide la configuration Keras/Torch et, optionnellement, la présence XPU."""
    diagnostic = diagnostiquer_environnement()
    if diagnostic.backend_keras != "torch":
        raise RuntimeError(
            "Keras n'utilise pas le backend Torch. Redémarrez le kernel puis "
            "définissez KERAS_BACKEND='torch' avant tout import de keras."
        )
    if exiger_xpu and not diagnostic.xpu_disponible:
        raise RuntimeError(
            "torch.xpu.is_available() est False. Vérifiez le pilote Intel GPU et "
            "l'installation de la roue PyTorch XPU."
        )
    return diagnostic


def creer_tableau_donnees(dossier_images: Path, dossier_masques: Path) -> pd.DataFrame:
    lignes: list[dict[str, str]] = []
    dossier_images = Path(dossier_images)
    dossier_masques = Path(dossier_masques)
    for chemin_image in sorted(dossier_images.rglob("*_leftImg8bit.png")):
        relatif = chemin_image.relative_to(dossier_images)
        nom_masque = chemin_image.name.replace(
            "_leftImg8bit.png", "_gtFine_labelIds.png"
        )
        chemin_masque = dossier_masques / relatif.parent / nom_masque
        if chemin_masque.exists():
            lignes.append(
                {
                    "ville": relatif.parts[0] if relatif.parts else "inconnue",
                    "chemin_image": str(chemin_image),
                    "chemin_masque": str(chemin_masque),
                }
            )
    return pd.DataFrame(lignes)


def verifier_couples_images_masques(tableau: pd.DataFrame) -> pd.DataFrame:
    lignes = []
    for indice, ligne in tableau.reset_index(drop=True).iterrows():
        chemin_image = str(ligne["chemin_image"])
        chemin_masque = str(ligne["chemin_masque"])
        image = cv2.imread(chemin_image, cv2.IMREAD_COLOR)
        masque = cv2.imread(chemin_masque, cv2.IMREAD_UNCHANGED)
        image_lisible = image is not None
        masque_lisible = masque is not None
        hauteur_image = largeur_image = hauteur_masque = largeur_masque = np.nan
        if image_lisible:
            hauteur_image, largeur_image = image.shape[:2]
        if masque_lisible:
            hauteur_masque, largeur_masque = masque.shape[:2]
        memes_dimensions = bool(
            image_lisible
            and masque_lisible
            and hauteur_image == hauteur_masque
            and largeur_image == largeur_masque
        )
        lignes.append(
            {
                "indice": indice,
                "ville": ligne.get("ville", "inconnue"),
                "chemin_image": chemin_image,
                "chemin_masque": chemin_masque,
                "image_lisible": image_lisible,
                "masque_lisible": masque_lisible,
                "hauteur_image": hauteur_image,
                "largeur_image": largeur_image,
                "hauteur_masque": hauteur_masque,
                "largeur_masque": largeur_masque,
                "memes_dimensions": memes_dimensions,
            }
        )
    return pd.DataFrame(lignes)


def filtrer_couples_valides(
    tableau: pd.DataFrame,
    rapport: pd.DataFrame | None = None,
    exiger_memes_dimensions: bool = True,
) -> pd.DataFrame:
    """Conserve uniquement les couples image/masque lisibles et cohérents."""
    if rapport is None:
        rapport = verifier_couples_images_masques(tableau)
    if rapport.empty:
        return tableau.iloc[0:0].copy()

    masque_valide = rapport["image_lisible"] & rapport["masque_lisible"]
    if exiger_memes_dimensions:
        masque_valide = masque_valide & rapport["memes_dimensions"]

    indices = rapport.loc[masque_valide, "indice"].astype(int).to_numpy()
    return tableau.iloc[indices].reset_index(drop=True)


def convertir_masque_huit_categories(masque: np.ndarray) -> np.ndarray:
    resultat = np.zeros_like(masque, dtype=np.uint8)
    for identifiant, categorie in CORRESPONDANCE_LABELS.items():
        resultat[masque == identifiant] = categorie
    return resultat


def analyser_distribution_categories(
    tableau: pd.DataFrame,
    noms_categories: list[str] | None = None,
) -> pd.DataFrame:
    noms = noms_categories or NOMS_CATEGORIES
    comptes = np.zeros(len(noms), dtype=np.int64)
    masques_analyses = 0
    for chemin_masque in tableau["chemin_masque"]:
        masque_brut = cv2.imread(str(chemin_masque), cv2.IMREAD_UNCHANGED)
        if masque_brut is None:
            continue
        if masque_brut.ndim == 3:
            masque_brut = masque_brut[:, :, 0]
        # Le cache prétraité contient déjà 0..7.
        uniques = np.unique(masque_brut)
        if len(uniques) and uniques.max() <= 7:
            masque_huit = masque_brut.astype(np.uint8, copy=False)
        else:
            masque_huit = convertir_masque_huit_categories(masque_brut)
        comptes += np.bincount(masque_huit.reshape(-1), minlength=len(noms))[: len(noms)]
        masques_analyses += 1
    total_pixels = int(comptes.sum())
    proportions = (
        comptes / total_pixels * 100.0
        if total_pixels > 0
        else np.zeros_like(comptes, dtype=float)
    )
    distribution = pd.DataFrame(
        {
            "identifiant_categorie": np.arange(len(noms), dtype=int),
            "categorie": noms,
            "nombre_pixels": comptes,
            "proportion_pixels_pourcent": proportions,
        }
    )
    distribution["masques_analyses"] = masques_analyses
    proportions_non_nulles = distribution.loc[
        distribution["proportion_pixels_pourcent"] > 0,
        "proportion_pixels_pourcent",
    ]
    distribution.attrs["rapport_desequilibre"] = (
        float(proportions_non_nulles.max() / proportions_non_nulles.min())
        if len(proportions_non_nulles) > 1
        else 0.0
    )
    return distribution


def separer_donnees_train_val_test(
    tableau: pd.DataFrame,
    proportion_validation: float = 0.15,
    proportion_test: float = 0.15,
    graine: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Crée trois splits exclusifs avant toute augmentation.

    Par défaut : 70 % train, 15 % validation, 15 % test.
    """
    if tableau.empty:
        raise ValueError("Aucun couple image-masque n'a été trouvé.")
    if proportion_validation <= 0 or proportion_test <= 0:
        raise ValueError("Les proportions validation et test doivent être > 0.")
    if proportion_validation + proportion_test >= 1:
        raise ValueError("validation + test doit être strictement inférieur à 1.")

    melange = tableau.sample(frac=1.0, random_state=graine).reset_index(drop=True)
    n = len(melange)
    n_test = max(1, int(round(n * proportion_test)))
    n_val = max(1, int(round(n * proportion_validation)))
    if n_test + n_val >= n:
        raise ValueError("Pas assez de données pour créer les trois splits.")

    test = melange.iloc[:n_test].reset_index(drop=True)
    validation = melange.iloc[n_test : n_test + n_val].reset_index(drop=True)
    entrainement = melange.iloc[n_test + n_val :].reset_index(drop=True)
    return entrainement, validation, test


def pretraiter_donnees(
    tableau: pd.DataFrame,
    dossier_sortie: Path,
    largeur: int,
    hauteur: int,
) -> pd.DataFrame:
    """Redimensionne et met en cache images + masques à 8 catégories."""
    dossier_sortie = Path(dossier_sortie)
    dossier_images = dossier_sortie / "images"
    dossier_masques = dossier_sortie / "masques_8_categories"
    dossier_images.mkdir(parents=True, exist_ok=True)
    dossier_masques.mkdir(parents=True, exist_ok=True)
    lignes = []
    for indice, ligne in tableau.reset_index(drop=True).iterrows():
        chemin_image = str(ligne["chemin_image"])
        chemin_masque = str(ligne["chemin_masque"])
        image_bgr = cv2.imread(chemin_image, cv2.IMREAD_COLOR)
        masque_brut = cv2.imread(chemin_masque, cv2.IMREAD_UNCHANGED)
        if image_bgr is None or masque_brut is None:
            continue
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if masque_brut.ndim == 3:
            masque_brut = masque_brut[:, :, 0]
        h_img, w_img = image.shape[:2]
        if masque_brut.shape[:2] != (h_img, w_img):
            masque_brut = cv2.resize(
                masque_brut,
                (w_img, h_img),
                interpolation=cv2.INTER_NEAREST,
            )
        masque = convertir_masque_huit_categories(masque_brut)
        image = cv2.resize(image, (largeur, hauteur), interpolation=cv2.INTER_LINEAR)
        masque = cv2.resize(
            masque, (largeur, hauteur), interpolation=cv2.INTER_NEAREST
        )
        nom_base = f"{indice:06d}"
        chemin_image_sortie = dossier_images / f"{nom_base}_image.png"
        chemin_masque_sortie = dossier_masques / f"{nom_base}_masque.png"
        cv2.imwrite(
            str(chemin_image_sortie), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        )
        cv2.imwrite(str(chemin_masque_sortie), masque)
        lignes.append(
            {
                "ville": ligne.get("ville", "inconnue"),
                "chemin_image": str(chemin_image_sortie),
                "chemin_masque": str(chemin_masque_sortie),
            }
        )
    return pd.DataFrame(lignes)


def cache_pretraite_valide(
    tableau: pd.DataFrame,
    largeur: int,
    hauteur: int,
    verifier_tout: bool = False,
) -> bool:
    """Vérifie qu'un index de cache pointe vers des images/masques utilisables."""
    if tableau is None or tableau.empty:
        return False
    if not {"chemin_image", "chemin_masque"}.issubset(tableau.columns):
        return False

    if verifier_tout or len(tableau) <= 20:
        indices = range(len(tableau))
    else:
        indices = np.linspace(0, len(tableau) - 1, num=20, dtype=int)

    for indice in indices:
        ligne = tableau.iloc[int(indice)]
        chemin_image = Path(str(ligne["chemin_image"]))
        chemin_masque = Path(str(ligne["chemin_masque"]))
        if not chemin_image.exists() or not chemin_masque.exists():
            return False
        image = cv2.imread(str(chemin_image), cv2.IMREAD_COLOR)
        masque = cv2.imread(str(chemin_masque), cv2.IMREAD_UNCHANGED)
        if image is None or masque is None:
            return False
        if image.shape[:2] != (hauteur, largeur):
            return False
        if masque.shape[:2] != (hauteur, largeur):
            return False
        uniques = np.unique(masque)
        if len(uniques) and (uniques.min() < 0 or uniques.max() > 7):
            return False
    return True


def creer_augmentations_pretraitees() -> A.Compose:
    """Augmentations réalistes à la volée, image et masque synchronisés."""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, border_mode=cv2.BORDER_REFLECT_101, p=0.30),
            A.Affine(
                scale=(0.90, 1.10),
                translate_percent=(-0.05, 0.05),
                rotate=0,
                shear=0,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.35,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.35,
            ),
            A.HueSaturationValue(
                hue_shift_limit=5,
                sat_shift_limit=10,
                val_shift_limit=8,
                p=0.20,
            ),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5)),
                    A.MotionBlur(blur_limit=5),
                ],
                p=0.15,
            ),
            A.GaussNoise(std_range=(0.01, 0.05), p=0.15),
        ]
    )


def creer_transformation_deterministe_pretraitee() -> A.Compose:
    return A.Compose([])


def creer_augmentations(largeur: int, hauteur: int) -> A.Compose:
    transformations = creer_augmentations_pretraitees().transforms + [
        A.Resize(height=hauteur, width=largeur)
    ]
    return A.Compose(transformations)


def creer_transformation_deterministe(largeur: int, hauteur: int) -> A.Compose:
    return A.Compose([A.Resize(height=hauteur, width=largeur)])


class JeuSegmentationKeras(keras.utils.PyDataset):
    """PyDataset Keras : sortie BHWC float32 0..255 et masque BHW int32."""

    def __init__(
        self,
        tableau: pd.DataFrame,
        transformations: A.Compose,
        taille_lot: int,
        melanger: bool,
        donnees_pretraitees: bool = False,
        graine: int = 42,
        nombre_travailleurs: int = 1,
        utiliser_multiprocessing: bool = False,
        taille_file: int = 8,
    ) -> None:
        super().__init__(
            workers=max(1, int(nombre_travailleurs)),
            use_multiprocessing=bool(utiliser_multiprocessing),
            max_queue_size=max(1, int(taille_file)),
        )
        self.tableau = tableau.reset_index(drop=True)
        self.transformations = transformations
        self.taille_lot = int(taille_lot)
        self.melanger = bool(melanger)
        self.donnees_pretraitees = bool(donnees_pretraitees)
        self.graine = int(graine)
        self._rng = np.random.default_rng(graine)
        self.indices = np.arange(len(self.tableau))
        if self.melanger:
            self._rng.shuffle(self.indices)

    def __len__(self) -> int:
        return int(np.ceil(len(self.tableau) / self.taille_lot))

    def _lire_exemple(self, indice: int) -> tuple[np.ndarray, np.ndarray]:
        ligne = self.tableau.iloc[indice]
        chemin_image = str(ligne.chemin_image)
        chemin_masque = str(ligne.chemin_masque)
        image_bgr = cv2.imread(chemin_image, cv2.IMREAD_COLOR)
        masque_brut = cv2.imread(chemin_masque, cv2.IMREAD_UNCHANGED)
        if image_bgr is None:
            raise FileNotFoundError(f"Impossible de lire l'image : {chemin_image}")
        if masque_brut is None:
            raise FileNotFoundError(f"Impossible de lire le masque : {chemin_masque}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if masque_brut.ndim == 3:
            masque_brut = masque_brut[:, :, 0]
        if self.donnees_pretraitees:
            masque = masque_brut.astype(np.uint8, copy=False)
        else:
            h_img, w_img = image.shape[:2]
            if masque_brut.shape[:2] != (h_img, w_img):
                masque_brut = cv2.resize(
                    masque_brut,
                    (w_img, h_img),
                    interpolation=cv2.INTER_NEAREST,
                )
            masque = convertir_masque_huit_categories(masque_brut)
        transforme = self.transformations(image=image, mask=masque)
        image = np.ascontiguousarray(transforme["image"].astype(np.float32))
        masque = np.ascontiguousarray(transforme["mask"].astype(np.int32))
        return image, masque

    def __getitem__(self, indice_lot: int) -> tuple[np.ndarray, np.ndarray]:
        debut = indice_lot * self.taille_lot
        fin = min((indice_lot + 1) * self.taille_lot, len(self.tableau))
        indices_lot = self.indices[debut:fin]
        images, masques = zip(*(self._lire_exemple(int(i)) for i in indices_lot))
        return np.stack(images, axis=0), np.stack(masques, axis=0)

    def on_epoch_end(self) -> None:
        if self.melanger:
            self._rng.shuffle(self.indices)


def creer_chargeurs_trois_splits(
    tableau_entrainement: pd.DataFrame,
    tableau_validation: pd.DataFrame,
    tableau_test: pd.DataFrame,
    largeur: int,
    hauteur: int,
    taille_lot: int,
    nombre_travailleurs: int = 1,
    donnees_pretraitees: bool = False,
    avec_augmentation: bool = True,
    utiliser_multiprocessing: bool = False,
    graine: int = 42,
) -> tuple[JeuSegmentationKeras, JeuSegmentationKeras, JeuSegmentationKeras]:
    """Crée train/val/test. Seul le train reçoit les augmentations aléatoires."""
    if donnees_pretraitees:
        trans_train = (
            creer_augmentations_pretraitees()
            if avec_augmentation
            else creer_transformation_deterministe_pretraitee()
        )
        trans_eval = creer_transformation_deterministe_pretraitee()
    else:
        trans_train = (
            creer_augmentations(largeur, hauteur)
            if avec_augmentation
            else creer_transformation_deterministe(largeur, hauteur)
        )
        trans_eval = creer_transformation_deterministe(largeur, hauteur)

    commun = dict(
        taille_lot=taille_lot,
        donnees_pretraitees=donnees_pretraitees,
        graine=graine,
        nombre_travailleurs=nombre_travailleurs,
        utiliser_multiprocessing=utiliser_multiprocessing,
    )
    train = JeuSegmentationKeras(
        tableau_entrainement,
        trans_train,
        melanger=True,
        **commun,
    )
    val = JeuSegmentationKeras(
        tableau_validation,
        trans_eval,
        melanger=False,
        **commun,
    )
    test = JeuSegmentationKeras(
        tableau_test,
        trans_eval,
        melanger=False,
        **commun,
    )
    return train, val, test



def creer_chargeurs_train_validation(
    tableau_entrainement: pd.DataFrame,
    tableau_validation: pd.DataFrame,
    largeur: int,
    hauteur: int,
    taille_lot: int,
    nombre_travailleurs: int = 1,
    donnees_pretraitees: bool = False,
    avec_augmentation: bool = True,
    utiliser_multiprocessing: bool = False,
    graine: int = 42,
) -> tuple[JeuSegmentationKeras, JeuSegmentationKeras]:
    """Crée uniquement train/validation ; utile pour tuning et entraînement final."""
    if donnees_pretraitees:
        trans_train = (
            creer_augmentations_pretraitees()
            if avec_augmentation
            else creer_transformation_deterministe_pretraitee()
        )
        trans_val = creer_transformation_deterministe_pretraitee()
    else:
        trans_train = (
            creer_augmentations(largeur, hauteur)
            if avec_augmentation
            else creer_transformation_deterministe(largeur, hauteur)
        )
        trans_val = creer_transformation_deterministe(largeur, hauteur)
    commun = dict(
        taille_lot=taille_lot,
        donnees_pretraitees=donnees_pretraitees,
        graine=graine,
        nombre_travailleurs=nombre_travailleurs,
        utiliser_multiprocessing=utiliser_multiprocessing,
    )
    train = JeuSegmentationKeras(
        tableau_entrainement, trans_train, melanger=True, **commun
    )
    val = JeuSegmentationKeras(
        tableau_validation, trans_val, melanger=False, **commun
    )
    return train, val


def creer_chargeur_test(
    tableau_test: pd.DataFrame,
    largeur: int,
    hauteur: int,
    taille_lot: int,
    nombre_travailleurs: int = 1,
    donnees_pretraitees: bool = False,
    utiliser_multiprocessing: bool = False,
    graine: int = 42,
) -> JeuSegmentationKeras:
    """Crée un chargeur test déterministe, à appeler uniquement à la toute fin."""
    trans_test = (
        creer_transformation_deterministe_pretraitee()
        if donnees_pretraitees
        else creer_transformation_deterministe(largeur, hauteur)
    )
    return JeuSegmentationKeras(
        tableau_test,
        trans_test,
        taille_lot=taille_lot,
        melanger=False,
        donnees_pretraitees=donnees_pretraitees,
        graine=graine,
        nombre_travailleurs=nombre_travailleurs,
        utiliser_multiprocessing=utiliser_multiprocessing,
    )


def creer_deux_scenarios_chargeurs(**kwargs: Any) -> dict[str, tuple[JeuSegmentationKeras, JeuSegmentationKeras, JeuSegmentationKeras]]:
    """Retourne les mêmes splits avec train augmenté et train non augmenté."""
    return {
        "avec_augmentation": creer_chargeurs_trois_splits(
            **kwargs, avec_augmentation=True
        ),
        "sans_augmentation": creer_chargeurs_trois_splits(
            **kwargs, avec_augmentation=False
        ),
    }


def _bloc_double_convolution(x, filtres: int, dropout: float = 0.0, nom: str = "bloc"):
    x = layers.Conv2D(filtres, 3, padding="same", use_bias=False, name=f"{nom}_conv1")(x)
    x = layers.BatchNormalization(name=f"{nom}_bn1")(x)
    x = layers.Activation("relu", name=f"{nom}_relu1")(x)
    x = layers.Conv2D(filtres, 3, padding="same", use_bias=False, name=f"{nom}_conv2")(x)
    x = layers.BatchNormalization(name=f"{nom}_bn2")(x)
    x = layers.Activation("relu", name=f"{nom}_relu2")(x)
    if dropout > 0:
        x = layers.SpatialDropout2D(dropout, name=f"{nom}_dropout")(x)
    return x


def construire_unet_baseline(
    hauteur: int,
    largeur: int,
    nombre_classes: int = 8,
    filtres_base: int = 16,
    dropout: float = 0.0,
) -> keras.Model:
    entrees = keras.Input((hauteur, largeur, 3), name="image")
    x0 = layers.Rescaling(1.0 / 255.0, name="normalisation_0_1")(entrees)
    e1 = _bloc_double_convolution(x0, filtres_base, dropout, "enc1")
    e2 = _bloc_double_convolution(layers.MaxPool2D()(e1), filtres_base * 2, dropout, "enc2")
    e3 = _bloc_double_convolution(layers.MaxPool2D()(e2), filtres_base * 4, dropout, "enc3")
    e4 = _bloc_double_convolution(layers.MaxPool2D()(e3), filtres_base * 8, dropout, "enc4")
    centre = _bloc_double_convolution(
        layers.MaxPool2D()(e4), filtres_base * 16, dropout, "centre"
    )

    def decoder(x, skip, filtres, nom):
        x = layers.Conv2DTranspose(filtres, 2, strides=2, padding="same", name=f"{nom}_up")(x)
        x = layers.Concatenate(name=f"{nom}_concat")([x, skip])
        return _bloc_double_convolution(x, filtres, dropout, nom)

    d4 = decoder(centre, e4, filtres_base * 8, "dec4")
    d3 = decoder(d4, e3, filtres_base * 4, "dec3")
    d2 = decoder(d3, e2, filtres_base * 2, "dec2")
    d1 = decoder(d2, e1, filtres_base, "dec1")
    logits = layers.Conv2D(nombre_classes, 1, name="logits")(d1)
    return keras.Model(entrees, logits, name="unet_baseline")


def construire_unet_mobilenetv2(
    hauteur: int,
    largeur: int,
    nombre_classes: int = 8,
    poids_preentraines: bool = True,
    dropout: float = 0.1,
) -> keras.Model:
    entrees = keras.Input((hauteur, largeur, 3), name="image")
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenet_preprocess")(entrees)
    encodeur = keras.applications.MobileNetV2(
        input_tensor=x,
        include_top=False,
        weights="imagenet" if poids_preentraines else None,
        alpha=1.0,
    )
    noms_skips = [
        "block_1_expand_relu",   # 1/2
        "block_3_expand_relu",   # 1/4
        "block_6_expand_relu",   # 1/8
        "block_13_expand_relu",  # 1/16
    ]
    skips = [encodeur.get_layer(n).output for n in noms_skips]
    x = encodeur.output  # 1/32
    for i, (skip, filtres) in enumerate(zip(reversed(skips), [256, 128, 64, 32]), start=1):
        x = layers.Conv2DTranspose(filtres, 3, strides=2, padding="same", name=f"mob_dec{i}_up")(x)
        x = layers.Concatenate(name=f"mob_dec{i}_concat")([x, skip])
        x = _bloc_double_convolution(x, filtres, dropout, f"mob_dec{i}")
    x = layers.Conv2DTranspose(24, 3, strides=2, padding="same", name="mob_sortie_up")(x)
    x = _bloc_double_convolution(x, 24, dropout, "mob_sortie")
    logits = layers.Conv2D(nombre_classes, 1, name="logits")(x)
    return keras.Model(entrees, logits, name="unet_mobilenetv2")


def _aspp(x, filtres: int = 128, taux: Sequence[int] = (6, 12, 18), dropout: float = 0.1):
    branches = [
        layers.Conv2D(filtres, 1, padding="same", use_bias=False, name="aspp_1x1")(x)
    ]
    branches[0] = layers.BatchNormalization(name="aspp_1x1_bn")(branches[0])
    branches[0] = layers.Activation("relu", name="aspp_1x1_relu")(branches[0])
    for taux_dilatation in taux:
        b = layers.Conv2D(
            filtres,
            3,
            padding="same",
            dilation_rate=taux_dilatation,
            use_bias=False,
            name=f"aspp_d{taux_dilatation}",
        )(x)
        b = layers.BatchNormalization(name=f"aspp_d{taux_dilatation}_bn")(b)
        b = layers.Activation("relu", name=f"aspp_d{taux_dilatation}_relu")(b)
        branches.append(b)

    h, w = int(x.shape[1]), int(x.shape[2])
    b = layers.GlobalAveragePooling2D(name="aspp_gap")(x)
    b = layers.Reshape((1, 1, int(x.shape[-1])), name="aspp_gap_reshape")(b)
    b = layers.Conv2D(filtres, 1, use_bias=False, name="aspp_image_conv")(b)
    b = layers.BatchNormalization(name="aspp_image_bn")(b)
    b = layers.Activation("relu", name="aspp_image_relu")(b)
    b = layers.Resizing(h, w, interpolation="bilinear", name="aspp_image_resize")(b)
    branches.append(b)

    x = layers.Concatenate(name="aspp_concat")(branches)
    x = layers.Conv2D(filtres, 1, use_bias=False, name="aspp_projection")(x)
    x = layers.BatchNormalization(name="aspp_projection_bn")(x)
    x = layers.Activation("relu", name="aspp_projection_relu")(x)
    if dropout > 0:
        x = layers.Dropout(dropout, name="aspp_dropout")(x)
    return x


def construire_deeplabv3_mobilenetv2(
    hauteur: int,
    largeur: int,
    nombre_classes: int = 8,
    poids_preentraines: bool = True,
    dropout: float = 0.1,
) -> keras.Model:
    entrees = keras.Input((hauteur, largeur, 3), name="image")
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenet_preprocess")(entrees)
    base = keras.applications.MobileNetV2(
        input_tensor=x,
        include_top=False,
        weights="imagenet" if poids_preentraines else None,
    )
    # Sortie ~1/16, suffisante pour un DeepLabV3 léger sur un iGPU.
    caracteristiques = base.get_layer("block_13_expand_relu").output
    x = _aspp(caracteristiques, filtres=128, dropout=dropout)
    x = layers.Conv2D(nombre_classes, 1, name="logits_basse_resolution")(x)
    logits = layers.Resizing(
        hauteur, largeur, interpolation="bilinear", name="logits"
    )(x)
    return keras.Model(entrees, logits, name="deeplabv3_mobilenetv2")


def construire_segformer_b0(
    hauteur: int,
    largeur: int,
    nombre_classes: int = 8,
    poids_preentraines: bool = True,
    projection_filters: int = 128,
) -> keras.Model:
    """Construit SegFormer B0 via KerasHub avec une tête à 8 classes.

    On recharge la configuration MiT B0 du preset Cityscapes dans tous les cas.
    ``load_weights`` contrôle uniquement le transfert des poids de l'encodeur ;
    la tête de segmentation est toujours reconstruite pour les 8 catégories métier.
    Cette approche évite de dépendre des noms d'arguments internes du constructeur
    MiT, qui peuvent évoluer entre versions de KerasHub.
    """
    try:
        import keras_hub
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Le modèle SegFormer nécessite `pip install keras-hub`."
        ) from exc

    preset_encodeur = "mit_b0_cityscapes_1024"
    encodeur = keras_hub.models.MiTBackbone.from_preset(
        preset_encodeur,
        load_weights=bool(poids_preentraines),
        image_shape=(hauteur, largeur, 3),
    )
    convertisseur = keras_hub.layers.ImageConverter.from_preset(
        preset_encodeur,
        image_size=(hauteur, largeur),
        crop_to_aspect_ratio=False,
    )
    backbone = keras_hub.models.SegFormerBackbone(
        image_encoder=encodeur,
        projection_filters=int(projection_filters),
    )
    coeur = keras_hub.models.SegFormerImageSegmenter(
        backbone=backbone,
        num_classes=nombre_classes,
        preprocessor=None,
    )

    entrees = keras.Input((hauteur, largeur, 3), name="image")
    x = convertisseur(entrees)
    sorties = coeur(x)
    # Garantit une résolution identique au masque métier.
    if sorties.shape[1] != hauteur or sorties.shape[2] != largeur:
        sorties = layers.Resizing(
            hauteur, largeur, interpolation="bilinear", name="logits_resize"
        )(sorties)
    return keras.Model(entrees, sorties, name="segformer_b0")


def construire_modele(
    nom_modele: str,
    nombre_classes: int,
    hauteur: int = 128,
    largeur: int = 256,
    poids_preentraines: bool = True,
    filtres_base: int = 16,
    dropout: float = 0.1,
    projection_filters_segformer: int = 128,
) -> keras.Model:
    if nom_modele == "unet_baseline":
        return construire_unet_baseline(
            hauteur, largeur, nombre_classes, filtres_base=filtres_base, dropout=dropout
        )
    if nom_modele == "unet_mobilenetv2":
        return construire_unet_mobilenetv2(
            hauteur, largeur, nombre_classes, poids_preentraines, dropout
        )
    if nom_modele in {"deeplabv3", "deeplabv3_mobilenetv2"}:
        return construire_deeplabv3_mobilenetv2(
            hauteur, largeur, nombre_classes, poids_preentraines, dropout
        )
    if nom_modele == "segformer":
        return construire_segformer_b0(
            hauteur,
            largeur,
            nombre_classes,
            poids_preentraines,
            projection_filters=projection_filters_segformer,
        )
    raise ValueError(f"Modèle inconnu : {nom_modele}")


@keras.saving.register_keras_serializable(package="FutureVision")
class EntropieCroiseeIgnoreVide(keras.losses.Loss):
    def __init__(self, id_vide: int = 0, name: str = "entropie_croisee_ignore_vide", **kwargs):
        super().__init__(name=name, **kwargs)
        self.id_vide = int(id_vide)

    def call(self, y_true, y_pred):
        y_true = ops.cast(y_true, "int32")
        pertes = ops.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=True, axis=-1
        )
        valide = ops.cast(ops.not_equal(y_true, self.id_vide), pertes.dtype)
        numerateur = ops.sum(pertes * valide)
        denominateur = ops.maximum(ops.sum(valide), ops.cast(1.0, pertes.dtype))
        return numerateur / denominateur

    def get_config(self):
        config = super().get_config()
        config.update({"id_vide": self.id_vide})
        return config


class _MetriqueSegmentationIgnoreVide(keras.metrics.Metric):
    def __init__(self, nombre_classes: int = 8, id_vide: int = 0, name: str = "metrique", **kwargs):
        super().__init__(name=name, **kwargs)
        self.nombre_classes = int(nombre_classes)
        self.id_vide = int(id_vide)
        self.classes = [c for c in range(nombre_classes) if c != id_vide]
        n = len(self.classes)
        self.intersections = self.add_weight(
            name="intersections", shape=(n,), initializer="zeros"
        )
        self.unions = self.add_weight(name="unions", shape=(n,), initializer="zeros")
        self.sommes = self.add_weight(name="sommes", shape=(n,), initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = ops.cast(y_true, "int32")
        y_pred = ops.cast(ops.argmax(y_pred, axis=-1), "int32")
        valide = ops.not_equal(y_true, self.id_vide)
        intersections, unions, sommes = [], [], []
        for classe in self.classes:
            p = ops.logical_and(ops.equal(y_pred, classe), valide)
            c = ops.logical_and(ops.equal(y_true, classe), valide)
            inter = ops.sum(ops.cast(ops.logical_and(p, c), "float32"))
            union = ops.sum(ops.cast(ops.logical_or(p, c), "float32"))
            somme = ops.sum(ops.cast(p, "float32")) + ops.sum(ops.cast(c, "float32"))
            intersections.append(inter)
            unions.append(union)
            sommes.append(somme)
        self.intersections.assign_add(ops.stack(intersections))
        self.unions.assign_add(ops.stack(unions))
        self.sommes.assign_add(ops.stack(sommes))

    def reset_state(self):
        self.intersections.assign(ops.zeros_like(self.intersections))
        self.unions.assign(ops.zeros_like(self.unions))
        self.sommes.assign(ops.zeros_like(self.sommes))

    def get_config(self):
        config = super().get_config()
        config.update(
            {"nombre_classes": self.nombre_classes, "id_vide": self.id_vide}
        )
        return config


@keras.saving.register_keras_serializable(package="FutureVision")
class MeanIoUIgnoreVide(_MetriqueSegmentationIgnoreVide):
    def __init__(self, nombre_classes: int = 8, id_vide: int = 0, name: str = "miou", **kwargs):
        super().__init__(nombre_classes, id_vide, name=name, **kwargs)

    def result(self):
        presentes = ops.greater(self.unions, 0)
        iou = ops.where(
            presentes,
            self.intersections / ops.maximum(self.unions, 1e-7),
            ops.zeros_like(self.unions),
        )
        n = ops.maximum(ops.sum(ops.cast(presentes, "float32")), 1.0)
        return ops.sum(iou) / n


@keras.saving.register_keras_serializable(package="FutureVision")
class DiceIgnoreVide(_MetriqueSegmentationIgnoreVide):
    def __init__(self, nombre_classes: int = 8, id_vide: int = 0, name: str = "dice", **kwargs):
        super().__init__(nombre_classes, id_vide, name=name, **kwargs)

    def result(self):
        presentes = ops.greater(self.sommes, 0)
        dice = ops.where(
            presentes,
            (2.0 * self.intersections) / ops.maximum(self.sommes, 1e-7),
            ops.zeros_like(self.sommes),
        )
        n = ops.maximum(ops.sum(ops.cast(presentes, "float32")), 1.0)
        return ops.sum(dice) / n


@keras.saving.register_keras_serializable(package="FutureVision")
class PrecisionPixelIgnoreVide(keras.metrics.Metric):
    def __init__(self, id_vide: int = 0, name: str = "precision_pixel", **kwargs):
        super().__init__(name=name, **kwargs)
        self.id_vide = int(id_vide)
        self.corrects = self.add_weight(name="corrects", initializer="zeros")
        self.total = self.add_weight(name="total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = ops.cast(y_true, "int32")
        y_pred = ops.cast(ops.argmax(y_pred, axis=-1), "int32")
        valide = ops.not_equal(y_true, self.id_vide)
        correct = ops.logical_and(ops.equal(y_true, y_pred), valide)
        self.corrects.assign_add(ops.sum(ops.cast(correct, "float32")))
        self.total.assign_add(ops.sum(ops.cast(valide, "float32")))

    def result(self):
        return self.corrects / ops.maximum(self.total, 1.0)

    def reset_state(self):
        self.corrects.assign(0.0)
        self.total.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update({"id_vide": self.id_vide})
        return config


def compiler_modele(
    modele: keras.Model,
    nombre_classes: int = 8,
    id_vide: int = 0,
    taux_apprentissage: float = 1e-3,
    weight_decay: float = 1e-4,
    clipnorm: float | None = 1.0,
    jit_compile: bool = False,
) -> keras.Model:
    """Compile le modèle. ``jit_compile=False`` est volontairement prudent sur XPU."""
    optimiseur = keras.optimizers.AdamW(
        learning_rate=taux_apprentissage,
        weight_decay=weight_decay,
        clipnorm=clipnorm,
    )
    modele.compile(
        optimizer=optimiseur,
        loss=EntropieCroiseeIgnoreVide(id_vide=id_vide),
        metrics=[
            MeanIoUIgnoreVide(nombre_classes, id_vide),
            DiceIgnoreVide(nombre_classes, id_vide),
            PrecisionPixelIgnoreVide(id_vide),
        ],
        jit_compile=jit_compile,
    )
    return modele


def evaluer_modele(modele: keras.Model, chargeur: JeuSegmentationKeras, verbose: int = 0) -> dict[str, float]:
    resultat = modele.evaluate(chargeur, return_dict=True, verbose=verbose)
    return {k: float(v) for k, v in resultat.items()}


def _vider_cache_accelerateur() -> None:
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        try:
            torch.xpu.empty_cache()
        except Exception:
            pass
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass




def _definir_hyperparametres_bayesiens(hp, nom_modele: str) -> dict[str, Any]:
    """Définit un espace de recherche raisonnable propre à chaque architecture.

    Les bornes sont volontairement limitées pour un iGPU Intel intégré. Les
    paramètres continus sont échantillonnés en log quand plusieurs ordres de
    grandeur sont pertinents.
    """
    nom = str(nom_modele).lower()
    if nom == "unet_baseline":
        return {
            "taille_lot": hp.Choice("taille_lot", values=[2, 4, 8], default=4),
            "taux_apprentissage": hp.Float(
                "taux_apprentissage", 1e-5, 1e-3, sampling="log", default=3e-4
            ),
            "dropout": hp.Float("dropout", 0.0, 0.30, step=0.05, default=0.10),
            "weight_decay": hp.Float(
                "weight_decay", 1e-6, 1e-3, sampling="log", default=1e-4
            ),
            "filtres_base": hp.Choice(
                "filtres_base", values=[16, 32, 64], default=16
            ),
        }
    if nom == "unet_mobilenetv2":
        return {
            "taille_lot": hp.Choice("taille_lot", values=[2, 4, 8], default=4),
            "taux_apprentissage": hp.Float(
                "taux_apprentissage", 1e-5, 5e-4, sampling="log", default=1e-4
            ),
            "dropout": hp.Float("dropout", 0.0, 0.30, step=0.05, default=0.10),
            "weight_decay": hp.Float(
                "weight_decay", 1e-6, 1e-3, sampling="log", default=1e-4
            ),
        }
    if nom in {"deeplabv3", "deeplabv3_mobilenetv2"}:
        return {
            "taille_lot": hp.Choice("taille_lot", values=[1, 2, 4], default=2),
            "taux_apprentissage": hp.Float(
                "taux_apprentissage", 1e-5, 5e-4, sampling="log", default=1e-4
            ),
            "dropout": hp.Float("dropout", 0.0, 0.30, step=0.05, default=0.10),
            "weight_decay": hp.Float(
                "weight_decay", 1e-6, 1e-3, sampling="log", default=1e-4
            ),
        }
    if nom == "segformer":
        return {
            "taille_lot": hp.Choice("taille_lot", values=[1, 2, 4], default=2),
            "taux_apprentissage": hp.Float(
                "taux_apprentissage", 1e-5, 3e-4, sampling="log", default=6e-5
            ),
            "weight_decay": hp.Float(
                "weight_decay", 1e-6, 1e-3, sampling="log", default=1e-4
            ),
            "projection_filters_segformer": hp.Choice(
                "projection_filters_segformer", values=[64, 128, 256], default=128
            ),
        }
    raise ValueError(f"Modèle inconnu pour le tuning bayésien : {nom_modele}")


_BaseHyperModel = kt.HyperModel if kt is not None else object


class HyperModeleSegmentationBayes(_BaseHyperModel):
    """HyperModel KerasTuner pour modèle + paramètres d'entraînement.

    ``build()`` optimise les paramètres de modèle/optimiseur. ``fit()``
    recrée les ``PyDataset`` avec le ``taille_lot`` du trial. Le jeu de test
    n'est volontairement pas accepté par cette classe.
    """

    def __init__(
        self,
        *,
        nom_modele: str,
        tableau_entrainement: pd.DataFrame,
        tableau_validation: pd.DataFrame,
        largeur: int,
        hauteur: int,
        nombre_classes: int,
        nombre_travailleurs: int = 1,
        donnees_pretraitees: bool = True,
        avec_augmentation_tuning: bool = True,
        utiliser_multiprocessing: bool = False,
        poids_preentraines: bool = False,
        graine: int = 42,
        id_vide: int = 0,
    ) -> None:
        if kt is None:
            raise ImportError(
                "KerasTuner est requis : `poetry add keras-tuner`."
            )
        super().__init__(name=f"{nom_modele}_bayes")
        self.nom_modele = nom_modele
        self.tableau_entrainement = tableau_entrainement
        self.tableau_validation = tableau_validation
        self.largeur = int(largeur)
        self.hauteur = int(hauteur)
        self.nombre_classes = int(nombre_classes)
        self.nombre_travailleurs = int(nombre_travailleurs)
        self.donnees_pretraitees = bool(donnees_pretraitees)
        self.avec_augmentation_tuning = bool(avec_augmentation_tuning)
        self.utiliser_multiprocessing = bool(utiliser_multiprocessing)
        self.poids_preentraines = bool(poids_preentraines)
        self.graine = int(graine)
        self.id_vide = int(id_vide)

    def build(self, hp):
        params = _definir_hyperparametres_bayesiens(hp, self.nom_modele)
        modele = construire_modele(
            self.nom_modele,
            nombre_classes=self.nombre_classes,
            hauteur=self.hauteur,
            largeur=self.largeur,
            poids_preentraines=self.poids_preentraines,
            filtres_base=int(params.get("filtres_base", 16)),
            dropout=float(params.get("dropout", 0.1)),
            projection_filters_segformer=int(
                params.get("projection_filters_segformer", 128)
            ),
        )
        compiler_modele(
            modele,
            nombre_classes=self.nombre_classes,
            id_vide=self.id_vide,
            taux_apprentissage=float(params["taux_apprentissage"]),
            weight_decay=float(params["weight_decay"]),
            jit_compile=False,
        )
        return modele

    def fit(self, hp, model, *args, **kwargs):
        # Le batch size est un hyperparamètre du processus d'entraînement :
        # KerasTuner recommande de le définir dans HyperModel.fit().
        taille_lot = int(hp.get("taille_lot"))
        if self.donnees_pretraitees:
            trans_train = (
                creer_augmentations_pretraitees()
                if self.avec_augmentation_tuning
                else creer_transformation_deterministe_pretraitee()
            )
            trans_val = creer_transformation_deterministe_pretraitee()
        else:
            trans_train = (
                creer_augmentations(self.largeur, self.hauteur)
                if self.avec_augmentation_tuning
                else creer_transformation_deterministe(self.largeur, self.hauteur)
            )
            trans_val = creer_transformation_deterministe(self.largeur, self.hauteur)

        commun = dict(
            taille_lot=taille_lot,
            donnees_pretraitees=self.donnees_pretraitees,
            graine=self.graine,
            nombre_travailleurs=self.nombre_travailleurs,
            utiliser_multiprocessing=self.utiliser_multiprocessing,
        )
        train = JeuSegmentationKeras(
            self.tableau_entrainement,
            trans_train,
            melanger=True,
            **commun,
        )
        val = JeuSegmentationKeras(
            self.tableau_validation,
            trans_val,
            melanger=False,
            **commun,
        )
        return model.fit(
            train,
            validation_data=val,
            *args,
            **kwargs,
        )


def optimiser_hyperparametres_bayesiens(
    *,
    nom_modele: str,
    tableau_entrainement: pd.DataFrame,
    tableau_validation: pd.DataFrame,
    largeur: int,
    hauteur: int,
    nombre_classes: int,
    max_trials: int = 15,
    num_initial_points: int = 5,
    nombre_epoques_tuning: int = 5,
    patience: int = 2,
    nombre_travailleurs: int = 1,
    donnees_pretraitees: bool = True,
    avec_augmentation_tuning: bool = True,
    utiliser_multiprocessing: bool = False,
    poids_preentraines: bool = False,
    graine: int = 42,
    dossier_resultats: Path | str = "tuning_bayesien",
    reprendre: bool = True,
    verbose: int = 1,
):
    """Lance un seul BayesianOptimization train/validation pour une architecture.

    Le test n'est pas un argument de cette fonction. Les meilleurs
    hyperparamètres obtenus doivent ensuite être réutilisés *à l'identique*
    pour les entraînements finaux avec et sans augmentation.

    Retourne ``(tuner, meilleurs_hyperparametres, tableau_trials)``.
    """
    if kt is None:
        raise ImportError("KerasTuner est requis : `poetry add keras-tuner`.")

    hypermodele = HyperModeleSegmentationBayes(
        nom_modele=nom_modele,
        tableau_entrainement=tableau_entrainement,
        tableau_validation=tableau_validation,
        largeur=largeur,
        hauteur=hauteur,
        nombre_classes=nombre_classes,
        nombre_travailleurs=nombre_travailleurs,
        donnees_pretraitees=donnees_pretraitees,
        avec_augmentation_tuning=avec_augmentation_tuning,
        utiliser_multiprocessing=utiliser_multiprocessing,
        poids_preentraines=poids_preentraines,
        graine=graine,
    )

    dossier_resultats = Path(dossier_resultats)
    dossier_resultats.mkdir(parents=True, exist_ok=True)
    tuner = kt.BayesianOptimization(
        hypermodel=hypermodele,
        objective=kt.Objective("val_miou", direction="max"),
        max_trials=int(max_trials),
        num_initial_points=int(num_initial_points),
        seed=int(graine),
        directory=str(dossier_resultats),
        project_name=f"{nom_modele}_bayes",
        overwrite=not bool(reprendre),
        max_retries_per_trial=1,
        max_consecutive_failed_trials=5,
    )
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_miou",
            mode="max",
            patience=int(patience),
            restore_best_weights=True,
            verbose=0,
        )
    ]
    debut = time.perf_counter()
    tuner.search(
        epochs=int(nombre_epoques_tuning),
        callbacks=callbacks,
        verbose=verbose,
    )
    duree = time.perf_counter() - debut

    meilleur_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    meilleurs = dict(meilleur_hp.values)
    meilleur_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
    meilleurs["meilleure_val_miou_tuning"] = (
        float(meilleur_trial.score) if meilleur_trial.score is not None else np.nan
    )
    meilleurs["duree_tuning_secondes"] = float(duree)
    meilleurs["augmentation_pendant_tuning"] = bool(avec_augmentation_tuning)

    lignes = []
    for trial_id, trial in tuner.oracle.trials.items():
        ligne = {
            "trial_id": trial_id,
            "statut": str(trial.status),
            "val_miou": float(trial.score) if trial.score is not None else np.nan,
            "best_step": getattr(trial, "best_step", None),
        }
        ligne.update(trial.hyperparameters.values)
        lignes.append(ligne)
    tableau_trials = pd.DataFrame(lignes)
    if not tableau_trials.empty and "val_miou" in tableau_trials:
        tableau_trials = tableau_trials.sort_values(
            "val_miou", ascending=False, na_position="last"
        ).reset_index(drop=True)

    (dossier_resultats / f"meilleurs_hyperparametres_{nom_modele}.json").write_text(
        json.dumps(meilleurs, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    tableau_trials.to_csv(
        dossier_resultats / f"trials_bayesiens_{nom_modele}.csv", index=False
    )
    return tuner, meilleurs, tableau_trials

def entrainer_modele_mlflow(
    *,
    modele: keras.Model,
    nom_modele: str,
    architecture: str,
    chargeur_entrainement: JeuSegmentationKeras,
    chargeur_validation: JeuSegmentationKeras,
    nombre_classes: int,
    nombre_epoques: int,
    taux_apprentissage: float,
    patience: int,
    dossier_artifacts: Path,
    parametres: dict[str, Any],
    weight_decay: float = 1e-4,
    id_vide: int = 0,
    utiliser_mlflow: bool = True,
    verbose: int = 1,
) -> dict[str, Any]:
    """Entraîne un modèle et sélectionne le checkpoint sur la validation uniquement.

    Le jeu de test n'est volontairement pas accepté par cette fonction. Il doit
    être évalué uniquement après le choix définitif du gagnant sur validation.
    """
    dossier_modele = Path(dossier_artifacts) / nom_modele
    dossier_modele.mkdir(parents=True, exist_ok=True)
    chemin_modele = dossier_modele / "meilleur_modele.keras"
    chemin_historique = dossier_modele / "historique.csv"

    compiler_modele(
        modele,
        nombre_classes=nombre_classes,
        id_vide=id_vide,
        taux_apprentissage=taux_apprentissage,
        weight_decay=weight_decay,
        jit_compile=False,
    )
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_miou",
            mode="max",
            patience=int(patience),
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=chemin_modele,
            monitor="val_miou",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
    ]

    activer_mlflow = bool(utiliser_mlflow and mlflow is not None)
    contexte = (
        mlflow.start_run(run_name=nom_modele)
        if activer_mlflow
        else contextlib.nullcontext(None)
    )
    debut = time.perf_counter()
    with contexte as execution:
        if activer_mlflow:
            mlflow.log_params(
                {
                    **parametres,
                    "architecture": architecture,
                    "backend_keras": keras.backend.backend(),
                    "peripherique": choisir_peripherique(),
                    "weight_decay": weight_decay,
                }
            )

        historique = modele.fit(
            chargeur_entrainement,
            validation_data=chargeur_validation,
            epochs=int(nombre_epoques),
            callbacks=callbacks,
            verbose=verbose,
        )
        duree = time.perf_counter() - debut
        table_historique = pd.DataFrame(historique.history)
        table_historique.index = np.arange(1, len(table_historique) + 1)
        table_historique.index.name = "epoque"
        table_historique.to_csv(chemin_historique)

        meilleur_modele = keras.saving.load_model(chemin_modele)
        scores_val = evaluer_modele(meilleur_modele, chargeur_validation, verbose=0)
        meilleure_epoque = (
            int(np.nanargmax(table_historique["val_miou"].to_numpy()) + 1)
            if "val_miou" in table_historique and len(table_historique)
            else len(table_historique)
        )

        resume: dict[str, Any] = {
            "nom_modele": nom_modele,
            "architecture": architecture,
            "scenario": parametres.get("scenario"),
            "taille_lot": parametres.get("taille_lot"),
            "taux_apprentissage": taux_apprentissage,
            "weight_decay": weight_decay,
            "miou_validation": scores_val.get("miou"),
            "dice_validation": scores_val.get("dice"),
            "perte_validation": scores_val.get("loss"),
            "precision_pixel_validation": scores_val.get("precision_pixel"),
            "temps_entrainement_secondes": float(duree),
            "meilleure_epoque": meilleure_epoque,
            "chemin_modele": str(chemin_modele),
            "backend_keras": keras.backend.backend(),
            "peripherique": choisir_peripherique(),
            "run_id": execution.info.run_id if activer_mlflow and execution is not None else None,
        }

        chemin_resume = dossier_modele / "resume_validation.json"
        chemin_resume.write_text(
            json.dumps(resume, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        if activer_mlflow:
            metriques = {
                k: float(v)
                for k, v in resume.items()
                if isinstance(v, (int, float)) and v is not None
            }
            mlflow.log_metrics(metriques)
            mlflow.log_artifact(str(chemin_historique), artifact_path="historique")
            mlflow.log_artifact(str(chemin_modele), artifact_path="modele_keras")
            mlflow.log_artifact(str(chemin_resume), artifact_path="resume")

    return resume


def journaliser_evaluation_test_mlflow(
    run_id: str | None,
    scores_test: dict[str, float],
    dossier_artifacts: Path,
    nom_modele: str,
) -> Path:
    """Enregistre le test final du gagnant, après sa sélection sur validation."""
    dossier = Path(dossier_artifacts) / nom_modele
    dossier.mkdir(parents=True, exist_ok=True)
    chemin = dossier / "evaluation_test_final.json"
    contenu = {
        "miou_test": scores_test.get("miou"),
        "dice_test": scores_test.get("dice"),
        "perte_test": scores_test.get("loss"),
        "precision_pixel_test": scores_test.get("precision_pixel"),
    }
    chemin.write_text(
        json.dumps(contenu, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if mlflow is not None and run_id:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics(
                {k: float(v) for k, v in contenu.items() if v is not None}
            )
            mlflow.log_artifact(str(chemin), artifact_path="test_final")
    return chemin


def predire_masque(
    modele: keras.Model,
    image_rgb: np.ndarray,
    largeur: int,
    hauteur: int,
) -> np.ndarray:
    """Prédit un masque 0..7 pour une image RGB, utile pour l'API future."""
    image = cv2.resize(image_rgb, (largeur, hauteur), interpolation=cv2.INTER_LINEAR)
    logits = modele.predict(image[None].astype(np.float32), verbose=0)
    return np.asarray(logits).argmax(axis=-1)[0].astype(np.uint8)


def coloriser_masque(masque: np.ndarray) -> np.ndarray:
    return COULEURS_CATEGORIES[np.asarray(masque, dtype=np.int64)]


def recuperer_classement_mlflow(nom_experience: str, dossier_artifacts: Path) -> pd.DataFrame:
    if mlflow is None:
        raise ImportError("MLflow n'est pas installé.")
    experience = mlflow.get_experiment_by_name(nom_experience)
    if experience is None:
        return pd.DataFrame()
    runs = mlflow.search_runs([experience.experiment_id])
    colonnes = {
        "tags.mlflow.runName": "modèle",
        "params.architecture": "architecture",
        "metrics.miou_validation": "mIoU validation",
        "metrics.dice_validation": "Dice validation",
        "metrics.miou_test": "mIoU test",
        "metrics.dice_test": "Dice test",
        "metrics.perte_validation": "Perte validation",
        "metrics.perte_test": "Perte test",
        "metrics.precision_pixel_validation": "Précision pixel validation",
        "metrics.precision_pixel_test": "Précision pixel test",
        "metrics.temps_entrainement_secondes": "Temps entraînement (s)",
        "metrics.meilleure_epoque": "Meilleure époque",
        "params.peripherique": "Périphérique",
        "run_id": "Run MLflow",
    }
    presentes = [c for c in colonnes if c in runs.columns]
    classement = runs[presentes].rename(columns=colonnes)
    if "mIoU validation" in classement:
        ordre = ["mIoU validation"]
        if "Dice validation" in classement:
            ordre.append("Dice validation")
        classement = classement.sort_values(ordre, ascending=False).reset_index(drop=True)
        classement.insert(0, "Rang", np.arange(1, len(classement) + 1))
    dossier_artifacts = Path(dossier_artifacts)
    dossier_artifacts.mkdir(parents=True, exist_ok=True)
    classement.to_csv(dossier_artifacts / "comparaison_modeles.csv", index=False)
    classement.to_json(
        dossier_artifacts / "comparaison_modeles.json",
        orient="records",
        force_ascii=False,
        indent=2,
    )
    if not classement.empty and "mIoU validation" in classement:
        ax = classement.plot.bar(
            x="modèle",
            y="mIoU validation",
            legend=False,
            figsize=(9, 4),
            title="Classement des modèles",
        )
        ax.set_ylabel("mIoU de validation")
        plt.tight_layout()
        plt.savefig(dossier_artifacts / "classement_modeles.png", dpi=160)
        plt.close()
    return classement


__all__ = [
    "NOMS_CATEGORIES",
    "COULEURS_CATEGORIES",
    "CORRESPONDANCE_LABELS",
    "DiagnosticEnvironnement",
    "fixer_aleatoire",
    "choisir_peripherique",
    "detecter_nombre_travailleurs",
    "configurer_threads_opencv",
    "diagnostiquer_environnement",
    "verifier_backend_keras_torch",
    "creer_tableau_donnees",
    "verifier_couples_images_masques",
    "filtrer_couples_valides",
    "convertir_masque_huit_categories",
    "analyser_distribution_categories",
    "separer_donnees_train_val_test",
    "pretraiter_donnees",
    "cache_pretraite_valide",
    "creer_augmentations_pretraitees",
    "creer_transformation_deterministe_pretraitee",
    "creer_chargeurs_trois_splits",
    "creer_chargeurs_train_validation",
    "creer_chargeur_test",
    "creer_deux_scenarios_chargeurs",
    "JeuSegmentationKeras",
    "construire_modele",
    "compiler_modele",
    "evaluer_modele",
    "HyperModeleSegmentationBayes",
    "optimiser_hyperparametres_bayesiens",
    "entrainer_modele_mlflow",
    "journaliser_evaluation_test_mlflow",
    "predire_masque",
    "coloriser_masque",
    "recuperer_classement_mlflow",
]

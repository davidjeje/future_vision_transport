"""Outils PyTorch pour la segmentation sémantique Cityscapes.

Toutes les fonctions métier et variables exposées sont nommées en français.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

NOMS_CATEGORIES = [
    "vide", "plat", "construction", "objet", "nature",
    "ciel", "humain", "vehicule",
]
COULEURS_CATEGORIES = np.array([
    [0, 0, 0], [128, 64, 128], [70, 70, 70], [153, 153, 153],
    [107, 142, 35], [70, 130, 180], [220, 20, 60], [0, 0, 142],
], dtype=np.uint8)

# Regroupement des labelIds Cityscapes en huit catégories métier.
CORRESPONDANCE_LABELS = {
    0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0,
    7: 1, 8: 1, 9: 1, 10: 1,
    11: 2, 12: 2, 13: 2, 14: 2, 15: 2, 16: 2,
    17: 3, 18: 3, 19: 3, 20: 3,
    21: 4, 22: 4,
    23: 5,
    24: 6, 25: 6,
    26: 7, 27: 7, 28: 7, 29: 7, 30: 7, 31: 7, 32: 7, 33: 7,
    -1: 0, 255: 0,
}


def fixer_aleatoire(graine: int = 42) -> None:
    random.seed(graine)
    np.random.seed(graine)
    torch.manual_seed(graine)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed_all(graine)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(graine)


def choisir_peripherique() -> torch.device:
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def creer_tableau_donnees(dossier_images: Path, dossier_masques: Path) -> pd.DataFrame:
    lignes = []
    for chemin_image in sorted(dossier_images.rglob("*_leftImg8bit.png")):
        relatif = chemin_image.relative_to(dossier_images)
        nom_masque = chemin_image.name.replace("_leftImg8bit.png", "_gtFine_labelIds.png")
        chemin_masque = dossier_masques / relatif.parent / nom_masque
        if chemin_masque.exists():
            lignes.append({
                "ville": relatif.parts[0] if relatif.parts else "inconnue",
                "chemin_image": str(chemin_image),
                "chemin_masque": str(chemin_masque),
            })
    return pd.DataFrame(lignes)



def verifier_couples_images_masques(tableau: pd.DataFrame) -> pd.DataFrame:
    """Vérifie la lisibilité et les dimensions de chaque couple image-masque.

    Le tableau retourné conserve les chemins d'origine et ajoute les dimensions,
    un indicateur de lisibilité et un indicateur d'égalité des dimensions.
    """
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

        lignes.append({
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
        })

    return pd.DataFrame(lignes)


def analyser_distribution_categories(
    tableau: pd.DataFrame,
    noms_categories: list[str] | None = None,
) -> pd.DataFrame:
    """Calcule la distribution pixel par pixel après regroupement en 8 catégories."""
    noms = noms_categories or NOMS_CATEGORIES
    comptes = np.zeros(len(noms), dtype=np.int64)
    masques_analyses = 0

    for chemin_masque in tableau["chemin_masque"]:
        masque_brut = cv2.imread(str(chemin_masque), cv2.IMREAD_UNCHANGED)
        if masque_brut is None:
            continue
        if masque_brut.ndim == 3:
            masque_brut = masque_brut[:, :, 0]

        masque_huit_categories = convertir_masque_huit_categories(masque_brut)
        comptes += np.bincount(
            masque_huit_categories.reshape(-1),
            minlength=len(noms),
        )[:len(noms)]
        masques_analyses += 1

    total_pixels = int(comptes.sum())
    proportions = (
        comptes / total_pixels * 100.0
        if total_pixels > 0
        else np.zeros_like(comptes, dtype=float)
    )

    distribution = pd.DataFrame({
        "identifiant_categorie": np.arange(len(noms), dtype=int),
        "categorie": noms,
        "nombre_pixels": comptes,
        "proportion_pixels_pourcent": proportions,
    })
    distribution["masques_analyses"] = masques_analyses

    proportions_non_nulles = distribution.loc[
        distribution["proportion_pixels_pourcent"] > 0,
        "proportion_pixels_pourcent",
    ]
    rapport_desequilibre = (
        float(proportions_non_nulles.max() / proportions_non_nulles.min())
        if len(proportions_non_nulles) > 1
        else 0.0
    )
    distribution.attrs["rapport_desequilibre"] = rapport_desequilibre
    return distribution


def separer_donnees(tableau: pd.DataFrame, proportion_validation: float = 0.2, graine: int = 42):
    if tableau.empty:
        raise ValueError("Aucun couple image-masque n'a été trouvé.")
    melange = tableau.sample(frac=1.0, random_state=graine).reset_index(drop=True)
    nombre_validation = max(1, int(round(len(melange) * proportion_validation)))
    validation = melange.iloc[:nombre_validation].reset_index(drop=True)
    entrainement = melange.iloc[nombre_validation:].reset_index(drop=True)
    return entrainement, validation


def convertir_masque_huit_categories(masque: np.ndarray) -> np.ndarray:
    resultat = np.zeros_like(masque, dtype=np.uint8)
    for identifiant, categorie in CORRESPONDANCE_LABELS.items():
        resultat[masque == identifiant] = categorie
    return resultat




def pretraiter_donnees(
    tableau: pd.DataFrame,
    dossier_sortie: Path,
    largeur: int,
    hauteur: int,
) -> pd.DataFrame:
    """Prétraite une fois les images et masques pour accélérer les époques.

    Les images sont redimensionnées à la taille cible et les masques sont
    réalignés si nécessaire, regroupés en 8 catégories puis redimensionnés
    avec INTER_NEAREST. Un tableau contenant les nouveaux chemins est retourné.
    """
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

        hauteur_image, largeur_image = image.shape[:2]
        if masque_brut.shape[:2] != (hauteur_image, largeur_image):
            masque_brut = cv2.resize(
                masque_brut,
                (largeur_image, hauteur_image),
                interpolation=cv2.INTER_NEAREST,
            )

        masque = convertir_masque_huit_categories(masque_brut)
        image = cv2.resize(
            image,
            (largeur, hauteur),
            interpolation=cv2.INTER_LINEAR,
        )
        masque = cv2.resize(
            masque,
            (largeur, hauteur),
            interpolation=cv2.INTER_NEAREST,
        )

        nom_base = f"{indice:06d}"
        chemin_image_sortie = dossier_images / f"{nom_base}_image.png"
        chemin_masque_sortie = dossier_masques / f"{nom_base}_masque.png"

        cv2.imwrite(
            str(chemin_image_sortie),
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(str(chemin_masque_sortie), masque)

        lignes.append({
            "ville": ligne.get("ville", "inconnue"),
            "chemin_image": str(chemin_image_sortie),
            "chemin_masque": str(chemin_masque_sortie),
        })

    return pd.DataFrame(lignes)


def creer_augmentations_pretraitees() -> A.Compose:
    """Augmentations à la volée sans redimensionnement supplémentaire."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=12, border_mode=cv2.BORDER_REFLECT_101, p=0.35),
        A.ShiftScaleRotate(
            shift_limit=0.06,
            scale_limit=0.12,
            rotate_limit=0,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.35,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.18,
            contrast_limit=0.18,
            p=0.35,
        ),
        A.GaussNoise(std_range=(0.02, 0.08), p=0.20),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5)),
            A.MotionBlur(blur_limit=5),
        ], p=0.20),
    ])


def creer_augmentation_validation_pretraitee() -> A.Compose:
    """Validation des données déjà redimensionnées : aucune transformation fixe."""
    return A.Compose([])


def creer_augmentations(largeur: int, hauteur: int) -> A.Compose:
    """Pipeline avec au moins cinq familles d'augmentations image+masque."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=12, border_mode=cv2.BORDER_REFLECT_101, p=0.35),
        A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.12, rotate_limit=0,
                           border_mode=cv2.BORDER_REFLECT_101, p=0.35),
        A.RandomBrightnessContrast(brightness_limit=0.18, contrast_limit=0.18, p=0.35),
        A.GaussNoise(std_range=(0.02, 0.08), p=0.20),
        A.OneOf([A.GaussianBlur(blur_limit=(3, 5)), A.MotionBlur(blur_limit=5)], p=0.20),
        A.Resize(height=hauteur, width=largeur),
    ])


def creer_augmentation_validation(largeur: int, hauteur: int) -> A.Compose:
    return A.Compose([A.Resize(height=hauteur, width=largeur)])


class JeuSegmentation(Dataset):
    def __init__(self, tableau: pd.DataFrame, transformations: A.Compose, donnees_pretraitees: bool = False):
        self.tableau = tableau.reset_index(drop=True)
        self.transformations = transformations
        self.donnees_pretraitees = donnees_pretraitees

    def __len__(self):
        return len(self.tableau)

    def __getitem__(self, indice: int):
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
            hauteur_image, largeur_image = image.shape[:2]
            hauteur_masque, largeur_masque = masque_brut.shape[:2]
            if (hauteur_image, largeur_image) != (hauteur_masque, largeur_masque):
                masque_brut = cv2.resize(
                    masque_brut,
                    (largeur_image, hauteur_image),
                    interpolation=cv2.INTER_NEAREST,
                )
            masque = convertir_masque_huit_categories(masque_brut)

        transforme = self.transformations(image=image, mask=masque)
        image = transforme["image"].astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        masque = transforme["mask"].astype(np.int64)

        return (
            torch.from_numpy(np.ascontiguousarray(image)),
            torch.from_numpy(np.ascontiguousarray(masque)),
        )


def creer_chargeurs(
    tableau_entrainement: pd.DataFrame,
    tableau_validation: pd.DataFrame,
    largeur: int,
    hauteur: int,
    taille_lot: int,
    nombre_travailleurs: int = 2,
    donnees_pretraitees: bool = False,
):
    if donnees_pretraitees:
        transformations_entrainement = creer_augmentations_pretraitees()
        transformations_validation = creer_augmentation_validation_pretraitee()
    else:
        transformations_entrainement = creer_augmentations(largeur, hauteur)
        transformations_validation = creer_augmentation_validation(largeur, hauteur)

    jeu_entrainement = JeuSegmentation(
        tableau_entrainement,
        transformations_entrainement,
        donnees_pretraitees=donnees_pretraitees,
    )
    jeu_validation = JeuSegmentation(
        tableau_validation,
        transformations_validation,
        donnees_pretraitees=donnees_pretraitees,
    )

    chargeur_entrainement = DataLoader(
        jeu_entrainement,
        batch_size=taille_lot,
        shuffle=True,
        num_workers=nombre_travailleurs,
        persistent_workers=nombre_travailleurs > 0,
        drop_last=True,
    )
    chargeur_validation = DataLoader(
        jeu_validation,
        batch_size=taille_lot,
        shuffle=False,
        num_workers=nombre_travailleurs,
        persistent_workers=nombre_travailleurs > 0,
    )
    return chargeur_entrainement, chargeur_validation


class BlocDoubleConvolution(nn.Module):
    def __init__(self, entree: int, sortie: int):
        super().__init__()
        self.bloc = nn.Sequential(
            nn.Conv2d(entree, sortie, 3, padding=1, bias=False), nn.BatchNorm2d(sortie), nn.ReLU(inplace=True),
            nn.Conv2d(sortie, sortie, 3, padding=1, bias=False), nn.BatchNorm2d(sortie), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.bloc(x)


class UNetBaseline(nn.Module):
    def __init__(self, nombre_classes: int = 8, filtres: int = 16):
        super().__init__()
        self.enc1 = BlocDoubleConvolution(3, filtres)
        self.enc2 = BlocDoubleConvolution(filtres, filtres*2)
        self.enc3 = BlocDoubleConvolution(filtres*2, filtres*4)
        self.enc4 = BlocDoubleConvolution(filtres*4, filtres*8)
        self.pool = nn.MaxPool2d(2)
        self.centre = BlocDoubleConvolution(filtres*8, filtres*16)
        self.up4 = nn.ConvTranspose2d(filtres*16, filtres*8, 2, 2)
        self.dec4 = BlocDoubleConvolution(filtres*16, filtres*8)
        self.up3 = nn.ConvTranspose2d(filtres*8, filtres*4, 2, 2)
        self.dec3 = BlocDoubleConvolution(filtres*8, filtres*4)
        self.up2 = nn.ConvTranspose2d(filtres*4, filtres*2, 2, 2)
        self.dec2 = BlocDoubleConvolution(filtres*4, filtres*2)
        self.up1 = nn.ConvTranspose2d(filtres*2, filtres, 2, 2)
        self.dec1 = BlocDoubleConvolution(filtres*2, filtres)
        self.sortie = nn.Conv2d(filtres, nombre_classes, 1)
    def forward(self, x):
        e1=self.enc1(x); e2=self.enc2(self.pool(e1)); e3=self.enc3(self.pool(e2)); e4=self.enc4(self.pool(e3))
        c=self.centre(self.pool(e4))
        d4=self.dec4(torch.cat([self.up4(c),e4],1)); d3=self.dec3(torch.cat([self.up3(d4),e3],1))
        d2=self.dec2(torch.cat([self.up2(d3),e2],1)); d1=self.dec1(torch.cat([self.up1(d2),e1],1))
        return self.sortie(d1)


def construire_modele(nom_modele: str, nombre_classes: int, poids_preentraines: bool = True):
    if nom_modele == "unet_baseline":
        return UNetBaseline(nombre_classes=nombre_classes)
    if nom_modele == "unet_mobilenetv2":
        import segmentation_models_pytorch as smp
        return smp.Unet(encoder_name="mobilenet_v2", encoder_weights="imagenet" if poids_preentraines else None,
                        in_channels=3, classes=nombre_classes)
    if nom_modele == "deeplabv3":
        import segmentation_models_pytorch as smp
        return smp.DeepLabV3(encoder_name="mobilenet_v2", encoder_weights="imagenet" if poids_preentraines else None,
                             in_channels=3, classes=nombre_classes)
    if nom_modele == "segformer":
        from transformers import SegformerForSemanticSegmentation
        identifiant = "nvidia/mit-b0"
        modele = SegformerForSemanticSegmentation.from_pretrained(
            identifiant if poids_preentraines else None,
            num_labels=nombre_classes,
            ignore_mismatched_sizes=True,
        ) if poids_preentraines else SegformerForSemanticSegmentation.from_config(
            __import__('transformers').SegformerConfig(num_labels=nombre_classes)
        )
        return modele
    raise ValueError(f"Modèle inconnu : {nom_modele}")


def extraire_logits(sortie, taille_cible):
    logits = sortie.logits if hasattr(sortie, "logits") else sortie
    if logits.shape[-2:] != taille_cible:
        logits = F.interpolate(logits, size=taille_cible, mode="bilinear", align_corners=False)
    return logits


def calculer_scores(prediction: torch.Tensor, cible: torch.Tensor, nombre_classes: int, id_vide: int = 0):
    prediction = prediction.reshape(-1)
    cible = cible.reshape(-1)
    masque_valide = cible != id_vide
    prediction, cible = prediction[masque_valide], cible[masque_valide]
    intersections, unions, dices = [], [], []
    for classe in range(nombre_classes):
        if classe == id_vide: continue
        p, c = prediction == classe, cible == classe
        intersection = (p & c).sum().item()
        union = (p | c).sum().item()
        somme = p.sum().item() + c.sum().item()
        if union > 0: intersections.append(intersection / union)
        if somme > 0: dices.append(2 * intersection / somme)
    precision = (prediction == cible).float().mean().item() if cible.numel() else 0.0
    return float(np.mean(intersections) if intersections else 0), float(np.mean(dices) if dices else 0), precision


def executer_epoque(
    modele,
    chargeur,
    fonction_perte,
    peripherique,
    nombre_classes,
    optimiseur=None,
    numero_epoque=None,
):
    from tqdm.auto import tqdm

    entrainement = optimiseur is not None
    modele.train(entrainement)

    pertes = []
    miou = []
    dice = []
    precision = []

    contexte = (
        torch.enable_grad()
        if entrainement
        else torch.no_grad()
    )

    type_phase = (
        "Entraînement"
        if entrainement
        else "Validation"
    )

    description = type_phase

    if numero_epoque is not None:
        description = (
            f"Époque {numero_epoque} - {type_phase}"
        )

    barre_progression = tqdm(
        chargeur,
        desc=description,
        total=len(chargeur),
        leave=False,
    )

    with contexte:
        for images, masques in barre_progression:

            # Transfert vers le périphérique
            images = images.to(peripherique)
            masques = masques.to(peripherique)

            # Réinitialisation des gradients
            if entrainement:
                optimiseur.zero_grad(
                    set_to_none=True
                )

            # Forward
            sortie = modele(images)

            logits = extraire_logits(
                sortie,
                masques.shape[-2:],
            )

            # Fonction de perte
            perte = fonction_perte(
                logits,
                masques,
            )

            # Backpropagation uniquement
            # pendant l'entraînement
            if entrainement:
                perte.backward()
                optimiseur.step()

            # Prédiction des classes
            predictions = logits.argmax(dim=1)

            # Calcul des métriques
            (
                s_miou,
                s_dice,
                s_precision,
            ) = calculer_scores(
                predictions.detach(),
                masques,
                nombre_classes,
            )

            # Sauvegarde des métriques
            pertes.append(
                perte.item()
            )

            miou.append(
                s_miou
            )

            dice.append(
                s_dice
            )

            precision.append(
                s_precision
            )

            # Affichage dynamique dans tqdm
            barre_progression.set_postfix(
                perte=f"{perte.item():.4f}",
                miou=f"{s_miou:.4f}",
                dice=f"{s_dice:.4f}",
            )

    return {
        "perte": float(
            np.mean(pertes)
        ),
        "miou": float(
            np.mean(miou)
        ),
        "dice": float(
            np.mean(dice)
        ),
        "precision_pixel": float(
            np.mean(precision)
        ),
    }

def entrainer_modele_mlflow(
    *,
    modele,
    nom_modele: str,
    architecture: str,
    chargeur_entrainement,
    chargeur_validation,
    peripherique,
    nombre_classes: int,
    nombre_epoques: int,
    taux_apprentissage: float,
    patience: int,
    dossier_artifacts: Path,
    parametres: dict,
):
    dossier_modele = (
        dossier_artifacts / nom_modele
    )

    dossier_modele.mkdir(
        parents=True,
        exist_ok=True,
    )

    modele = modele.to(peripherique)

    fonction_perte = nn.CrossEntropyLoss(
        ignore_index=0
    )

    optimiseur = torch.optim.AdamW(
        modele.parameters(),
        lr=taux_apprentissage,
    )

    historique = []
    meilleure_miou = -1.0
    attentes = 0

    chemin_poids = (
        dossier_modele
        / "meilleurs_poids.pt"
    )

    debut = time.perf_counter()

    with mlflow.start_run(
        run_name=nom_modele
    ) as execution:

        mlflow.log_params({
            **parametres,
            "architecture": architecture,
            "peripherique": str(peripherique),
        })

        for epoque in range(
            1,
            nombre_epoques + 1,
        ):

            print(
                f"\n{'=' * 60}\n"
                f"{nom_modele} — "
                f"Époque {epoque}/{nombre_epoques}"
            )

            # -------------------------
            # Entraînement
            # -------------------------

            train = executer_epoque(
                modele,
                chargeur_entrainement,
                fonction_perte,
                peripherique,
                nombre_classes,
                optimiseur,
                numero_epoque=epoque,
            )

            # -------------------------
            # Validation
            # -------------------------

            val = executer_epoque(
                modele,
                chargeur_validation,
                fonction_perte,
                peripherique,
                nombre_classes,
                numero_epoque=epoque,
            )

            ligne = {
                "epoque": epoque,
                **{
                    f"entrainement_{k}": v
                    for k, v
                    in train.items()
                },
                **{
                    f"validation_{k}": v
                    for k, v
                    in val.items()
                },
            }

            historique.append(ligne)

            # -------------------------
            # MLflow
            # -------------------------

            mlflow.log_metrics(
                {
                    k: v
                    for k, v
                    in ligne.items()
                    if k != "epoque"
                },
                step=epoque,
            )

            # -------------------------
            # Résumé de l'époque
            # -------------------------

            print(
                f"{nom_modele} | "
                f"époque {epoque:02d} | "
                f"perte val={val['perte']:.4f} | "
                f"mIoU={val['miou']:.4f} | "
                f"Dice={val['dice']:.4f}"
            )

            # -------------------------
            # Meilleur modèle
            # -------------------------

            if val["miou"] > meilleure_miou:

                meilleure_miou = val["miou"]
                attentes = 0

                torch.save(
                    modele.state_dict(),
                    chemin_poids,
                )

                print(
                    "✓ Nouvelle meilleure mIoU : "
                    f"{meilleure_miou:.4f}"
                )

            else:

                attentes += 1

                print(
                    "Pas d'amélioration de la mIoU "
                    f"({attentes}/{patience})"
                )

                # -------------------------
                # Early stopping
                # -------------------------

                if attentes >= patience:
                    print(
                        f"\nEarly stopping : "
                        f"aucune amélioration pendant "
                        f"{patience} époques."
                    )

                    break

        # -----------------------------
        # Résultats finaux
        # -----------------------------

        duree = (
            time.perf_counter()
            - debut
        )

        tableau = pd.DataFrame(
            historique
        )

        chemin_historique = (
            dossier_modele
            / "historique.csv"
        )

        tableau.to_csv(
            chemin_historique,
            index=False,
        )

        meilleure = (
            tableau
            .sort_values(
                "validation_miou",
                ascending=False,
            )
            .iloc[0]
        )

        resume = {
            "nom_modele": nom_modele,
            "architecture": architecture,

            "miou_validation":
                float(
                    meilleure.validation_miou
                ),

            "dice_validation":
                float(
                    meilleure.validation_dice
                ),

            "perte_validation":
                float(
                    meilleure.validation_perte
                ),

            "precision_pixel_validation":
                float(
                    meilleure
                    .validation_precision_pixel
                ),

            "temps_entrainement_secondes":
                float(duree),

            "meilleure_epoque":
                int(meilleure.epoque),

            "run_id":
                execution.info.run_id,

            "chemin_poids":
                str(chemin_poids),

            "peripherique":
                str(peripherique),
        }

        # -----------------------------
        # Enregistrement MLflow
        # -----------------------------

        mlflow.log_metrics({
            k: v
            for k, v
            in resume.items()
            if isinstance(
                v,
                (int, float),
            )
        })

        mlflow.log_artifact(
            str(chemin_historique),
            artifact_path="historique",
        )

        mlflow.log_artifact(
            str(chemin_poids),
            artifact_path="poids",
        )

        chemin_resume = (
            dossier_modele
            / "resume.json"
        )

        chemin_resume.write_text(
            json.dumps(
                resume,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        mlflow.log_artifact(
            str(chemin_resume),
            artifact_path="resume",
        )

    return resume


def recuperer_classement_mlflow(nom_experience: str, dossier_artifacts: Path) -> pd.DataFrame:
    experience = mlflow.get_experiment_by_name(nom_experience)
    if experience is None:
        return pd.DataFrame()
    runs = mlflow.search_runs([experience.experiment_id])
    colonnes = {
        "tags.mlflow.runName":"modèle", "params.architecture":"architecture",
        "metrics.miou_validation":"mIoU validation", "metrics.dice_validation":"Dice validation",
        "metrics.perte_validation":"Perte validation", "metrics.precision_pixel_validation":"Précision pixel",
        "metrics.temps_entrainement_secondes":"Temps entraînement (s)",
        "metrics.meilleure_epoque":"Meilleure époque", "params.peripherique":"Périphérique",
        "run_id":"Run MLflow",
    }
    presentes = [c for c in colonnes if c in runs.columns]
    classement = runs[presentes].rename(columns=colonnes)
    if "mIoU validation" in classement:
        classement = classement.sort_values(["mIoU validation", "Dice validation"], ascending=False).reset_index(drop=True)
        classement.insert(0, "Rang", np.arange(1, len(classement)+1))
    dossier_artifacts.mkdir(parents=True, exist_ok=True)
    classement.to_csv(dossier_artifacts / "comparaison_modeles.csv", index=False)
    classement.to_json(dossier_artifacts / "comparaison_modeles.json", orient="records", force_ascii=False, indent=2)
    if not classement.empty and "mIoU validation" in classement:
        ax = classement.plot.bar(x="modèle", y="mIoU validation", legend=False, figsize=(9,4), title="Classement des modèles")
        ax.set_ylabel("mIoU de validation"); plt.tight_layout(); plt.savefig(dossier_artifacts / "classement_modeles.png", dpi=160); plt.close()
    return classement

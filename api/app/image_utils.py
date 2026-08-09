from io import BytesIO

import cv2
import numpy as np
import torch
from PIL import Image


COULEURS_CATEGORIES = np.array(
    [
        [0, 0, 0],        # vide
        [128, 64, 128],   # plat
        [70, 70, 70],     # construction
        [153, 153, 153],  # objet
        [107, 142, 35],   # nature
        [70, 130, 180],   # ciel
        [220, 20, 60],    # humain
        [0, 0, 142],      # vehicule
    ],
    dtype=np.uint8,
)


def decode_image(data: bytes) -> Image.Image:
    """
    Convertit les bytes reçus par l'API
    en image PIL RGB.
    """
    image = Image.open(BytesIO(data))
    image.load()

    return image.convert("RGB")


def image_to_tensor(
    image: Image.Image,
    largeur: int = 256,
    hauteur: int = 128,
) -> torch.Tensor:
    """
    Reproduit le preprocessing utilisé
    pendant l'entraînement.

    RGB
    -> resize
    -> float32 / 255
    -> HWC vers CHW
    -> ajout batch
    """

    array = np.asarray(image)

    array = cv2.resize(
        array,
        (largeur, hauteur),
        interpolation=cv2.INTER_LINEAR,
    )

    array = (
        array.astype(np.float32)
        / 255.0
    )

    # HWC -> CHW
    array = np.transpose(
        array,
        (2, 0, 1),
    )

    # CHW -> BCHW
    array = np.expand_dims(
        array,
        axis=0,
    )

    return torch.from_numpy(
        np.ascontiguousarray(array)
    )


def prediction_to_mask(
    prediction,
    original_size: tuple[int, int],
) -> Image.Image:
    """
    Transforme la prédiction du réseau
    en masque contenant les IDs 0 à 7.
    """

    prediction = np.asarray(prediction)

    # Retire la dimension batch
    # (1, H, W) -> (H, W)
    if (
        prediction.ndim == 3
        and prediction.shape[0] == 1
    ):
        prediction = prediction[0]

    prediction = np.squeeze(
        prediction
    )

    if prediction.ndim != 2:
        raise ValueError(
            "La prédiction doit être un masque 2D. "
            f"Shape reçue : {prediction.shape}"
        )

    mask = prediction.astype(
        np.uint8
    )

    image_mask = Image.fromarray(
        mask,
        mode="L",
    )

    # Remet le masque à la taille
    # de l'image envoyée à l'API
    if image_mask.size != original_size:
        image_mask = image_mask.resize(
            original_size,
            resample=Image.Resampling.NEAREST,
        )

    return image_mask


def colorize_mask(
    prediction,
    original_size: tuple[int, int],
) -> Image.Image:
    """
    Transforme les IDs 0-7
    en masque RGB coloré.
    """

    prediction = np.asarray(prediction)

    if (
        prediction.ndim == 3
        and prediction.shape[0] == 1
    ):
        prediction = prediction[0]

    prediction = np.squeeze(
        prediction
    ).astype(np.uint8)

    if prediction.ndim != 2:
        raise ValueError(
            "La prédiction doit être un masque 2D. "
            f"Shape reçue : {prediction.shape}"
        )

    prediction = np.clip(
        prediction,
        0,
        len(COULEURS_CATEGORIES) - 1,
    )

    mask_rgb = COULEURS_CATEGORIES[
        prediction
    ]

    image_mask = Image.fromarray(
        mask_rgb,
        mode="RGB",
    )

    if image_mask.size != original_size:
        image_mask = image_mask.resize(
            original_size,
            resample=Image.Resampling.NEAREST,
        )

    return image_mask


def encode_png(
    image: Image.Image,
) -> bytes:
    """
    Convertit une image PIL en PNG
    sous forme de bytes.
    """

    buffer = BytesIO()

    image.save(
        buffer,
        format="PNG",
    )

    return buffer.getvalue()
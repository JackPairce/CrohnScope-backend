"""
Service for handling image and mask patch operations.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
from sqlalchemy.orm import Session
from tqdm import tqdm
from API.db.models import Image, Mask, Patch
from API.db.session import SessionLocal
from API.services.image import region_service
from API.services.image.preprocess_service import DataAugmentation, normalizeStaining
import math


# compute optimal stride to cover the entire image
def compute_stride(image_dim: int, patch_dim: int) -> Tuple[int, int]:
    n_patches = math.ceil(image_dim / patch_dim)
    stride = (
        math.floor((image_dim - patch_dim) / (n_patches - 1))
        if n_patches > 1
        else patch_dim
    )
    return stride, n_patches


def pad_to_size(img, target_size=256):
    """Pad the image to the target size."""
    h, w = img.shape[:2]
    c = img.shape[2] if len(img.shape) == 3 else 1
    top = (target_size - h) // 2
    bottom = target_size - h - top
    left = (target_size - w) // 2 if w < target_size else 0
    right = target_size - w - left if w < target_size else 0
    return cv2.copyMakeBorder(
        img,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=0 if c == 1 else [0, 0, 0],
    )


def split_image_into_patches(
    image: np.ndarray, patch_size: int = 256
) -> Tuple[
    List[np.ndarray], List[Tuple[int, int]], Tuple[int, int], Tuple[float, float]
]:
    if image.shape[0] < patch_size or image.shape[1] < patch_size:
        image = pad_to_size(image, patch_size)
    image_height, image_width = image.shape[:2]
    stride_x, n_patches_x = compute_stride(image_width, patch_size)
    stride_y, n_patches_y = compute_stride(image_height, patch_size)

    patches = []
    positions = []
    for i in range(0, image_height - patch_size + 1, stride_y):
        for j in range(0, image_width - patch_size + 1, stride_x):
            patch = image[i : i + patch_size, j : j + patch_size]
            patches.append(patch)
            positions.append((i, j))
    return patches, positions, (stride_x, stride_y), (n_patches_x, n_patches_y)


def save_image_as_patches(
    image: Image,
    patch_size: int = 256,
):
    """Save image and mask patches to disk.

    Args:
        image: Input image (2D array)
        patch_size: Size of patches (default: 128)

    """
    # get masks from db
    session: Session = SessionLocal()
    db_masks = session.query(Mask).filter_by(image_id=image.id).all()

    if db_masks is None or len(db_masks) == 0:
        raise ValueError(f"No masks found for image ID {image.id}")

    image_array = cv2.imread(image.img_path)
    if image_array is None:
        raise ValueError(f"Image at {image.img_path} could not be read")

    normalized_image, _, _ = normalizeStaining(image_array)

    patched_images, *_ = split_image_into_patches(normalized_image, patch_size)

    loaded_masks = np.stack([np.load(mask.mask_path) for mask in db_masks])
    patched_masks = [
        patched_masks
        for patched_masks, *_ in [
            split_image_into_patches(loaded_mask, patch_size)
            for loaded_mask in loaded_masks
        ]
    ]
    patched_images = np.array(patched_images)
    patched_masks = np.array(patched_masks).transpose(1, 0, 2, 3)

    if len(patched_images) != len(patched_masks):
        raise ValueError(
            "Number of image patches does not match number of mask patches.\n"
            f"Image patches: {len(patched_images)}, Mask patches: {len(patched_masks)}\n"
            f"Image Patch Shape: {patched_images.shape}, "
            f"Mask Patch Shape: {patched_masks.shape}"
        )
    # delete existing patches for this image
    session.query(Patch).filter_by(image_id=image.id).delete()
    session.commit()

    for img_patch, mask_patch in zip(patched_images, patched_masks):
        # Save patch in database (delete existing patch if exists)
        for agm_img_patch, agm_mask_patch in zip(
            DataAugmentation(img_patch), DataAugmentation(mask_patch)
        ):
            session.add(
                Patch(
                    image_id=image.id,
                    img_patch=agm_img_patch,
                    mask_patch=agm_mask_patch,
                )
            )
    session.commit()


def load_patches(
    patch_names: List[str], patches_dir: Path
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Load image and mask patches from disk.

    Args:
        patch_names: List of patch names to load
        patches_dir: Base directory containing the patches

    Returns:
        List of (image_patch, mask_patch) tuples
    """
    patches = []
    for patch_name in patch_names:
        img_patch = np.load(str(patches_dir / "images" / f"{patch_name}.npy"))
        mask_patch = np.load(str(patches_dir / "masks" / f"{patch_name}.npy"))
        patches.append((img_patch, mask_patch))
    return patches

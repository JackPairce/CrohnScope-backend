"""
Data preprocessing module for AI training in CrohnScope

This module handles loading, preprocessing and patch-based dataset creation
for training medical image segmentation models.
"""

import os
import torch
import numpy as np
import psutil
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from pathlib import Path
from app.db.session import SessionLocal
from app.db.models import Image as DbImage, Mask, Cell
import gc


class PatchDataset(Dataset):
    """Dataset for patch-based segmentation."""

    def __init__(self, img_patches, mask_patches, transform=None):
        self.img_patches = img_patches
        self.mask_patches = mask_patches
        self.transform = transform

    def __len__(self):
        return len(self.img_patches)

    def __getitem__(self, idx):
        img = self.img_patches[idx]
        mask = self.mask_patches[idx]

        if self.transform:
            img = self.transform(img)
            # Make sure mask is binary
            mask = torch.from_numpy(mask).float().unsqueeze(0)

        return img, mask


def create_patches(img, mask, patch_size=256, stride=128):
    """
    Split image and corresponding mask into patches.

    Args:
        img: PIL Image or numpy array
        mask: PIL Image or numpy array (binary)
        patch_size: Size of the square patches
        stride: Step size between patches

    Returns:
        img_patches, mask_patches: Lists of image and mask patches
    """
    if isinstance(img, Image.Image):
        img = np.array(img)
    if isinstance(mask, Image.Image):
        mask = np.array(mask)

    # Convert RGB mask to binary if needed
    if len(mask.shape) > 2 and mask.shape[2] > 1:
        mask = (mask.sum(axis=2) > 0).astype(np.float32)
    elif len(mask.shape) == 2:
        mask = mask.astype(np.float32)

    h, w = img.shape[:2]
    img_patches = []
    mask_patches = []

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            img_patch = img[y : y + patch_size, x : x + patch_size]
            mask_patch = mask[y : y + patch_size, x : x + patch_size]

            # Skip patches with little information
            if mask_patch.sum() > 20:  # At least some pixels in the mask
                img_patches.append(img_patch)
                mask_patches.append(mask_patch)

    return img_patches, mask_patches


def load_and_preprocess_data(batch_size=8):
    """
    Load images and masks from the database and create a DataLoader.
    Only processes images that have all masks marked as done.

    Args:
        batch_size: Batch size for the DataLoader

    Returns:
        dataloader: PyTorch DataLoader object
    """
    from app.api.routes.image import get_images_by_done_status
    from app.services.ai.train import training_status, status_lock, check_memory

    # Check available memory before processing
    has_memory, _ = check_memory(min_required_mb=500)
    if not has_memory:
        return None

    # Update status to preprocessing
    with status_lock:
        training_status["is_preprocessing"] = True
        training_status["preprocessing_progress"] = 0.0

    session = SessionLocal()

    try:
        # Get only images with all masks done (done=1)
        done_images, total_count = get_images_by_done_status(session, done=1)

        if not done_images:
            print("No images with completed masks available for training")
            with status_lock:
                training_status["is_preprocessing"] = False
            return None

        all_img_patches = []
        all_mask_patches = []

        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Process each image and its masks
        for i, img_record in enumerate(done_images):
            # Update preprocessing progress
            with status_lock:
                training_status["preprocessing_progress"] = (i / len(done_images)) * 100

            # Load image
            img_path = img_record.img_path
            if not os.path.exists(img_path):
                print(f"Warning: Image file not found - {img_path}")
                continue

            # Get all masks for this image
            masks = session.query(Mask).filter_by(image_id=img_record.id).all()
            if not masks:
                continue

            # Check if all masks are done
            all_masks_done = all(mask.is_mask_done == 1 for mask in masks)
            if not all_masks_done:
                continue

            # Load the image
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                continue

            # Process each mask for this image
            for mask_record in masks:
                if not os.path.exists(mask_record.mask_path):
                    continue

                try:
                    mask = Image.open(mask_record.mask_path).convert(
                        "L"
                    )  # Convert to grayscale
                except Exception as e:
                    print(f"Error loading mask {mask_record.mask_path}: {e}")
                    continue

                # Check memory before creating patches
                has_memory, _ = check_memory(min_required_mb=500)
                if not has_memory:
                    print(f"FATAL: Memory running low during preprocessing. Stopping.")
                    with status_lock:
                        training_status["is_preprocessing"] = False
                    return None

                # Create patches
                img_patches, mask_patches = create_patches(img, mask)

                if img_patches:  # Only add if we have valid patches
                    all_img_patches.extend(img_patches)
                    all_mask_patches.extend(mask_patches)

            # Free memory
            del img
            gc.collect()

        # Complete preprocessing
        with status_lock:
            training_status["preprocessing_progress"] = 100.0
            training_status["is_preprocessing"] = False

        # Create dataset and dataloader
        if all_img_patches:
            dataset = PatchDataset(all_img_patches, all_mask_patches, transform)
            dataloader = DataLoader(
                dataset, batch_size=batch_size, shuffle=True, num_workers=4
            )
            return dataloader
        else:
            print("No valid patches created from available images")
            return None

    finally:
        session.close()

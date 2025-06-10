"""
Auto-masking module for generating masks using the AI model

This module provides functionality to automatically generate masks for images
using the trained segmentation model.
"""

import os
from pathlib import Path
from typing import List

import cv2
from numpy import ndarray
from API.db.models import Image, Cell, Mask
from API.db.session import SessionLocal
from API.services.image import image_service
from API.services.image import mask_service
from API.services.image.mask_service import save_masks
from shared.models.Unet.predict import predict_mask
from shared.types.image import MaskArray, MaskSaveRequest


def generate_mask_for_image(image_id: int) -> List[MaskArray]:
    """
    Generate a mask for an image using the AI model.

    Args:
        image_id: ID of the image to generate the mask for

    Returns:
    """
    session = SessionLocal()
    try:
        # Get the image
        db_image = session.query(Image).filter(Image.id == image_id).first()
        if not db_image:
            raise ValueError(f"Image with ID {image_id} not found")

        image = cv2.imread(db_image.img_path)
        if image is None:
            raise ValueError(f"Image at {db_image.img_path} could not be read")

        # predict the mask using the AI model
        return predict_mask(image)

    except Exception as e:
        session.rollback()
        raise e

    finally:
        session.close()


GENERATE_ALL_MASKS_STATE = ""


def generate_masks_for_new_images():
    """
    Generate masks for all images that don't have masks yet.

    Returns:
        int: Number of masks generated
    """
    global GENERATE_ALL_MASKS_STATE
    session = SessionLocal()
    count = 0
    try:
        for page in range(1, 100):
            # Get all images without masks
            images_without_masks, _ = (
                image_service.get_images_by_done_status(
                    session,
                    done=0,
                    which="segmentation",
                    offset=(page - 1) * 10,
                    limit=10,
                )
                # session.query(Image)
                # .filter(~Image.id.in_(session.query(Mask.image_id).distinct()))
                # .all()
            )

            if not images_without_masks:
                print("No images without masks found.")
                return count

            # Generate masks for each image
            for image in images_without_masks:
                outputs = generate_mask_for_image(image.id)
                masks_ids = (
                    session.query(Mask.id, Mask.cell_id)
                    .filter(Mask.image_id == image.id)
                    .order_by(Mask.cell_id)
                    .all()
                )
                if len(outputs) == 0:
                    print(f"No masks generated for image ID {image.id}")
                    continue
                # save the generated masks
                save_masks(
                    session,
                    image.id,
                    [
                        MaskSaveRequest(id=masks_id, cell_id=cell_id, data=output)
                        for (masks_id, cell_id), output in zip(masks_ids, outputs)
                    ],
                )
                count += 1
                GENERATE_ALL_MASKS_STATE = (
                    f"{count}/{len(images_without_masks)} masks generated"
                )

    finally:
        session.close()

    return count

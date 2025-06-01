"""
Auto-masking module for generating masks using the AI model

This module provides functionality to automatically generate masks for images
using the trained segmentation model.
"""

import os
from pathlib import Path
from API.services.ai.predict import predict_mask
from API.db.models import Image, Cell, Mask
from API.db.session import SessionLocal


def generate_mask_for_image(image_id, cell_id):
    """
    Generate a mask for an image using the AI model.

    Args:
        image_id: ID of the image to generate the mask for
        cell_id: ID of the cell type to generate the mask for

    Returns:
        bool: True if mask was generated successfully, False otherwise
    """
    session = SessionLocal()
    try:
        # Get the image and cell from the database
        image = session.query(Image).filter_by(id=image_id).first()
        cell = session.query(Cell).filter_by(id=cell_id).first()

        if not image or not cell:
            print(f"Image ID {image_id} or Cell ID {cell_id} not found")
            return False

        # Check if mask already exists
        existing_mask = (
            session.query(Mask).filter_by(image_id=image_id, cell_id=cell_id).first()
        )

        if existing_mask:
            print(f"Mask already exists for Image ID {image_id} and Cell ID {cell_id}")
            return False

        # Define paths
        image_path = image.img_path
        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            return False

        # Create directory structure for the mask
        name = os.path.splitext(image.filename)[0]
        mask_dir = Path(f"data/dataset/masks/{name}")
        mask_dir.mkdir(exist_ok=True, parents=True)

        # Define output path for the mask
        ext = os.path.splitext(image.filename)[1]
        mask_path = str(mask_dir / f"{cell.name}{ext}")

        # Generate mask using the prediction model
        result = predict_mask(image_path, mask_path)

        if result is None:
            print(f"Failed to generate mask for {image_path}")
            return False

        # Add the mask to the database
        mask = Mask(
            image_id=image_id,
            mask_path=mask_path,
            cell_id=cell_id,
            is_segmented=0,  # Mark as not done, needs review
        )
        session.add(mask)
        session.commit()

        print(f"Generated mask for Image ID {image_id} and Cell ID {cell_id}")
        return True

    except Exception as e:
        print(f"Error generating mask: {e}")
        return False
    finally:
        session.close()


def generate_masks_for_new_images():
    """
    Generate masks for all images that don't have masks yet.

    Returns:
        int: Number of masks generated
    """
    session = SessionLocal()
    count = 0
    try:
        # Get all images without masks
        images_without_masks = (
            session.query(Image)
            .filter(~Image.id.in_(session.query(Mask.image_id).distinct()))
            .all()
        )

        # Get all cell types
        cells = session.query(Cell).all()

        # Generate masks for each image and cell type
        for image in images_without_masks:
            for cell in cells:
                success = generate_mask_for_image(image.id, cell.id)
                if success:
                    count += 1

    finally:
        session.close()

    return count

"""
Service to handle mask operations and trigger AI training based on modifications.
"""

from app.services.ai.train import increment_mask_modifications, start_training_if_needed


def record_mask_modification(image_id, cell_id):
    """
    Record that a mask was modified and check if training should be triggered.

    Args:
        image_id: ID of the image the mask belongs to
        cell_id: ID of the cell type for the mask
    """
    # Increment the counter for mask modifications
    increment_mask_modifications()

    # Check if we should start training
    start_training_if_needed()

    return True

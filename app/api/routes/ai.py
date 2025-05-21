from fastapi import APIRouter, BackgroundTasks, Depends
from app.services.ai.train import get_training_status, start_training_if_needed
from app.services.ai.auto_mask import (
    generate_mask_for_image,
    generate_masks_for_new_images,
)
from app.db.session import SessionLocal
from app.db.models import Image

router = APIRouter()

# Set this to the number of new images before retraining
RETRAINING_THRESHOLD = 10


@router.get("/status")
def get_model_status():
    """
    Get the current status of the AI model training process.

    Returns:
        dict: Status information including training state, progress, metrics, and timing.
    """
    # Get detailed training status from the training module
    status = get_training_status()

    # Add additional contextual information
    session = SessionLocal()
    image_count = session.query(Image).count()
    session.close()

    # Calculate images needed for next training
    if status["is_training"]:
        next_training_in = 0
    else:
        images_since_last = image_count - status["image_count_at_last_training"]
        mask_mods = status["mask_modifications_since_last_training"]
        next_training_in_images = max(0, RETRAINING_THRESHOLD - images_since_last)
        next_training_in_masks = max(0, RETRAINING_THRESHOLD - mask_mods)
        next_training_in = min(next_training_in_images, next_training_in_masks)

    # Determine status
    if status["is_preprocessing"]:
        current_status = "preprocessing"
        progress = status["preprocessing_progress"]
    elif status["is_training"]:
        current_status = "training"
        progress = status["progress"] * 100
    else:
        current_status = "idle"
        progress = None

    result = {
        "status": current_status,
        "details": {
            "is_preprocessing": status["is_preprocessing"],
            "is_training": status["is_training"],
            "current_epoch": status["current_epoch"],
            "total_epochs": status["total_epochs"],
            "current_loss": round(status["current_loss"], 4),
            "best_loss": (
                round(status["best_loss"], 4)
                if status["best_loss"] != float("inf")
                else None
            ),
            "progress_percent": (round(progress, 1) if progress is not None else None),
            "stable_epochs": status["stable_epochs"],
        },
        "timing": {
            "started_at": status["start_time"],
            "elapsed": status.get("elapsed_formatted", "N/A"),
            "completed_at": (
                status["end_time"]
                if not status["is_training"] and status["end_time"]
                else None
            ),
        },
        "next_training": {
            "images_needed": next_training_in,
            "image_count": image_count,
            "last_training_at": status["image_count_at_last_training"],
            "mask_modifications_since_last": status[
                "mask_modifications_since_last_training"
            ],
        },
    }

    return result


@router.post("/train")
def trigger_training(background_tasks: BackgroundTasks):
    """
    Manually trigger model training.

    Returns:
        dict: Status message and whether training was started.
    """
    if start_training_if_needed():
        return {"message": "Training started", "status": "training"}
    else:
        return {
            "message": "Training already in progress or no new data available",
            "status": "unchanged",
        }


@router.post("/generate-mask/{image_id}/{cell_id}")
def generate_mask(image_id: int, cell_id: int):
    """
    Generate a mask for a specific image and cell type using the AI model.

    Args:
        image_id: ID of the image to generate the mask for
        cell_id: ID of the cell type to generate the mask for

    Returns:
        dict: Status message indicating success or failure
    """
    success = generate_mask_for_image(image_id, cell_id)

    if success:
        return {"message": "Mask generated successfully", "status": "success"}
    else:
        return {"message": "Failed to generate mask", "status": "error"}


@router.post("/generate-masks")
def generate_all_masks(background_tasks: BackgroundTasks):
    """
    Generate masks for all images that don't have masks yet.
    This runs in the background as it may take some time.

    Returns:
        dict: Status message
    """
    # Run in background
    background_tasks.add_task(generate_masks_for_new_images)

    return {
        "message": "Started generating masks for new images",
        "status": "processing",
    }


# Background task to check for training conditions on each request
@router.get("/check-training-conditions")
def check_conditions():
    """Check if training should be triggered based on current conditions."""
    start_training_if_needed()
    return {"message": "Training conditions checked"}

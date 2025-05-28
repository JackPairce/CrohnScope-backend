"""
Training module for Cr# Global tracking variables
training_status = {
    "is_training": False,
    "is_preprocessing": False,
    "current_epoch": 0,
    "total_epochs": 0,
    "current_loss": 0.0,
    "best_loss": float('inf'),
    "start_time": None,
    "end_time": None,
    "elapsed_time": 0,
    "progress": 0.0,
    "preprocessing_progress": 0.0,
    "stable_epochs": 0,
    "img_processed": 0,
    "mask_processed": 0,
    "image_count_at_last_training": 0,
    "mask_modifications_since_last_training": 0
}els

This module implements training functionality for the UNet segmentation model,
tracking progress and providing status information.
"""

import time
import torch
import threading
import datetime
import psutil
from pathlib import Path
import torch.nn as nn
from app.db.models import Image
from app.db.session import SessionLocal
from app.types.ai import WhichModel

# Constants
RETRAINING_THRESHOLD = 10  # Retrain after every 10 new images
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS_DIR = Path("data/models")
MODELS_DIR.mkdir(exist_ok=True, parents=True)

# Global tracking variables
training_status = {
    "is_training": False,
    "is_preprocessing": False,
    "current_epoch": 0,
    "total_epochs": 0,
    "current_loss": 0.0,
    "best_loss": float("inf"),
    "start_time": None,
    "end_time": None,
    "elapsed_time": 0,
    "progress": 0.0,
    "preprocessing_progress": 0.0,
    "stable_epochs": 0,
    "img_processed": 0,
    "mask_processed": 0,
    "image_count_at_last_training": 0,
    "mask_modifications_since_last_training": 0,
}

# Create lock for thread safety
status_lock = threading.Lock()
training_thread = None


def dice_loss(pred, target):
    """
    Dice loss for segmentation tasks.
    """
    smooth = 1.0
    pred = torch.sigmoid(pred)

    # Flatten
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice


def combined_loss(pred, target):
    """
    Combined BCE and Dice loss.
    """
    bce = nn.BCEWithLogitsLoss()(pred, target)
    dice = dice_loss(pred, target)
    return 0.5 * bce + 0.5 * dice


def check_memory(min_required_mb=500):
    """
    Check if there's enough available memory.

    Args:
        min_required_mb: Minimum required free memory in MB

    Returns:
        tuple: (bool, float) - (has_enough_memory, available_mb)
    """
    available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)
    has_enough_memory = available_memory_mb >= min_required_mb

    if not has_enough_memory:
        print(
            f"FATAL: Not enough memory available. Only {available_memory_mb:.2f}MB free."
        )

    return has_enough_memory, available_memory_mb


def check_training_conditions():
    """
    Check if training or retraining should be triggered based on conditions:
    - When 10 new images are added
    - When 10 mask modifications occur
    """
    with status_lock:
        # Skip if currently training
        if training_status["is_training"]:
            return False

    session = SessionLocal()
    try:
        # Count images
        img_count = session.query(Image).count()

        # Check conditions
        new_images = img_count - training_status["image_count_at_last_training"]
        mask_mods = training_status["mask_modifications_since_last_training"]

        should_train = (
            new_images >= RETRAINING_THRESHOLD or mask_mods >= RETRAINING_THRESHOLD
        )

        return should_train
    finally:
        session.close()


def increment_mask_modifications():
    """
    Call this function whenever a mask is modified by a user.
    """
    with status_lock:
        training_status["mask_modifications_since_last_training"] += 1


def get_model(which_model: WhichModel):

    match which_model:
        case "unet":
            ...
        case _:
            raise ValueError(f"Unsupported model type: {which_model}")


# def train_model_thread():
#     """
#     Training function to be run in a separate thread.
#     """
#     # Check available memory before starting
#     has_memory, _ = check_memory(min_required_mb=500)
#     if not has_memory:
#         with status_lock:
#             training_status["is_training"] = False
#         return

#     model = UNet(in_channels=1, n_classes=1)
#     model_path = MODELS_DIR / "unet_segmentation.pth"

#     # Load existing model if available
#     if model_path.exists():
#         try:
#             model.load_state_dict(torch.load(model_path, map_location=DEVICE))
#             print(f"Loaded existing model from {model_path}")
#         except Exception as e:
#             print(f"Error loading model: {e}")

#     model = model.to(DEVICE)
#     optimizer = Adam(model.parameters(), lr=0.001)

#     # Update training status
#     with status_lock:
#         training_status.update(
#             {
#                 "is_training": True,
#                 "is_preprocessing": False,
#                 "start_time": time.time(),
#                 "current_epoch": 0,
#                 "total_epochs": 100,  # Maximum epochs
#                 "stable_epochs": 0,
#                 "progress": 0.0,
#             }
#         )

#         # Set the new baseline for image count
#         session = SessionLocal()
#         try:
#             training_status["image_count_at_last_training"] = session.query(
#                 Image
#             ).count()
#             training_status["mask_modifications_since_last_training"] = 0
#         finally:
#             session.close()

#     # Load data - this will set is_preprocessing to True during data loading
#     try:
#         dataloader = load_and_preprocess_data(batch_size=8)

#         if not dataloader or len(dataloader) == 0:
#             with status_lock:
#                 training_status["is_training"] = False
#                 training_status["end_time"] = time.time()
#                 training_status["elapsed_time"] = (
#                     training_status["end_time"] - training_status["start_time"]
#                 )
#             print("No data available for training")
#             return

#         # Training loop
#         previous_loss = float("inf")

#         for epoch in range(1, training_status["total_epochs"] + 1):
#             # Check memory before each epoch
#             has_memory, _ = check_memory(min_required_mb=500)
#             if not has_memory:
#                 print(
#                     f"FATAL: Memory running low during training. Stopping at epoch {epoch}."
#                 )
#                 break

#             model.train()
#             running_loss = 0.0

#             for batch_idx, (imgs, masks) in enumerate(dataloader):
#                 # Check memory periodically during batches
#                 if batch_idx % 10 == 0:
#                     has_memory, _ = check_memory(min_required_mb=500)
#                     if not has_memory:
#                         print(
#                             f"FATAL: Memory running low during batch processing. Stopping."
#                         )
#                         raise MemoryError("Insufficient memory to continue training")

#                 imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

#                 optimizer.zero_grad()
#                 outputs = model(imgs)
#                 loss = combined_loss(outputs, masks)
#                 loss.backward()
#                 optimizer.step()

#                 running_loss += loss.item()

#                 # Update status periodically
#                 if batch_idx % 5 == 0:
#                     with status_lock:
#                         training_status["current_loss"] = loss.item()
#                         training_status["progress"] = (batch_idx + 1) / len(dataloader)

#             # Epoch completed
#             epoch_loss = running_loss / len(dataloader)

#             with status_lock:
#                 training_status["current_epoch"] = epoch
#                 training_status["current_loss"] = epoch_loss

#                 # Check if this is the best model so far
#                 if epoch_loss < training_status["best_loss"]:
#                     training_status["best_loss"] = epoch_loss
#                     # Save the model
#                     torch.save(model.state_dict(), model_path)
#                     print(
#                         f"Saved best model at epoch {epoch} with loss {epoch_loss:.4f}"
#                     )

#             # Early stopping check
#             if abs(previous_loss - epoch_loss) < 0.0001:  # Loss is stable
#                 with status_lock:
#                     training_status["stable_epochs"] += 1

#                 if (
#                     training_status["stable_epochs"] >= 10
#                 ):  # Stop if loss is stable for 10 epochs
#                     print(f"Early stopping at epoch {epoch} - loss is stable")
#                     break
#             else:
#                 with status_lock:
#                     training_status["stable_epochs"] = 0

#             # Stop if loss is very low
#             if epoch_loss < 0.01:
#                 print(
#                     f"Stopping at epoch {epoch} - reached very low loss: {epoch_loss:.4f}"
#                 )
#                 break

#             previous_loss = epoch_loss

#     except MemoryError as me:
#         print(f"Training stopped due to memory constraints: {me}")
#     except Exception as e:
#         print(f"Error during training: {e}")

#     finally:
#         # Update status when training is complete
#         with status_lock:
#             training_status["is_training"] = False
#             training_status["end_time"] = time.time()
#             training_status["elapsed_time"] = (
#                 training_status["end_time"] - training_status["start_time"]
#             )
#             print(
#                 f"Training completed in {training_status['elapsed_time']:.2f} seconds"
#             )


# def start_training_if_needed():
#     """
#     Check conditions and start training in a background thread if needed.
#     """
#     global training_thread

#     if check_training_conditions() and (
#         training_thread is None or not training_thread.is_alive()
#     ):
#         training_thread = threading.Thread(target=train_model_thread)
#         training_thread.daemon = True
#         training_thread.start()
#         return True
#     return False


def get_training_status():
    """
    Returns the current training status.
    """
    with status_lock:
        status_copy = training_status.copy()

        # Add some computed/formatted fields
        if status_copy["is_training"]:
            current_time = time.time()
            elapsed = current_time - status_copy["start_time"]
            status_copy["elapsed_time"] = elapsed
            status_copy["elapsed_formatted"] = str(
                datetime.timedelta(seconds=int(elapsed))
            )
        elif status_copy["end_time"] and status_copy["start_time"]:
            status_copy["elapsed_formatted"] = str(
                datetime.timedelta(
                    seconds=int(status_copy["end_time"] - status_copy["start_time"])
                )
            )

        return status_copy

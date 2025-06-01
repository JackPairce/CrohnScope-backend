"""
Prediction module for using the trained U-Net model to generate masks.

This module provides functionality to load the trained model and use it for predicting
masks on new images.
"""

import os
import cv2
import torch
import numpy as np
from pathlib import Path
from torchvision import transforms
from shared.models.Unet.models import UNet

# Global model cache
model_cache = {}


def load_model(model_path=None):
    """
    Load the trained model for inference.

    Args:
        model_path: Path to the model weights file

    Returns:
        UNet model
    """
    if model_path is None:
        model_path = "data/models/best_unet_weights.pth"

    # If model is already loaded, return from cache
    if model_path in model_cache:
        return model_cache[model_path]

    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Determine number of cell types
    from API.db.session import SessionLocal
    from API.services.cell.cell_service import count_cells

    session = SessionLocal()
    num_cells = count_cells(session)
    session.close()

    # Load model
    model = UNet(in_channels=3, num_classes=num_cells)

    # Load weights
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])

    # Set model to evaluation mode
    model.eval()

    # Cache the model
    model_cache[model_path] = model

    return model


def preprocess_image(image_path):
    """
    Preprocess an image for model input.

    Args:
        image_path: Path to the image file

    Returns:
        Preprocessed tensor
    """
    # Read image
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = image_path  # Assume it's already a numpy array

    # Normalize
    image = image.astype(np.float32) / 255.0

    # Convert to tensor
    image = torch.from_numpy(image.transpose(2, 0, 1)).float()

    # Add batch dimension
    image = image.unsqueeze(0)

    return image


def predict_mask(image_path, output_path=None, model_path=None, threshold=0.5):
    """
    Predict mask for an image.

    Args:
        image_path: Path to the image file
        output_path: Path to save the output mask
        model_path: Path to the model weights file
        threshold: Threshold for binary mask

    Returns:
        Predicted mask as a numpy array
    """
    try:
        # Load model
        model = load_model(model_path)

        # Preprocess image
        input_tensor = preprocess_image(image_path)

        # Predict
        with torch.no_grad():
            # Use the predict method to get probabilities
            output = model.predict(input_tensor)

        # Convert to numpy
        output = output.squeeze().cpu().numpy()

        # Apply threshold to create binary mask
        binary_mask = (output > threshold).astype(np.uint8) * 255

        # Save mask if output path is provided
        if output_path:
            # Make sure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, binary_mask)

        return binary_mask

    except Exception as e:
        print(f"Error in predict_mask: {e}")
        return None

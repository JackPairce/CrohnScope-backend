"""
Prediction module for CrohnScope's AI models

This module implements functionality for predicting masks from medical images
using the trained UNet segmentation model.
"""

import os
import numpy as np
from pathlib import Path
import torch
import torchvision.transforms as T
from PIL import Image
from app.services.ai.models import UNet

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS_DIR = Path("data/models")
MODEL_PATH = MODELS_DIR / "unet_segmentation.pth"


def load_model():
    """
    Load the trained UNet model.
    """
    model = UNet(n_channels=3, n_classes=1)

    # Check if model file exists
    if not MODEL_PATH.exists():
        print(f"Warning: Model file not found at {MODEL_PATH}")
        return model

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return model


def predict_mask(image_path, output_path=None, threshold=0.5):
    """
    Generate a segmentation mask for the given image.

    Args:
        image_path: Path to the input image
        output_path: Optional path to save the output mask
        threshold: Probability threshold for binary mask

    Returns:
        Binary mask as numpy array
    """
    # Load the model
    model = load_model()

    # Load and preprocess the image
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

    # Preprocessing transformations
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Apply transformations and add batch dimension
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    # Make prediction
    with torch.no_grad():
        output = model(img_tensor)
        output = torch.sigmoid(output)

        # Convert to binary mask using threshold
        pred_mask = (output > threshold).cpu().squeeze().numpy().astype(np.uint8) * 255

    # Save if output_path is provided
    if output_path:
        mask_img = Image.fromarray(pred_mask)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        mask_img.save(output_path)

    return pred_mask

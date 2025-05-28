from typing import Literal, Optional, TypedDict
from numpy import ndarray
from pydantic import BaseModel
from datetime import datetime


class TrainingDetails(BaseModel):
    """Training details response model"""

    is_preprocessing: bool
    is_training: bool
    current_epoch: int
    total_epochs: int
    current_loss: float
    best_loss: Optional[float]
    progress_percent: Optional[float]
    stable_epochs: int


class TrainingTiming(BaseModel):
    """Training timing information"""

    started_at: Optional[datetime]
    elapsed: str
    completed_at: Optional[datetime]


class NextTraining(BaseModel):
    """Next training information"""

    images_needed: int
    image_count: int
    last_training_at: int
    mask_modifications_since_last: int


class ModelStatusResponse(BaseModel):
    """AI model status response"""

    status: str
    details: TrainingDetails
    timing: TrainingTiming
    next_training: NextTraining


class TrainingResponse(BaseModel):
    """Training trigger response"""

    message: str
    status: str


class MaskGenerationResponse(BaseModel):
    """Mask generation response"""

    message: str
    status: str


type WhichModel = Literal["unet", "yolo"]


class ImageWithMasks(TypedDict):
    image: ndarray
    masks: ndarray
    cell_id: int
    is_done: bool


class PatchData(TypedDict):
    """Data model for patch information"""

    img_patch: ndarray
    mask_patch: ndarray
    cell_id: int
    is_done: bool

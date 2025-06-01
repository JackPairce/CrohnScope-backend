from torch import from_numpy
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum
import time
from torch.utils.data import Dataset, DataLoader

from app.types.ai import PatchData
import torchvision.transforms as transforms


class TrainingStatusEnum(str, Enum):
    """Enum for training process status values."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    NOT_RUNNING = "not_running"
    UNKNOWN = "unknown"


class ProcessInfo(BaseModel):
    """Model for training process information with strict typing."""

    pid: Optional[int] = None
    status: TrainingStatusEnum = TrainingStatusEnum.NOT_RUNNING
    start_time: float = Field(default_factory=time.time)
    last_update: float = Field(default_factory=time.time)
    error: Optional[str] = None
    # Optional additional fields
    resources: Optional[Dict[str, Any]] = None
    progress: Optional[float] = None
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    loss: Optional[float] = None

    class Config:
        use_enum_values = True  # Store enum as string value
        extra = "allow"  # Allow additional fields


class TrainingProcessResponse(BaseModel):
    """Response model for training process status API endpoint."""

    status: TrainingStatusEnum
    pid: Optional[int] = None
    start_time: Optional[float] = None
    last_update: Optional[float] = None
    error: Optional[str] = None
    resources: Optional[Dict[str, Any]] = None
    elapsed_seconds: Optional[float] = None
    log: Optional[List[str]] = None


class PatchDataset(Dataset):
    """Dataset for patch-based segmentation using database"""

    def __init__(self, patches: List[PatchData], transform=None):
        self.patches = patches

        if transform is None:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.5], std=[0.5]
                    ),  # Adjust normalization as needed
                ]
            )
        self.transform = transform

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch_info = self.patches[idx]
        # Access dictionary items using string keys
        img_tensor = from_numpy(patch_info["img_patch"]).float()
        mask_tensor = from_numpy(patch_info["mask_patch"]).float()

        # Normalize image to [0,1] range
        img_tensor = img_tensor / 255.0

        # Convert to tensors with correct shape
        img_tensor = img_tensor.unsqueeze(0)  # Add channel dimension
        mask_tensor = mask_tensor.unsqueeze(0)  # Add channel dimension

        return img_tensor, mask_tensor

from typing import List, Optional
from pydantic import BaseModel


class ImageBase(BaseModel):
    id: int
    filename: str


class ApiImage(ImageBase):
    src: str  # Base64 encoded image
    is_done: bool = False  # If the user has finished annotating the image


class Mask(BaseModel):
    id: int
    image_id: int
    mask_path: str
    cell_id: Optional[int] = None
    is_mask_done: bool = False  # If the user has finished editing the mask
    is_annotation_done: bool = (
        False  # If the user has finished setting the mask's health status
    )


class ApiMask(Mask):
    src: str  # Base64 encoded mask


class ImageWithMasks(ApiImage):
    masks: List[ApiMask] = []

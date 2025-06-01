"""
Image related data models and types.
"""

from typing import List, Optional, Dict
from pydantic import BaseModel
from datetime import datetime
from enum import Enum


class DiagnosisEnum(str, Enum):
    """Diagnosis status enum"""

    unknown = "unknown"
    healthy = "healthy"
    unhealthy = "unhealthy"
    needs_review = "needs_review"


class HealthStatusEnum(str, Enum):
    """Health status enum"""

    healthy = "healthy"
    unhealthy = "unhealthy"
    unknown = "unknown"


class ImageBase(BaseModel):
    """Base image model"""

    filename: str
    img_path: str
    diagnosis: DiagnosisEnum = DiagnosisEnum.unknown


class ImageCreate(ImageBase):
    """Image creation model"""

    pass


class Image(ImageBase):
    """Image response model"""

    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ImageStatus(BaseModel):
    """Image service status response"""

    status: str


class ApiImage(BaseModel):
    id: int
    filename: str
    src: str  # Base64 encoded image
    is_done: bool = False  # If the user has finished annotating the image


class ImageListResponse(BaseModel):
    """Response model for paginated image list"""

    images: List[ApiImage]
    page: int
    total: int


from typing import Literal


# (0|1|2)[][]
type MaskArray = List[
    List[Literal[0, 1, 2]]
]  # 0: background, 1: unhealthy cells, 2: healthy cells


class MaskBase(BaseModel):
    """Base mask model"""

    mask_path: str
    is_segmented: bool = False
    health_status: HealthStatusEnum = HealthStatusEnum.unknown


class MaskCreate(MaskBase):
    """Mask creation model"""

    image_id: int
    cell_id: int


class RegionInfo(BaseModel):
    """Model for region information in a labeled mask."""

    id: int
    area: int
    boundingBox: Dict[str, int]
    centroid: Dict[str, int]


class LabeledMaskResponse(BaseModel):
    """Response model for labeled mask data"""

    labeledMask: str
    regions: List[RegionInfo]


class MaskMatrix(BaseModel):
    """Matrix representation of a mask"""

    mask_id: int
    cell_id: int
    labeledRegions: List[
        List[int]
    ]  # Matrix where each cell has a value from 0 (background) to N (regions)
    mask: MaskArray


class MaskMatricesResponse(BaseModel):
    """Response model for multiple mask matrices"""

    masks: List[MaskMatrix]


class ImageList(BaseModel):
    """Image list response"""

    items: List[Image]
    total: int


class ApiMask(BaseModel):
    """API representation of a mask"""

    id: int
    image_id: int
    mask_path: str
    cell_id: Optional[int] = None
    is_segmented: bool = False
    is_annotated: bool = False
    src: str  # Base64 encoded mask
    labeledMask: Optional[str] = (
        None  # Base64 encoded labeled mask where each region has a unique color
    )
    regions: Optional[List[RegionInfo]] = None  # List of region statistics


class MaskList(BaseModel):
    """Mask list response"""

    items: List[ApiMask]
    total: int


class ImageResponse(BaseModel):
    """Image operation response"""

    message: str
    status: str
    image: Optional[Image] = None


class MaskResponse(BaseModel):
    """Mask operation response"""

    message: str
    status: str
    mask: Optional[ApiMask] = None


from pydantic import BaseModel


class SaveMaskResponse(BaseModel):
    """Model for mask saving response."""

    id: int
    cell_id: int
    data: MaskArray


class MaskUpdateResponse(BaseModel):
    """Response model for mask update operations"""

    message: str


class MaskAnnotationResponse(BaseModel):
    """Response model for mask annotation operations"""

    message: str

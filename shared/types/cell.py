from typing import Optional, List, Tuple
from pydantic import BaseModel


class ApiCell(BaseModel):
    id: int
    name: str
    description: str
    img: str  # Base64 encoded image string


class CellTypeResponse(BaseModel):
    """Cell type operation response"""

    message: str = "Operation successful"
    status: str = "success"
    cell_type: Optional[ApiCell] = None


class CellTypeCreateResponse(BaseModel):
    """Response model for cell type creation"""

    message: str = "Cell type created successfully"
    cell_type: ApiCell


class CellTypeUpdateResponse(BaseModel):
    """Response model for cell type update"""

    message: str = "Cell type updated successfully"
    cell_type: ApiCell


class CellTypeDeleteResponse(BaseModel):
    """Response model for cell type deletion"""

    message: str = "Cell type deleted successfully"
    id: int


class CellTypeCSV(BaseModel):
    """Model for cell type CSV data"""

    name: str
    description: Optional[str] = None
    image: Optional[str] = None  # Base64 encoded image string

    def to_tuple(self) -> Tuple[str, Optional[str], Optional[str]]:
        """Convert to tuple format for legacy compatibility"""
        return (self.name, self.description, self.image)

    @classmethod
    def from_tuple(
        cls, data: Tuple[str, Optional[str], Optional[str]]
    ) -> "CellTypeCSV":
        """Create from tuple format for legacy compatibility"""
        return cls(name=data[0], description=data[1], image=data[2])

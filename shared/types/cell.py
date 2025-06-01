from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel


class ApiCell(BaseModel):
    id: int
    name: str
    description: Optional[str] = None


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

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel
from app.db.models import Cell, Mask
from app.db.session import SessionLocal
from app.types.image import ApiCell
import os

router = APIRouter()


class CellCreate(BaseModel):
    name: str
    description: Optional[str] = None


class CellUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


@router.get("/", response_model=List[ApiCell])
def get_all_cells():
    """Get all cells in the database"""
    session = SessionLocal()
    try:
        cells = session.query(Cell).all()
        return [
            ApiCell(id=cell.id, name=cell.name, description=cell.description)
            for cell in cells
        ]
    finally:
        session.close()


@router.get("/get/{image_id}")
def get_cells() -> List[ApiCell]:
    """Compatibility method for existing code - get all cells for an image"""
    session = SessionLocal()
    try:
        cells = session.query(Cell).all()
        return [
            ApiCell(id=cell.id, name=cell.name, description=cell.description)
            for cell in cells
        ]
    finally:
        session.close()


@router.get("/{cell_id}", response_model=ApiCell)
def get_cell(cell_id: int):
    """Get a specific cell by ID"""
    session = SessionLocal()
    try:
        cell = session.query(Cell).filter(Cell.id == cell_id).first()
        if cell is None:
            raise HTTPException(status_code=404, detail="Cell not found")
        return ApiCell(id=cell.id, name=cell.name, description=cell.description)
    finally:
        session.close()


@router.post("/", response_model=ApiCell)
def create_cell(cell: CellCreate):
    """Create a new cell"""
    session = SessionLocal()
    try:
        # Check if a cell with the same name already exists
        existing = session.query(Cell).filter(Cell.name == cell.name).first()
        if existing:
            raise HTTPException(
                status_code=400, detail="Cell with this name already exists"
            )

        db_cell = Cell(name=cell.name, description=cell.description)
        session.add(db_cell)
        session.commit()
        session.refresh(db_cell)
        return ApiCell(
            id=db_cell.id, name=db_cell.name, description=db_cell.description
        )
    finally:
        session.close()


@router.post("/save/{image_id}")
def add_cell(name: str):
    """Legacy method for compatibility with existing code"""
    session = SessionLocal()
    try:
        existing = session.query(Cell).filter(Cell.name == name).first()
        if existing:
            return {"message": "Cell already exists", "id": existing.id}

        cell = Cell(name=name)
        session.add(cell)
        session.commit()
        session.refresh(cell)
        return {"message": "Cell added successfully", "id": cell.id}
    finally:
        session.close()


@router.put("/{cell_id}", response_model=ApiCell)
def update_cell(cell_id: int, cell: CellUpdate):
    """Update an existing cell"""
    session = SessionLocal()
    try:
        db_cell = session.query(Cell).filter(Cell.id == cell_id).first()
        if db_cell is None:
            raise HTTPException(status_code=404, detail="Cell not found")

        # Update fields if provided
        if cell.name is not None:
            # Check if another cell with the same name exists
            if cell.name != db_cell.name:
                existing = session.query(Cell).filter(Cell.name == cell.name).first()
                if existing:
                    raise HTTPException(
                        status_code=400, detail="Cell with this name already exists"
                    )
            db_cell.name = cell.name

        if cell.description is not None:
            db_cell.description = cell.description

        session.commit()
        session.refresh(db_cell)
        return ApiCell(
            id=db_cell.id, name=db_cell.name, description=db_cell.description
        )
    finally:
        session.close()


@router.delete("/{cell_id}", response_model=dict)
def delete_cell(cell_id: int):
    """Delete a cell and all associated masks"""
    session = SessionLocal()
    try:
        # Check if the cell exists
        cell = session.query(Cell).filter(Cell.id == cell_id).first()
        if cell is None:
            raise HTTPException(status_code=404, detail="Cell not found")

        # Check if there are masks using this cell
        masks = session.query(Mask).filter(Mask.cell_id == cell_id).all()

        # Delete associated masks if any
        if masks:
            # First delete the mask records
            for mask in masks:
                # Delete the mask file if it exists
                if mask.mask_path and os.path.exists(mask.mask_path):
                    try:
                        os.remove(mask.mask_path)
                    except Exception as e:
                        print(f"Error deleting mask file {mask.mask_path}: {e}")
                # Delete the database record
                session.delete(mask)

        # Delete the cell
        session.delete(cell)
        session.commit()

        return {
            "message": "Cell deleted successfully",
            "id": cell_id,
            "masks_deleted": len(masks) if masks else 0,
        }
    finally:
        session.close()

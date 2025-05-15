from typing import List
from fastapi import APIRouter
from app.db.models import Cell
from app.db.session import SessionLocal
from app.types.image import ApiCell

router = APIRouter()


@router.get("/get/{image_id}")
def get_cells() -> List[ApiCell]:
    session = SessionLocal()
    try:
        cells = session.query(Cell).all()
        return [ApiCell(id=cell.id, name=cell.name) for cell in cells]
    finally:
        session.close()


@router.post("/save/{image_id}")
def add_cell(name: str):
    session = SessionLocal()
    try:
        cell = Cell(name=name)
        session.add(cell)
        session.commit()
        return {"message": "Cell added successfully"}
    finally:
        session.close()

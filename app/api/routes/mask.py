from email import message
from typing import Any, List, TypedDict
import cv2
from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel
from app.api.routes.cells import get_cells
from app.db.models import Image, Mask
from app.db.session import SessionLocal
from app.services.image import mask
from app.types.image import ApiMask
import os
import base64

from utils.converters import ToBase64, base64_to_file

router = APIRouter()


@router.get("/get/{image_id}")
def get_masks(image_id: int) -> List[ApiMask]:
    session = SessionLocal()
    try:
        # Get all masks for the image
        masks = session.query(Mask).filter_by(image_id=image_id).all()
        api_masks = [
            ApiMask(
                id=mask.id,
                image_id=mask.image_id,
                mask_path=mask.mask_path,
                src=ToBase64(mask.mask_path),
                cell_id=mask.cell_id,
            )
            for mask in masks
        ]
        return api_masks
    finally:
        session.close()


class SaveMaskResponse(TypedDict):
    id: int
    cell_id: int
    src: str


@router.post("/save/{image_id}")
def save_masks(image_id: int, body=Body(...)):
    masks: List[SaveMaskResponse] = body
    session = SessionLocal()
    try:
        # Find the image in the database
        image = session.query(Image).filter_by(id=image_id).first()
        cells = get_cells()
        if not image:
            raise HTTPException(status_code=404, detail="Image not found")
        # Save each mask
        for mask in masks:
            cell_name = next(
                (cell.name for cell in cells if cell.id == mask["cell_id"]), None
            )
            if cell_name is None:
                continue
            name, mimetype = str(image.filename).split(".")
            mask_path = os.path.join(
                "data/dataset/masks",
                name,
                cell_name + "." + mimetype,
            )

            base64_to_file(mask["src"], mask_path)

            # Update the database
            db_mask = session.query(Mask).filter_by(id=mask.get("id")).first()
            if db_mask:
                db_mask.mask_path = mask_path
            else:
                # Add the mask to the database
                mask_record = Mask(
                    image_id=image_id, mask_path=mask_path, cell_id=mask["cell_id"]
                )
                session.add(mask_record)
            session.commit()

        return {"message": "Masks saved successfully"}
    finally:
        session.close()


@router.put("/done/{mask_id}")
def mask_done(mask_id: int):
    session = SessionLocal()
    try:
        # Update the database
        db_mask = session.query(Mask).filter_by(id=mask_id).first()
        if db_mask:
            db_mask.is_mask_done = 1
        else:
            raise ValueError("the mask is not found")
    finally:
        session.close()


@router.post("/alternate")
def alternate_masks(image_id: int, mask1: str, mask2: str):
    session = SessionLocal()
    try:
        # Find the mask directory
        mask_dir = os.path.join("data/dataset/masks", str(image_id))
        if not os.path.exists(mask_dir):
            raise HTTPException(status_code=404, detail="Mask directory not found")

        # Alternate the mask names
        mask1_path = os.path.join(mask_dir, mask1)
        mask2_path = os.path.join(mask_dir, mask2)
        if not os.path.exists(mask1_path) or not os.path.exists(mask2_path):
            raise HTTPException(status_code=404, detail="One or both masks not found")

        temp_path = os.path.join(mask_dir, "temp_mask")
        os.rename(mask1_path, temp_path)
        os.rename(mask2_path, mask1_path)
        os.rename(temp_path, mask2_path)

        return {"message": "Masks alternated successfully"}
    finally:
        session.close()

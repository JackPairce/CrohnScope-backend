from typing import List, Literal, Optional, TypedDict
from fastapi import APIRouter, File, HTTPException, Query
from sqlalchemy import distinct, func
from sqlalchemy.orm import Session
from app.api.routes.cells import get_cells
from app.db.models import Image, Mask
from app.db.session import SessionLocal
from app.types.image import ApiImage, ImageWithMasks, ApiMask
import os
import shutil
import numpy as np
import cv2
import base64

from utils.converters import ToBase64

router = APIRouter()


# Helper function to get the next image ID
def get_next_image_id(session: Session):
    last_image = session.query(Image).order_by(Image.id.desc()).first()
    return (int(last_image.filename) + 1) if last_image else 1


@router.get("/status")
def get_model_status():
    return {"status": "data service is ready"}


class ImageListResponse(TypedDict):
    images: List[ApiImage]
    page: int
    total: int


@router.get("/all/{page}")
def get_images(
    page: int, done: Optional[Literal["0", "1"]] = Query(None)
) -> ImageListResponse:
    session = SessionLocal()
    try:
        page_size = 10
        offset = (page - 1) * page_size
        db_query = session.query(distinct(Mask.image_id)).filter(
            Mask.is_mask_done == (int(done) if (done) else 0)
        )

        db_images = (
            session.query(Image)
            .filter(Image.id.in_([id[0] for id in db_query.all()]))
            .order_by(Image.filename)
            .offset(offset)
            .limit(page_size)
            .all()
        )
        api_images = [
            ApiImage(
                id=img.id,
                filename=img.filename,
                src=ToBase64(img.img_path),
                is_done=False,
            )
            for img in db_images
        ]
        return {"images": api_images, "page": page, "total": db_query.count()}
    finally:
        session.close()


@router.post("/upload")
def upload_image(file: ApiImage = File(...)):
    session = SessionLocal()
    try:
        # Get the next image ID
        next_id = get_next_image_id(session)
        ext = file.filename.split(".")[-1]
        new_filename = f"{next_id}.{ext}"
        image_path = os.path.join("data/dataset/images", new_filename)

        # Save the file
        with open(image_path, "wb") as buffer:
            base64_image = file.src.split(",")[1]
            image_data = base64.b64decode(base64_image)
            buffer.write(image_data)

        # Add to the database
        image = Image(filename=new_filename)
        session.add(image)
        session.commit()
        return {"message": "Image uploaded successfully", "id": next_id}
    finally:
        session.close()


@router.delete("/delete/{image_id}")
def delete_image(image_id: int):
    session = SessionLocal()
    try:
        # Find the image in the database
        image = session.query(Image).filter_by(id=image_id).first()
        if not image:
            raise HTTPException(status_code=404, detail="Image not found")

        # Delete the image file
        image_path = os.path.join("data/dataset/images", image.filename)
        if os.path.exists(image_path):
            os.remove(image_path)

        # Delete associated masks
        mask_dir = os.path.join("data/dataset/masks", str(image_id))
        if os.path.exists(mask_dir):
            shutil.rmtree(mask_dir)

        # Remove from the database
        session.delete(image)
        session.commit()
        return {"message": "Image deleted successfully"}
    finally:
        session.close()



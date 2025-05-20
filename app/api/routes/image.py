from typing import List, Literal, Optional, TypedDict
from fastapi import APIRouter, File, HTTPException, Query
from sqlalchemy import distinct, func
from sqlalchemy.orm import Session
from app.api.routes.cells import get_cells
from app.db.models import Image, Mask
from app.db.session import SessionLocal
from app.types.image import ApiImage, ImageWithMasks, ApiMask
from app.services.ai.train import start_training_if_needed
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


def get_images_by_done_status(
    session: Session, done: int, offset: int = 0, limit: int = 10
):
    base_query = session.query(Image.id).outerjoin(Mask).group_by(Image.id)

    if done == 1:
        filtered_query = base_query.having(
            func.count(Mask.id) > 0, func.count(Mask.id) == func.sum(Mask.is_mask_done)
        )
    else:
        filtered_query = base_query.having(
            (func.count(Mask.id) == 0)
            | (func.count(Mask.id) != func.sum(Mask.is_mask_done))
        )

    subquery = filtered_query.subquery()

    # Get paginated images
    images = (
        session.query(Image)
        .join(subquery, Image.id == subquery.c.id)
        .order_by(Image.filename)
        .offset(offset)
        .limit(limit)
        .all()
    )

    # Get total count
    total_count = session.query(func.count()).select_from(subquery).scalar()

    return images, total_count


@router.get("/all/{page}")
def get_images(
    page: int, done: Optional[Literal["0", "1"]] = Query(None)
) -> ImageListResponse:
    session = SessionLocal()
    try:
        page_size = 10
        db_images, db_count = get_images_by_done_status(
            session,
            (int(done) if done else 0),
            offset=(page - 1) * page_size,
            limit=page_size,
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
        return {"images": api_images, "page": page, "total": db_count}
    finally:
        session.close()


@router.post("/upload")
def upload_image(image_data: ApiImage) -> ApiImage:
    session = SessionLocal()
    try:
        # Get the next image ID
        next_id = get_next_image_id(session)

        # Validate file extension
        ext = image_data.filename.split(".")[-1].lower()
        if ext not in ["jpg", "jpeg", "png"]:
            raise HTTPException(status_code=400, detail="Invalid file format")

        new_filename = f"{next_id}.{ext}"
        upload_dir = "data/dataset/images"

        # Ensure directory exists
        os.makedirs(upload_dir, exist_ok=True)
        image_path = os.path.join(upload_dir, new_filename)

        try:
            # Save the file
            with open(image_path, "wb") as buffer:
                base64_image = image_data.src.split(",")[1]
                image_data_bytes = base64.b64decode(base64_image)
                buffer.write(image_data_bytes)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to save image: {str(e)}"
            )

        # Add to the database
        db_image = Image(filename=new_filename)
        session.add(db_image)
        session.commit()

        # Check if training should be triggered
        start_training_if_needed()

        return ApiImage(
            id=next_id, filename=new_filename, src=image_data.src, is_done=False
        )
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=str(e))
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

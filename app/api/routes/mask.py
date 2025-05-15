from fastapi import APIRouter, UploadFile, File, HTTPException
from enum import Enum
from sqlalchemy.orm import Session
from app.db.models import Image, Mask, Cell
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
    return (last_image.id + 1) if last_image else 1


@router.get("/status")
def get_model_status():
    return {"status": "data service is ready"}

@router.get("/all/{page}")
def get_images(page: int) :
    session = SessionLocal()
    try:
        page_size = 10
        offset = (page - 1) * page_size
        db_images_count = session.query(Image).count()
        db_images = session.query(Image).order_by(Image.filename).offset(offset).limit(page_size).all()
        api_images = [ApiImage(id=img.id, filename=img.filename, src=ToBase64(img.img_path), is_done=False) for img in db_images]
        return {"images": api_images, "page": page, "total": db_images_count}   
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


@router.get("/get/{image_id}", response_model=ImageWithMasks)
def get_image(image_id: int):
    session = SessionLocal()
    try:
        # Find the image in the database
        image = session.query(Image).filter_by(id=image_id).first()
        if not image:
            raise HTTPException(status_code=404, detail="Image not found")

        # Read and encode the image to base64
        image_path = os.path.join("data/dataset/images", image.filename)
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail="Image file not found")

        # Create a black mask for the image
        img = cv2.imread(image_path)
        height, width, _ = img.shape
        mask = np.zeros((height, width, 3), dtype=np.uint8)

        # Save the mask
        mask_dir = os.path.join("data/dataset/masks", str(image_id))
        os.makedirs(mask_dir, exist_ok=True)
        mask_path = os.path.join(mask_dir, "mask.png")
        cv2.imwrite(mask_path, mask)

        # Add the mask to the database
        mask_record = Mask(image_id=image_id, mask_path=mask_path)
        session.add(mask_record)
        session.commit()

        # Prepare response
        api_image = ApiImage(
            id=image.id,
            filename=image.filename,
            src=ToBase64(image_path),
            is_done=False,
        )
        api_mask = ApiMask(
            id=mask_record.id, image_id=image_id, mask_path=mask_path, src=""
        )
        return ImageWithMasks(**api_image.__dict__, masks=[api_mask])
    finally:
        session.close()


@router.post("/masks/alternate")
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

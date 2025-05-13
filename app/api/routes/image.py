from typing import List
from fastapi import APIRouter
import cv2

from app.types.image import Mask

router = APIRouter()


@router.post("/cc")
def ConnectedComponents(mask: str):
    print(mask)
    return mask

    # binary = cv2.threshold(mask.matrix, 127, 255, cv2.THRESH_BINARY)[1]
    # _, labels_im = cv2.connectedComponents(binary)

    # return labels_im.tolist()

from fastapi import APIRouter
from enum import Enum

router = APIRouter()


@router.get("/status")
def get_model_status():
    return {"status": "data service is ready"}

from fastapi import APIRouter

router = APIRouter()


@router.get("/status")
def get_model_status():
    return {"status": "AI model is ready"}

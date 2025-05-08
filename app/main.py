from fastapi import FastAPI
from app.api.routes import ai, data

app = FastAPI()

app.include_router(ai.router, prefix="/ai")
app.include_router(data.router, prefix="/data")
# app.include_router(image.router, prefix="/image")

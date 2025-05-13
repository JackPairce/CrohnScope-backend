from fastapi import FastAPI
from app.api.routes import ai, data, image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.db.models import Base

app = FastAPI()

# Database configuration
DATABASE_URL = "postgresql://postgres:postgres@127.0.0.1:5432/crohnscope" # TODO: use env vars
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Include routers
app.include_router(ai.router, prefix="/ai")
app.include_router(data.router, prefix="/data")
app.include_router(image.router, prefix="/image")


@app.on_event("startup")
def startup():
    # Initialize the database
    Base.metadata.create_all(bind=engine)

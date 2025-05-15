from fastapi import FastAPI
from app.api.routes import ai, cells, image, mask
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.db.models import Base, Image, Cell, DiagnosisEnum, Mask
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="CrohnScope API",
    description="API for CrohnScope, a web application for annotating and analyzing medical images.",
    version="0.1.0",
    docs_url="/docs",
    openapi_url="/openapi.json",
)


# Database configuration
DATABASE_URL = (
    "postgresql://postgres:postgres@127.0.0.1:5432/crohnscope"  # TODO: use env vars
)
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Include routers
app.include_router(ai.router, prefix="/ai")
app.include_router(mask.router, prefix="/mask")
app.include_router(image.router, prefix="/image")
app.include_router(cells.router, prefix="/cells")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    # check if the database is up and running if not, crash the server
    try:
        with engine.connect() as connection:
            pass
    except Exception as e:
        print(f"Database connection error: {e}")
        raise

    # Drop all tables and reinitialize the database
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


@app.on_event("startup")
def populate_database():
    session = SessionLocal()
    images_path = "data/dataset/images"
    masks_path = "data/dataset/masks"

    for image_file in os.listdir(images_path):
        image_path = os.path.join(images_path, image_file)
        if os.path.isfile(image_path):
            # Create an Image record
            image = Image(
                filename=image_file,
                img_path=image_path,
                diagnosis=DiagnosisEnum.unknown,
            )
            session.add(image)
            session.commit()

            # Add corresponding masks for the image
            image_masks_path = os.path.join(masks_path, os.path.splitext(image_file)[0])
            if os.path.exists(image_masks_path):
                for mask_file in os.listdir(image_masks_path):
                    mask_path = os.path.join(image_masks_path, mask_file)
                    if os.path.isfile(mask_path):
                        # Check if the cell exists, if not, create it
                        cell_name = os.path.splitext(mask_file)[
                            0
                        ]  # Assuming mask file name represents cell name
                        cell = session.query(Cell).filter_by(name=cell_name).first()
                        if not cell:
                            cell = Cell(name=cell_name)
                            session.add(cell)
                            session.commit()
                        # Create a Mask record
                        mask = Mask(
                            image_id=image.id, mask_path=mask_path, cell_id=cell.id
                        )
                        session.add(mask)
    session.commit()
    session.close()

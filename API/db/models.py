from typing import Literal
from sqlalchemy import Integer, String, Text, ForeignKey, Enum
from sqlalchemy.orm.properties import MappedColumn
from sqlalchemy.types import TypeDecorator, LargeBinary
from sqlalchemy.orm import relationship, declarative_base, mapped_column
import enum
import numpy as np
import io


Base = declarative_base()

# Generated by Copilot
DB_BOOL = Literal[0, 1]  # Use 0 for False, 1 for True


class NumpyArray(TypeDecorator):
    impl = LargeBinary

    def process_bind_param(self, value, dialect):
        if value is not None:
            out = io.BytesIO()
            np.save(out, value)
            return out.getvalue()
        return None

    def process_result_value(self, value, dialect):
        if value is not None:
            inp = io.BytesIO(value)
            return np.load(inp, allow_pickle=False)
        return None


class DiagnosisEnum(str, enum.Enum):
    crohn = "Crohn"
    healthy = "Healthy"
    unknown = "Unknown"


# Delete the wrong enum - we'll use the one defined below


class Image(Base):
    __tablename__ = "images"

    id: MappedColumn[int] = mapped_column(Integer, primary_key=True, index=True)
    filename: MappedColumn[str] = mapped_column(String, unique=True, nullable=False)
    img_path: MappedColumn[str] = mapped_column(Text, nullable=False)
    diagnosis: MappedColumn[DiagnosisEnum] = mapped_column(
        Enum(DiagnosisEnum), default=DiagnosisEnum.unknown
    )
    phase: MappedColumn[int] = mapped_column(Integer, nullable=True)

    masks = relationship("Mask", back_populates="image", cascade="all, delete")


class Cell(Base):
    __tablename__ = "cells"

    id: MappedColumn[int] = mapped_column(Integer, primary_key=True, index=True)
    name: MappedColumn[str] = mapped_column(String, nullable=False, unique=True)
    description: MappedColumn[str] = mapped_column(Text, nullable=True)
    image: MappedColumn[str] = mapped_column(
        Text, default="", nullable=True
    )  # Base64 encoded image string

    masks = relationship("Mask", back_populates="cell")


# Generated by Copilot
class HealthStatusEnum(str, enum.Enum):
    """Health status enum for mask regions.
    Used both in the database and to match numpy array values:
    0: background (not stored in enum)
    1: unhealthy
    2: healthy
    """

    unhealthy = "unhealthy"  # maps to value 1 in mask
    healthy = "healthy"  # maps to value 2 in mask


class Mask(Base):
    __tablename__ = "masks"

    id: MappedColumn[int] = mapped_column(Integer, primary_key=True, index=True)
    image_id: MappedColumn[int] = mapped_column(
        Integer, ForeignKey("images.id"), nullable=False
    )
    mask_path: MappedColumn[str] = mapped_column(Text, nullable=False)
    cell_id: MappedColumn[int] = mapped_column(
        Integer, ForeignKey("cells.id"), nullable=True
    )
    is_segmented: MappedColumn[DB_BOOL] = mapped_column(
        Integer, default=0
    )  # 0 for False, 1 for True
    is_annotated: MappedColumn[DB_BOOL] = mapped_column(
        Integer, default=0
    )  # 0 for False, 1 for True
    health_status: MappedColumn[HealthStatusEnum] = mapped_column(
        Enum(HealthStatusEnum), default=HealthStatusEnum.unhealthy
    )  # Default to unhealthy

    image = relationship("Image", back_populates="masks")
    cell = relationship("Cell", back_populates="masks")

from sqlalchemy import Integer, String, Text, ForeignKey, Enum
from sqlalchemy.orm import relationship, declarative_base, mapped_column
import enum

Base = declarative_base()


class DiagnosisEnum(str, enum.Enum):
    crohn = "Crohn"
    healthy = "Healthy"
    unknown = "Unknown"


class HealthStatusEnum(str, enum.Enum):
    healthy = "healthy"
    abnormal = "abnormal"


class Image(Base):
    __tablename__ = "images"

    id = mapped_column(Integer, primary_key=True, index=True)
    filename = mapped_column(String, unique=True, nullable=False)
    img_path = mapped_column(Text, nullable=False)
    diagnosis = mapped_column(Enum(DiagnosisEnum), default=DiagnosisEnum.unknown)
    phase = mapped_column(Integer, nullable=True)

    masks = relationship("Mask", back_populates="image", cascade="all, delete")


class Cell(Base):
    __tablename__ = "cells"

    id = mapped_column(Integer, primary_key=True, index=True)
    name = mapped_column(String, nullable=False, unique=True)
    description = mapped_column(Text, nullable=True)

    masks = relationship("Mask", back_populates="cell")


class Mask(Base):
    __tablename__ = "masks"

    id = mapped_column(Integer, primary_key=True, index=True)
    image_id = mapped_column(Integer, ForeignKey("images.id"), nullable=False)
    mask_path = mapped_column(Text, nullable=False)
    cell_id = mapped_column(Integer, ForeignKey("cells.id"), nullable=True)
    is_mask_done = mapped_column(Integer, default=0)  # 0 for False, 1 for True
    is_annotation_done = mapped_column(Integer, default=0)  # 0 for False, 1 for True

    image = relationship("Image", back_populates="masks")
    cell = relationship("Cell", back_populates="masks")

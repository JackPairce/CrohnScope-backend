from sqlalchemy import Column, Integer, String, Text, ForeignKey, Enum
from sqlalchemy.orm import relationship, declarative_base
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

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, unique=True, nullable=False)
    diagnosis = Column(Enum(DiagnosisEnum), default=DiagnosisEnum.unknown)
    phase = Column(Integer, nullable=True)

    cells = relationship("Cell", back_populates="image", cascade="all, delete")


class Cell(Base):
    __tablename__ = "cells"

    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(Integer, ForeignKey("images.id"), nullable=False)
    mask_path = Column(Text, nullable=False)
    cell_type = Column(String, nullable=True)  # granuloma, ulcer, etc.
    health_status = Column(Enum(HealthStatusEnum), default=HealthStatusEnum.healthy)

    image = relationship("Image", back_populates="cells")

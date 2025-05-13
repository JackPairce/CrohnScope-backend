from sqlalchemy.orm import Session
from . import models

class CRUDBase:
    def __init__(self, model):
        """CRUD object with default methods to Create, Read, Update, Delete (CRUD).
        :param model: A SQLAlchemy model class
        """
        self.model = model

    def get(self, db: Session, id: int):
        return db.query(self.model).filter(self.model.id == id).first()

    def get_all(self, db: Session):
        return db.query(self.model).all()

    def create(self, db: Session, obj_in):
        db_obj = self.model(**obj_in.dict())  # Assuming obj_in is a Pydantic model
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def update(self, db: Session, db_obj, obj_in):
        obj_data = obj_in.dict(exclude_unset=True)  # Assuming obj_in is a Pydantic model
        for field, value in obj_data.items():
            setattr(db_obj, field, value)
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def delete(self, db: Session, id: int):
        obj = db.query(self.model).get(id)
        if obj:
            db.delete(obj)
            db.commit()
        return obj
from app.db.session import SessionLocal, engine
from app.db.models import Base, Image, Cell, DiagnosisEnum, Mask
import os
import cv2
import numpy as np
import json
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy.inspection import inspect
from tqdm import tqdm


def serialize_instance(obj):
    """Helper function to serialize SQLAlchemy objects"""
    if hasattr(obj, "__dict__"):
        fields = {}
        for field in [x for x in dir(obj) if not x.startswith("_") and x != "metadata"]:
            data = obj.__getattribute__(field)
            try:
                # Handle special types like datetime, numpy arrays
                if isinstance(data, datetime):
                    fields[field] = data.isoformat()
                elif isinstance(data, np.ndarray):
                    fields[field] = data.tolist()
                else:
                    json.dumps({field: data})  # Test if serializable
                    fields[field] = data
            except (TypeError, OverflowError):
                fields[field] = None
        return fields
    return str(obj)


def backup_database(backup_dir="data/backups") -> str | None:
    """Backup the database to a JSON file using SQLAlchemy. Returns the backup file path if successful, None otherwise."""
    os.makedirs(backup_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = os.path.join(backup_dir, f"crohnscope_{timestamp}.json")

    session = SessionLocal()
    try:
        backup_data = {}
        for table in tqdm(Base.metadata.sorted_tables, desc="Backing up tables"):
            model = next(
                (c for c in Base.__subclasses__() if c.__table__ == table), None
            )
            if model:
                instances = session.query(model).all()
                backup_data[table.name] = [
                    serialize_instance(inst)
                    for inst in tqdm(
                        instances, desc=f"Serializing {table.name}", leave=False
                    )
                ]

        with open(backup_file, "w") as f:
            json.dump(backup_data, f, indent=2)
        print(f"Database backup created at: {backup_file}")
        return backup_file
    except Exception as e:
        print(f"Error creating database backup: {e}")
        return None
    finally:
        session.close()


def restore_database(backup_file: str) -> bool:
    """Restore the database from a JSON file using SQLAlchemy"""
    if not os.path.exists(backup_file):
        print(f"Backup file not found: {backup_file}")
        return False

    session = SessionLocal()
    try:
        with open(backup_file, "r") as f:
            backup_data = json.load(f)

        # Clear existing data
        for table in tqdm(
            reversed(Base.metadata.sorted_tables), desc="Clearing tables"
        ):
            session.execute(table.delete())

        # Restore data table by table
        for table_name, instances in tqdm(backup_data.items(), desc="Restoring tables"):
            model = next(
                (c for c in Base.__subclasses__() if c.__table__.name == table_name),
                None,
            )
            if model:
                for instance_data in tqdm(
                    instances, desc=f"Restoring {table_name}", leave=False
                ):
                    # Filter out any fields that aren't columns
                    valid_columns = {c.key for c in inspect(model).columns}
                    filtered_data = {
                        k: v for k, v in instance_data.items() if k in valid_columns
                    }
                    instance = model(**filtered_data)
                    session.add(instance)

        session.commit()
        print(f"Database restored from: {backup_file}")
        return True
    except Exception as e:
        session.rollback()
        print(f"Error restoring database: {e}")
        return False
    finally:
        session.close()


def load_cell_types(cell_types_file):
    """Load cell types from file or create default ones"""
    cell_type_info = []
    if os.path.exists(cell_types_file):
        with open(cell_types_file, "r") as f:
            print("Loading cell types from file...")
            lines = f.readlines()
            for line in tqdm(lines, desc="Processing cell types"):
                line = line.strip()
                if line and not line.startswith("#"):
                    if "|" in line:
                        name, description = [x.strip() for x in line.split("|", 1)]
                        cell_type_info.append((name, description))
                    else:
                        cell_type_info.append((line, None))

    if not cell_type_info:
        cell_type_info = create_default_cell_types(cell_types_file)

    return cell_type_info


def create_default_cell_types(cell_types_file):
    """Create default cell types if no file exists"""
    print(f"Warning: {cell_types_file} not found or empty, using default cell types")
    cell_type_info = [
        (
            "cryptes",
            "Crypts of Lieberkühn, also known as intestinal glands, are tubular invaginations in the intestinal epithelium",
        ),
        (
            "granulom",
            "Granulomas are organized collections of macrophages and other immune cells that form in response to inflammation",
        ),
    ]
    os.makedirs(os.path.dirname(cell_types_file), exist_ok=True)
    with open(cell_types_file, "w") as f:
        f.write(
            "# Generated by Copilot\n# List of cell types for mask annotation\n# Format: cell_name | description\n"
        )
        for name, desc in cell_type_info:
            f.write(f"{name} | {desc}\n")
    return cell_type_info


def create_or_update_cells(session, cell_type_info):
    """Create or update cell records in database"""
    cells = []
    for name, description in cell_type_info:
        cell = session.query(Cell).filter_by(name=name).first()
        if cell:
            if description and cell.description != description:
                cell.description = description
                session.add(cell)
        else:
            cell = Cell(name=name, description=description)
            session.add(cell)
            cells.append(cell)
    if cells:
        session.commit()
    return session.query(Cell).all()


def process_mask(image_masks_path, cell, h, w):
    """Process individual mask for an image"""
    mask_path = os.path.join(image_masks_path, f"{cell.name}.npy")

    if os.path.exists(mask_path):
        try:
            mask_array = np.load(mask_path)
            if mask_array.dtype != np.uint8 or mask_array.shape != (h, w):
                mask_array = np.zeros((h, w), dtype=np.uint8)
                np.save(mask_path, mask_array)
            else:
                invalid_values = mask_array > 2
                if np.any(invalid_values):
                    mask_array[invalid_values] = 0
                    np.save(mask_path, mask_array)
        except:
            mask_array = np.zeros((h, w), dtype=np.uint8)
            np.save(mask_path, mask_array)
    else:
        mask_array = np.zeros((h, w), dtype=np.uint8)
        np.save(mask_path, mask_array)

    return mask_path


def update_mask_health_status(mask_path):
    """Update health status based on mask content"""
    try:
        mask_array = np.load(mask_path)
        total_annotated = np.sum(mask_array > 0)
        if total_annotated > 0:
            healthy_ratio = np.sum(mask_array == 2) / total_annotated
            return (
                HealthStatusEnum.healthy
                if healthy_ratio > 0.5
                else HealthStatusEnum.unhealthy
            )
    except:
        pass
    return HealthStatusEnum.unhealthy


def process_image_masks(session, image, cells, image_masks_path, h, w):
    """Process all masks for a single image"""
    for cell in tqdm(cells, desc=f"Processing masks for {image.filename}", leave=False):
        mask_path = process_mask(image_masks_path, cell, h, w)

        existing_mask = (
            session.query(Mask).filter_by(image_id=image.id, cell_id=cell.id).first()
        )

        if existing_mask:
            if existing_mask.mask_path != mask_path:
                existing_mask.mask_path = mask_path
                session.add(existing_mask)
            if os.path.exists(mask_path):
                existing_mask.health_status = update_mask_health_status(mask_path)
        else:
            mask = Mask(
                image_id=image.id,
                mask_path=mask_path,
                cell_id=cell.id,
                is_mask_done=0,
                health_status=HealthStatusEnum.unhealthy,
            )
            session.add(mask)


def init_database(
    session,
    images_path="data/dataset/images",
    masks_path="data/dataset/masks",
    cell_types_file="data/cell_types.txt",
):
    """Initialize the database with images, masks, and cell types"""

    cell_type_info = load_cell_types(cell_types_file)
    cells = create_or_update_cells(session, cell_type_info)

    image_files = os.listdir(images_path)
    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(images_path, image_file)
        if os.path.isfile(image_path):
            image = Image(
                filename=image_file,
                img_path=image_path,
                diagnosis=DiagnosisEnum.unknown,
            )
            session.add(image)
            session.commit()

            img = cv2.imread(image_path)
            if img is None:
                print(f"Error reading image: {image_path}")
                continue
            h, w = img.shape[:2]

            image_masks_path = os.path.join(masks_path, os.path.splitext(image_file)[0])
            if not os.path.exists(image_masks_path):
                os.makedirs(image_masks_path)

            process_image_masks(session, image, cells, image_masks_path, h, w)
    session.commit()


from app.db.models import HealthStatusEnum

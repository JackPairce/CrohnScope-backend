"""
Training worker script for running model training in a separate process.
"""
import os
import sys
import time
import json
import logging
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data/models/training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("train_worker")

# Import the training function and types
from app.services.ai.train import start_training_if_needed
from app.types.training import ProcessInfo, TrainingStatusEnum

# Process info file path
PROCESS_INFO_FILE = "data/models/training_process.json"

def update_status(status, error=None):
    """Update the status of the training process using ProcessInfo model."""
    try:
        # Load current process info
        if os.path.exists(PROCESS_INFO_FILE):
            with open(PROCESS_INFO_FILE, 'r') as f:
                data = json.load(f)

            # Create ProcessInfo instance, preserving existing data
            process_info = ProcessInfo(**data)
        else:
            logger.error("Process info file not found, cannot update status")
            return False

        # Update status and error if provided
        process_info.status = TrainingStatusEnum(status) if isinstance(status, str) else status
        process_info.last_update = time.time()

        if error is not None:
            process_info.error = str(error)

        # Write back to file atomically
        temp_file = f"{PROCESS_INFO_FILE}.tmp"
        with open(temp_file, 'w') as f:
            json.dump(process_info.dict(), f, indent=2)

        os.replace(temp_file, PROCESS_INFO_FILE)
        return True
    except Exception as e:
        logger.error(f"Failed to update process status: {e}")
        return False

if __name__ == "__main__":
    logger.info("Training worker started")
    try:
        update_status(TrainingStatusEnum.RUNNING)
        # Run the actual training
        result = start_training_if_needed()
        if result:
            update_status(TrainingStatusEnum.COMPLETED)
            logger.info("Training completed successfully")
        else:
            update_status(TrainingStatusEnum.SKIPPED)
            logger.info("Training conditions not met, skipped")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        update_status(TrainingStatusEnum.FAILED, error=str(e))
    finally:
        # Ensure we don't leave the process in a running state
        try:
            with open(PROCESS_INFO_FILE, 'r') as f:
                process_info = json.load(f)
            if process_info.get("status") == TrainingStatusEnum.RUNNING:
                update_status(TrainingStatusEnum.FAILED, error="Process terminated unexpectedly")
        except Exception as e:
            logger.error(f"Error in cleanup: {e}")

        logger.info("Training worker exiting")

# Folder structure

backend/
├── app/
│ ├── main.py # Entry point for FastAPI
│ ├── api/
│ │ ├── routes/
│ │ │ ├── ai.py # AI predictions and training trigger
│ │ │ ├── image.py # Image preprocessing & region labeling
│ │ │ ├── data.py # Data upload, labeling, sync
│ │ │ ├── csv.py # CSV generation/reading (optional)
│ │ │ └── drive.py # Google Drive integration
│ │ └── deps.py # Shared dependencies
│ ├── core/
│ │ ├── settings.py # Config & environment variables
│ │ ├── scheduler.py # Retraining jobs, cron logic (if any)
│ │ └── logger.py # Central logging
│ ├── services/
│ │ ├── ai/
│ │ │ ├── train.py # Model training logic
│ │ │ └── predict.py # Inference logic
│ │ ├── image/
│ │ │ └── mask_processor.py# OpenCV logic for region mapping
│ │ ├── drive/
│ │ │ └── gdrive.py # Google Drive API wrapper
│ │ └── data/
│ │ └── validator.py # Data consistency, type checks
│ ├── db/
│ │ ├── models.py # DB Models (if you use PostgreSQL/SQLite)
│ │ └── crud.py # DB read/write helpers
│ └── utils/
│ ├── file_ops.py # Temporary file handling
│ └── constants.py
├── requirements.txt
└── README.md

# CrohnScope Backend

CrohnScope is an AI-powered medical imaging platform designed to assist in the analysis and diagnosis of Crohn's disease through advanced image segmentation and cell identification.

## Overview

This backend service provides:
- AI-powered image segmentation for medical images
- Real-time mask annotation and processing
- Automated model training with progress tracking
- Resource-aware processing pipeline
- Secure data management and backup

## Key Features

### AI Model Capabilities
- Automated cell identification and segmentation
- Progressive model training with preprocessing status
- Memory-aware processing (500MB threshold protection)
- GPU acceleration support with CUDA
- Real-time training progress monitoring

### Image Processing
- Advanced mask generation and validation
- Support for multiple cell types
- Three-state mask encoding (background, unhealthy, healthy)
- Efficient patch-based processing
- Progress tracking for large operations

### Data Management
- Automated database backups
- State preservation and restoration
- Efficient image and mask storage
- Progress tracking for data operations

## Project Structure

```
backend/
├── app/
│   ├── main.py                 # FastAPI application entry point
│   ├── api/
│   │   └── routes/
│   │       ├── ai.py          # AI model endpoints
│   │       ├── cells.py       # Cell type management
│   │       ├── image.py       # Image processing
│   │       ├── mask.py        # Mask management
│   │       └── monitoring.py   # System monitoring
│   ├── core/
│   │   ├── config.py         # Configuration management
│   │   ├── init.py          # Database initialization
│   │   └── logger.py        # Logging configuration
│   ├── services/
│   │   ├── ai/
│   │   │   ├── auto_mask.py   # Automated mask generation
│   │   │   ├── models.py      # Neural network models
│   │   │   ├── preprocessing.py# Data preprocessing
│   │   │   ├── predict.py     # Model inference
│   │   │   ├── scheduler.py   # Training scheduler
│   │   │   └── train.py       # Model training logic
│   │   ├── cell/             # Cell type services
│   │   ├── image/            # Image processing services
│   │   └── monitoring/       # System monitoring
│   ├── db/
│   │   ├── models.py         # Database models
│   │   ├── crud.py          # Database operations
│   │   └── session.py       # Database connection
│   └── types/
│       ├── image.py         # Image-related types
│       └── monitor.py       # Monitoring types
├── data/
│   ├── backups/            # Database backups
│   ├── dataset/           # Training dataset
│   │   ├── images/       # Source images
│   │   └── masks/        # Annotation masks
│   └── models/           # Trained model storage
├── utils/
│   ├── converters.py    # Data conversion utilities
│   └── file_ops.py      # File operations
├── test/                # Testing notebooks
└── requirements.txt     # Project dependencies
````

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/CrohnScope-backend.git
cd CrohnScope-backend
```

2. Create and activate a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Usage

1. Start the server:
```bash
uvicorn app.main:app --reload
```

2. Initialize the database (first run):
The system will automatically:
- Initialize the database schema
- Create necessary tables
- Set up default cell types
- Create initial backup

3. Access the API:
- API documentation: http://localhost:8000/docs
- OpenAPI spec: http://localhost:8000/openapi.json

## API Endpoints

### AI Operations
- `POST /ai/train`: Trigger model training
- `GET /ai/status`: Get training status and progress
- `POST /ai/predict`: Generate predictions for new images

### Image Management
- `POST /images/`: Upload new images
- `GET /images/`: List available images
- `GET /images/{id}/masks`: Get image masks
- `POST /images/{id}/masks`: Update image masks

### Cell Types
- `GET /cells/`: List available cell types
- `POST /cells/`: Add new cell type
- `PUT /cells/{id}`: Update cell type

### Monitoring
- `GET /monitoring/status`: Get system status
- `GET /monitoring/resources`: Get resource usage

## Development

### Running Tests
```bash
pytest
```

### Docker Support
```bash
# Build the image
docker build -t crohnscope-backend .

# Run the container
docker run -p 8000:8000 crohnscope-backend
```

## Memory Management

The system includes built-in memory protection:
- Monitors available system memory
- Stops processing if memory drops below 500MB
- Provides progress tracking for long operations
- Implements efficient patch-based processing

## Backup and Recovery

The system automatically:
- Creates backups before shutdown
- Maintains backup history
- Supports state restoration
- Validates backup integrity

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

# MLOps Item Recognizer Project

## Progress Overview

### ✅ HW1: Label Studio + DVC Integration
- Set up Label Studio for image classification
- Configured MinIO buckets:
  - `items-dataset`: Raw images storage
  - `items-labeled-dataset`: Label Studio annotations
  - `items-dvc-storage`: DVC versioned data
- Implemented data versioning with DVC
- Created export script for Label Studio annotations

**Setup Instructions:**
```bash
# Python environment setup
pyenv shell 3.10
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Start Label Studio and MinIO
cd data/
docker compose up -d

# DVC setup
dvc init
dvc remote add -d storage s3://items-dvc-storage
dvc remote modify storage endpointurl http://localhost:9000
dvc add data/dataset

# Configure MinIO credentials
export MINIO_ACCESS_KEY="minioadmin"
export MINIO_SECRET_KEY="minioadmin"
dvc remote modify storage --local access_key_id ${MINIO_ACCESS_KEY}
dvc remote modify storage --local secret_access_key ${MINIO_SECRET_KEY}
```
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

### 🔄 HW2: Ray + W&B Integration (In Progress)
- Implemented distributed CNN training using Ray
- Integrated Weights & Biases for experiment tracking
- Key features:
  - Distributed batch processing
  - Model artifact logging
  - Training metrics visualization
  - Hyperparameter tracking

**Setup Instructions:**
```bash
# Install Ray and W&B dependencies
pip install -r requirements.txt

# Start Ray cluster using Kubernetes
chmod +x k8s/setup_cluster.sh
./k8s/setup_cluster.sh

# Configure W&B (create .env file)
cat > model/cnn-cpu/.env << EOL
WANDB_API_KEY=your_api_key
WANDB_ENTITY=your_entity
WANDB_PROJECT=item-recognizer
EOL

# Run training job
cd model/cnn-cpu
python submit_ray_job.py

# Monitor Ray dashboard (port forwarded by setup script)
open http://localhost:8265
```

### HW3: Model Serving (Pending)
- Plan to implement model serving infrastructure
- Will add deployment instructions here

### HW4: Monitoring (Pending)
- Will implement monitoring solutions
- Will add setup instructions here

### HW5: CI/CD (Pending)
- Will implement CI/CD pipeline
- Will add pipeline configuration details here
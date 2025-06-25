from dotenv import load_dotenv
import os
import ray
from ray import serve
from serve_app import entrypoint

# Load environment variables from .env
load_dotenv()

# Initialize Ray
ray_address = os.getenv("RAY_ADDRESS")
print(f"Connecting to Ray cluster at {ray_address}...")

ray.init(
    address=ray_address,  # Use RAY_ADDRESS directly
    runtime_env={
        "pip": [
            "tensorflow==2.13.0",
            "keras==2.13.1",
            "wandb==0.16.6",
            "fastapi==0.95.2",
            "uvicorn==0.22.0",
            "Pillow==10.2.0",
            "python-multipart==0.0.9",
            "python-dotenv==1.0.1"
        ],
        "env_vars": {
            "WANDB_API_KEY": os.getenv("WANDB_API_KEY"),
            "WANDB_PROJECT": os.getenv("WANDB_PROJECT"),
            "WANDB_ENTITY": os.getenv("WANDB_ENTITY"),
            "WANDB_MODEL_ARTIFACT": os.getenv("WANDB_MODEL_ARTIFACT"),
            "WANDB_MODEL_FILENAME": os.getenv("WANDB_MODEL_FILENAME"),
            "WANDB_MODE": os.getenv("WANDB_MODE", "online"),
            "CLASS_NAMES": os.getenv("CLASS_NAMES")
        }
    }
)

# Run the deployment
serve.run(entrypoint, name="cnn-classifier")
print("✅ Ray Serve app is running")

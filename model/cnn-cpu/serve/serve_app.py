import os
import tempfile
import tensorflow as tf
import wandb
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ray import serve
from ray.serve.handle import DeploymentHandle

load_dotenv()

app = FastAPI()

@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": 1},
)
@serve.ingress(app)
class APIIngress:
    def __init__(self, model_handle):
        self.handle: DeploymentHandle = model_handle.options(
            use_new_handle_api=True
        )

    @app.post("/predict")
    async def predict(self, file: UploadFile = File(...)):
        try:
            print(f"Received prediction request for file {file.filename}")
            image_bytes = await file.read()
            print(f"Read {len(image_bytes)} bytes from file")
            result = await self.handle.predict.remote(image_bytes)
            print(f"Got prediction result: {result}")
            return JSONResponse(content=result)
        except Exception as e:
            print(f"Error in predict endpoint: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return JSONResponse(content={"error": str(e)}, status_code=500)

@serve.deployment(
    autoscaling_config={"min_replicas": 1, "max_replicas": 2},
    ray_actor_options={"num_cpus": 1},
)
class CNNModel:
    def __init__(self):
        class_names_str = os.getenv("CLASS_NAMES", "book,stamp,coin,drumstick,vinyl")
        print(f"Loading class names from env: {class_names_str}")
        self.class_names = class_names_str.split(",")
        print(f"Initialized class names: {self.class_names}")

        print("Initializing W&B...")
        print(f"Project: {os.getenv('WANDB_PROJECT')}")
        print(f"Entity: {os.getenv('WANDB_ENTITY')}")
        print(f"Artifact: {os.getenv('WANDB_MODEL_ARTIFACT')}")
        
        try:
            run = wandb.init(
                project=os.getenv("WANDB_PROJECT"),
                entity=os.getenv("WANDB_ENTITY"),
                job_type="inference",
                mode="online"
            )
            print("✅ W&B initialized")

            print("Fetching model artifact...")
            artifact = run.use_artifact(os.getenv("WANDB_MODEL_ARTIFACT"), type="model")
            print("✅ Artifact found")
            
            print("Downloading artifact...")
            path = artifact.download()
            print(f"✅ Artifact downloaded to {path}")

            print("Looking for model file...")
            print(f"Loading SavedModel from {path}...")
            self.model = tf.keras.models.load_model(path, compile=False)
            print("✅ Model loaded successfully")

        except Exception as e:
            print(f"❌ Failed to load W&B model: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            print("Traceback:")
            print(traceback.format_exc())
            raise

        finally:
            print("Cleaning up W&B...")
            wandb.finish()
            print("✅ W&B finished")

    async def predict(self, image_bytes: bytes):
        print("Starting prediction...")
        try:
            if self.model is None:
                print("Model is not initialized, attempting to load...")
                await self.reconfigure()
                if self.model is None:
                    raise ValueError("Failed to initialize model")

            print("Writing image to temp file...")
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(image_bytes)
                tmp_path = tmp.name
                print(f"Image written to {tmp_path}")

            print("Loading and preprocessing image...")
            image = tf.keras.utils.load_img(tmp_path, target_size=(128, 128))
            print("Image loaded successfully")
            image = tf.convert_to_tensor(image)
            print("Converted to tensor")
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            print("Converted to float32")
            image = tf.expand_dims(image, axis=0)
            print("Added batch dimension")

            print("Running prediction...")
            predictions = self.model.predict(image, verbose=1)[0]
            print(f"Raw predictions shape: {predictions.shape}")
            print(f"Raw predictions: {predictions}")
            print(f"Class names: {self.class_names}")
            idx = tf.argmax(predictions).numpy()
            print(f"Predicted class index: {idx}")
            result = {
                "predicted_class": self.class_names[idx],
                "confidence": float(predictions[idx])
            }
            print(f"Final result: {result}")
            return result

        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            print("Traceback:")
            print(traceback.format_exc())
            raise

entrypoint = APIIngress.bind(CNNModel.bind())

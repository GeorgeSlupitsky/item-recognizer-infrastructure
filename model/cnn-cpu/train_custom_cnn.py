import os
import json
import yaml
import numpy as np
import tensorflow as tf
import ray
from ray import train
from ray.air import session
from ray.air.config import ScalingConfig, RunConfig

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow.keras as keras
from tensorflow.keras import utils as keras_utils
from tensorflow.keras import preprocessing as keras_preprocessing
from tensorflow.keras import layers as keras_layers
from tensorflow.keras import models as keras_models
from tensorflow.keras import optimizers as keras_optimizers
from tensorflow.keras import callbacks as keras_callbacks
from tensorflow.keras.optimizers import Adam
from dotenv import load_dotenv
import wandb
from PIL import Image
import gc
from datetime import datetime

# === ENV ===
load_dotenv()

# === CONFIG ===
def load_config(config_path='/data/model/cnn-cpu/config.yaml'):
    print(f"🔍 Loading config from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        print(f"✅ Config loaded: {config}")
    
    # Convert img_size list to tuple for TF compatibility
    if isinstance(config.get('img_size'), list):
        config['img_size'] = tuple(config['img_size'])
    
    # Set defaults for missing values
    defaults = {
        'json_path': 'data/label_studio_export.json',
        'model_save_path': 'best_custom_cnn.keras',
        'img_size': (224, 224),
        'batch_size': 32,
        'epochs': 20,
        'num_workers': 4,
        'run_name': 'custom-cnn-ray'
    }
    
    for key, value in defaults.items():
        if key not in config:
            config[key] = value
    
    return config

CONFIG = load_config()

# === LOAD DATA ===
def load_dataset(config):
    with open(config['json_path'], 'r') as f:
        data = json.load(f)
    
    X, y = [], []
    label_set = set()
    processed_files = set()
    
    print(f'Loading {len(data)} items from JSON')
    for item in data:
        if 'data' in item and 'image' in item['data'] and 'annotations' in item:
            # Convert S3 path to local path
            s3_path = item['data']['image']
            # Get filename from s3://items-dataset/filename.jpg
            filename = s3_path.replace('s3://items-dataset/', '')
            
            # Skip if we've already processed this file
            if filename in processed_files:
                print(f'Skipping duplicate file: {filename}')
                continue
            
            # Get category from the label
            label = item['annotations'][0]['result'][0]['value']['choices'][0]
            category = label.lower() + 's'
            # Since we're running in /data/model/cnn-cpu, we need to go up to /data
            local_path = os.path.join('/data/dataset', category, filename)
            
            try:
                # Load and process image
                img = tf.keras.utils.load_img(local_path, target_size=config['img_size'])
                img = tf.keras.utils.img_to_array(img) / 255.0
                label = item['annotations'][0]['result'][0]['value']['choices'][0]
                X.append(img)
                y.append(label)
                label_set.add(label)
                processed_files.add(filename)
                print(f'Successfully loaded {filename} with label {label}')
            except Exception as e:
                print(f"Error processing {local_path}: {e}")
                continue
    
    label_names = sorted(list(label_set))
    label_map = {name: idx for idx, name in enumerate(label_names)}
    y = [label_map[label] for label in y]
    
    # Save dataset to W&B if enabled
    if use_wandb:
        try:
            # Create dataset artifact
            artifact = wandb.Artifact(
                name=f"dataset-{wandb.run.name}",
                type="dataset",
                description="Item recognition dataset with images and labels"
            )
            
            # Add dataset metadata
            artifact.metadata = {
                "num_samples": len(X),
                "num_classes": len(label_set),
                "class_names": label_names,
                "image_size": config['img_size']
            }
            
            # Log artifact to W&B
            wandb.log_artifact(artifact)
            # Link dataset to registry
            artifact = wandb.run.summary['logged_artifacts'][-1]
            run.link_artifact(artifact, f"wandb-registry-dataset/item-recognizer-datasets")
            print("✅ Dataset metadata saved and linked to W&B registry")
        except Exception as e:
            print(f"⚠️ Failed to save dataset to W&B: {e}")
    
    # Print dataset statistics
    print('\nDataset statistics:')
    print(f'Total images loaded: {len(X)}')
    print('Label distribution:')
    for label in label_names:
        count = y.count(label_map[label])
        print(f'  {label}: {count} images')
    
    return np.array(X), np.array(y), label_names

def prepare_data(config):
    X, y, label_names = load_dataset(config)
    X_train, X_test, y_train_idx, y_test_idx = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )
    
    y_train_cat = tf.keras.utils.to_categorical(y_train_idx, num_classes=len(label_names))
    y_test_cat = tf.keras.utils.to_categorical(y_test_idx, num_classes=len(label_names))
    
    y_train = np.argmax(y_train_cat, axis=1)
    class_weights_array = compute_class_weight(
        class_weight='balanced', classes=np.unique(y_train), y=y_train
    )
    class_weights = dict(enumerate(class_weights_array))
    
    return X_train, X_test, y_train_cat, y_test_cat, class_weights, len(label_names)

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# === MODEL ===
def build_cnn(num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(*CONFIG["img_size"], 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

@ray.remote
class DataWorker:
    def __init__(self, config):
        self.config = config
    
    def prepare_batch(self, start_idx, end_idx, X, y):
        """Prepare a batch of data for training"""
        batch_X = X[start_idx:end_idx]
        batch_y = y[start_idx:end_idx]
        
        # Apply data augmentation
        augmented = []
        for img in batch_X:
            if np.random.random() > 0.5:  # 50% chance of augmentation
                img = datagen.random_transform(img)
            augmented.append(img)
        
        return np.array(augmented), batch_y

@ray.remote
class DataWorker:
    def __init__(self, config):
        self.config = config
    
    def prepare_batch(self, start_idx, end_idx, X_train, y_train):
        X_batch = X_train[start_idx:end_idx]
        y_batch = y_train[start_idx:end_idx]
        return X_batch, y_batch

@ray.remote
class Trainer:
    def __init__(self, config, num_classes):
        self.config = config
        self.model = build_cnn(num_classes)
        self.model.compile(optimizer=tf.keras.optimizers.legacy.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    def train_on_batch(self, X_batch, y_batch):
        return self.model.train_on_batch(X_batch, y_batch)
    
    def get_weights(self):
        return self.model.get_weights()
    
    def set_weights(self, weights):
        self.model.set_weights(weights)
    
    def save_model(self):
        model_path = os.path.join('/data/model/cnn-cpu', CONFIG['model_save_path'])
        self.model.save(model_path)
        print(f"✅ Model saved to {model_path}")
        
        # Save to W&B if enabled
        if use_wandb:
            try:
                # Create model artifact
                artifact = wandb.Artifact(
                    name=f"item-recognizer-model",
                    type="model",
                    description="Custom CNN model for item recognition"
                )
                
                # Add model file to artifact
                artifact.add_file(model_path)
                
                # Log artifact to W&B
                wandb.log_artifact(artifact)
                
                # Link model to registry and mark as latest
                artifact.wait()
                artifact.aliases.append('latest')
                artifact.save()
                print("✅ Model saved and linked to W&B registry")
            except Exception as e:
                print(f"⚠️ Failed to save model to W&B: {e}")
        
        # Finish W&B run
        if use_wandb:
            wandb.finish()

def train_distributed(config):
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(address=os.environ.get("RAY_ADDRESS", "auto"))
    print("🔍 Ray initialized with resources:", ray.available_resources())
    
    # Initialize wandb if configured
    global use_wandb
    use_wandb = all([os.getenv(k) for k in ["WANDB_API_KEY", "WANDB_PROJECT", "WANDB_ENTITY"]])
    if use_wandb:
        try:
            wandb.init(
                project=os.getenv("WANDB_PROJECT"),
                entity=os.getenv("WANDB_ENTITY"),
                name=os.getenv("WANDB_RUN_NAME", "cnn-training"),
                job_type="training",
                config=config
            )
            print("✅ W&B initialized successfully")
            
            # Create dataset artifact
            dataset_artifact = wandb.Artifact(
                name="item-recognition-dataset",
                type="dataset",
                description="Item recognition dataset with images and labels"
            )
            dataset_artifact.add_dir("/data/dataset")
            wandb.log_artifact(dataset_artifact)
            print("✅ Dataset logged to W&B")
        except Exception as e:
            print(f"⚠️ W&B initialization failed: {e}")
            use_wandb = False
    
    # Get the data
    X_train, X_test, y_train_cat, y_test_cat, class_weights, num_classes = prepare_data(config)
    
    # Calculate batch parameters
    batch_size = config['batch_size']
    num_batches = len(X_train) // batch_size
    indices = np.random.permutation(len(X_train))
    
    # Create workers
    num_workers = config['num_workers']
    data_workers = [DataWorker.remote(config) for _ in range(num_workers)]
    
    # Create trainer actor
    print("🚀 Creating trainer actor...")
    trainer = Trainer.remote(config, num_classes)
    print("✅ Trainer actor created")
    
    # Training loop
    batch_size = config['batch_size']
    num_batches = len(X_train) // batch_size
    
    for epoch in range(config["epochs"]):
        print(f"Epoch {epoch + 1}/{config['epochs']}")
        print("🔍 Ray resources:", ray.available_resources())
        
        # Distribute batches to workers
        for batch in range(0, num_batches, num_workers):
            batch_tasks = []
            
            # Prepare batches in parallel
            for i in range(num_workers):
                if batch + i < num_batches:
                    start_idx = (batch + i) * batch_size
                    end_idx = start_idx + batch_size
                    batch_tasks.append(
                        data_workers[i].prepare_batch.remote(
                            start_idx, end_idx, X_train, y_train_cat
                        )
                    )
            
            # Get prepared batches
            prepared_batches = ray.get(batch_tasks)
            
            # Train on batches
            for X_batch, y_batch in prepared_batches:
                loss = ray.get(trainer.train_on_batch.remote(X_batch, y_batch))
                print(f"Batch loss: {loss[0]:.4f}")
                
                # Log metrics to W&B
                if use_wandb:
                    wandb.log({
                        "batch_loss": loss[0],
                        "batch_accuracy": loss[1]
                    })
        
        # Evaluate on test set
        model_weights = ray.get(trainer.get_weights.remote())
        temp_model = build_cnn(num_classes)
        temp_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
        temp_model.set_weights(model_weights)
        test_loss = temp_model.evaluate(X_test, y_test_cat, verbose=0)
        print(f"Test loss: {test_loss[0]:.4f}, Test accuracy: {test_loss[1]:.4f}")
        
        # Log test metrics to W&B
        if use_wandb:
            wandb.log({
                "test_loss": test_loss[0],
                "test_accuracy": test_loss[1],
                "epoch": epoch + 1
            })
    
    # Save the final model
    ray.get(trainer.save_model.remote())
    return {"status": "success"}

def main():
    # Load config
    config = load_config()
    
    # Initialize Ray
    ray.init()
    
    # Run distributed training
    result = train_distributed(config)
    
    print("Training completed!")
    ray.shutdown()

if __name__ == "__main__":
    main()


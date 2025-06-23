# ray_cnn_wandb.py

import os
import json
import yaml
import numpy as np
import tensorflow as tf
import ray
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.optimizers import Adam
from dotenv import load_dotenv
import wandb
import traceback
from ray import train
from ray.air import session

# === ENV ===
load_dotenv()
os.environ["WANDB_MODE"] = "online"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# === CONFIG ===
def load_config(config_path='/data/model/cnn-cpu/config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    if isinstance(config.get('img_size'), list):
        config['img_size'] = tuple(config['img_size'])
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
        config.setdefault(key, value)
    return config

CONFIG = load_config()

# === DATA ===
def load_dataset(config):
    with open(config['json_path'], 'r') as f:
        data = json.load(f)

    X, y, label_set, processed_files = [], [], set(), set()

    for item in data:
        if 'data' in item and 'image' in item['data'] and 'annotations' in item:
            filename = item['data']['image'].replace('s3://items-dataset/', '')
            if filename in processed_files:
                continue
            label = item['annotations'][0]['result'][0]['value']['choices'][0]
            category = label.lower() + 's'
            local_path = os.path.join('/data/dataset', category, filename)
            try:
                img = tf.keras.utils.load_img(local_path, target_size=config['img_size'])
                img = tf.keras.utils.img_to_array(img) / 255.0
                X.append(img)
                y.append(label)
                label_set.add(label)
                processed_files.add(filename)
            except:
                continue

    label_names = sorted(list(label_set))
    label_map = {name: idx for idx, name in enumerate(label_names)}
    y = [label_map[label] for label in y]
    return np.array(X), np.array(y), label_names

def prepare_data(config):
    X, y, label_names = load_dataset(config)
    X_train, X_test, y_train_idx, y_test_idx = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )
    y_train_cat = tf.keras.utils.to_categorical(y_train_idx, num_classes=len(label_names))
    y_test_cat = tf.keras.utils.to_categorical(y_test_idx, num_classes=len(label_names))
    class_weights_array = compute_class_weight('balanced', classes=np.unique(y_train_idx), y=y_train_idx)
    class_weights = dict(enumerate(class_weights_array))
    return X_train, X_test, y_train_cat, y_test_cat, class_weights, len(label_names)

# === MODEL ===
def build_cnn(num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(*CONFIG['img_size'], 3)),
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

# === ACTORS ===
@ray.remote
class DataWorker:
    def __init__(self, config):
        self.config = config

    def prepare_batch(self, start_idx, end_idx, X, y):
        return X[start_idx:end_idx], y[start_idx:end_idx]

@ray.remote
class Trainer:
    def __init__(self, config, num_classes):
        self.config = config
        print(f"✅ Trainer actor started on PID {os.getpid()}")
        self.model = build_cnn(num_classes)
        self.model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    def train_on_batch(self, X_batch, y_batch):
        try:
            return self.model.train_on_batch(X_batch, y_batch)
        except Exception as e:
            print("❌ Trainer crashed during batch training:")
            traceback.print_exc()
            raise e

    def get_weights(self):
        return self.model.get_weights()

# === TRAINING ===
def train_distributed(config):
    X_train, X_test, y_train_cat, y_test_cat, class_weights, num_classes = prepare_data(config)
    batch_size = config['batch_size']
    num_batches = len(X_train) // batch_size
    indices = np.random.permutation(len(X_train))

    data_workers = [DataWorker.remote(config) for _ in range(config['num_workers'])]
    trainer = Trainer.options(max_restarts=1, max_task_retries=1).remote(config, num_classes)

    for epoch in range(config['epochs']):
        for batch in range(0, num_batches, config['num_workers']):
            batch_tasks = []
            for i in range(config['num_workers']):
                if batch + i < num_batches:
                    start = (batch + i) * batch_size
                    end = start + batch_size
                    batch_tasks.append(data_workers[i].prepare_batch.remote(start, end, X_train, y_train_cat))
            prepared = ray.get(batch_tasks)
            for X_batch, y_batch in prepared:
                loss = ray.get(trainer.train_on_batch.remote(X_batch, y_batch))
                if use_wandb:
                    wandb.log({"batch_loss": loss[0], "batch_accuracy": loss[1]})
        temp_model = build_cnn(num_classes)
        temp_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
        temp_model.set_weights(ray.get(trainer.get_weights.remote()))
        test_loss = temp_model.evaluate(X_test, y_test_cat, verbose=0)
        if use_wandb:
            wandb.log({"test_loss": test_loss[0], "test_accuracy": test_loss[1], "epoch": epoch + 1})

    return trainer, num_classes, X_test, y_test_cat

# === MAIN ===
def main():
    try:
        config = load_config()
        ray.init(address=os.environ.get("RAY_ADDRESS", "auto"), ignore_reinit_error=True)

        global use_wandb
        use_wandb = all([os.getenv(k) for k in ["WANDB_API_KEY", "WANDB_PROJECT", "WANDB_ENTITY"]])
        if use_wandb:
            wandb.init(
                project=os.getenv("WANDB_PROJECT"),
                entity=os.getenv("WANDB_ENTITY"),
                name=os.getenv("WANDB_RUN_NAME", "cnn-training"),
                job_type="training",
                config=config
            )

        trainer, num_classes, X_test, y_test_cat = train_distributed(config)

        print("📥 Getting final model weights...")
        trained_weights = ray.get(trainer.get_weights.remote())
        final_model = build_cnn(num_classes)
        final_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
        final_model.set_weights(trained_weights)

        model_path = os.path.join('/data/model/cnn-cpu', config['model_save_path'])
        final_model.save(model_path)
        print(f"✅ Model saved locally at {model_path}")

        if use_wandb:
            try:
                artifact = wandb.Artifact(
                    name="item-recognizer-model",
                    type="model",
                    description="Custom CNN model for item classification",
                    metadata={"img_size": config["img_size"], "epochs": config["epochs"], "run_name": config["run_name"]}
                )
                artifact.add_file(model_path)
                wandb.log_artifact(artifact, aliases=["latest", "v1"])
                artifact.wait()
                print("✅ Model artifact logged to W&B")
            except Exception as e:
                print(f"⚠️ Failed to log model artifact: {e}")

            try:
                dataset_artifact = wandb.Artifact(
                    name="label-studio-dataset",
                    type="dataset",
                    description="Label Studio export in JSON format",
                    metadata={"source": config["json_path"]}
                )
                dataset_artifact.add_file(config["json_path"])
                wandb.log_artifact(dataset_artifact, aliases=["latest"])
                dataset_artifact.wait()
                print("✅ Dataset artifact logged to W&B")
            except Exception as e:
                print(f"⚠️ Failed to log dataset artifact: {e}")

            wandb.finish()

        print("🏁 Training & export completed!")
        ray.shutdown()

    except Exception as e:
        print("❌ Training failed with exception:")
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    main()

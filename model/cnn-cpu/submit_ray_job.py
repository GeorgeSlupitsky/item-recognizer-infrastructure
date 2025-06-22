import os
import ray
import yaml
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
logging.getLogger("ray").setLevel(logging.WARNING)

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def check_required_files():
    local_files = [
        "train_custom_cnn.py",
        "requirements.txt",
        "ray_job.py",
        "config.yaml",
        "../../data/label_studio_export.json"
    ]
    missing = [f for f in local_files if not Path(f).exists()]
    if missing:
        print(f"❌ Missing local files: {missing}")
        return False
    print("✅ All required files found")
    return True

def prepare_job_files():
    local_files = [
        "train_custom_cnn.py",
        "requirements.txt",
        "ray_job.py",
        "config.yaml"
    ]
    data = {}
    for file in local_files:
        with open(file, 'r') as f:
            data[f"/data/model/cnn-cpu/{file}"] = f.read()
    
    # Add data file with proper path handling
    data_file = "../../data/label_studio_export.json"
    if os.path.exists(data_file):
        with open(data_file, 'r') as f:
            data["/data/label_studio_export.json"] = f.read()
    else:
        print(f"⚠️ Data file not found: {data_file}")
        return None
    
    return data

@ray.remote
def run_ray_job(file_contents):
    print("🔍 Ray job environment variables:")
    for k, v in sorted(os.environ.items()):
        if k.startswith('WANDB_'):
            print(f"  {k}: {'[SET]' if v else '[NOT SET]'}")
    import tempfile, subprocess, sys
    temp_dir = "/data/model/cnn-cpu"
    os.makedirs(temp_dir, exist_ok=True)
    os.chdir(temp_dir)
    
    # Create necessary directories
    os.makedirs('/data/model/cnn-cpu/data', exist_ok=True)
    
    # Write files
    for name, content in file_contents.items():
        dir_path = os.path.dirname(name)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(name, 'w') as f:
            f.write(content)
    

    
    # Run the job
    print("🚀 Running ray_job.py...")
    try:
        process = subprocess.Popen(
            [sys.executable, "/data/model/cnn-cpu/ray_job.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Use select to handle both stdout and stderr without blocking
        import select
        outputs = [process.stdout, process.stderr]
        while outputs:
            readable, _, _ = select.select(outputs, [], [])
            for output in readable:
                line = output.readline()
                if line:
                    if output == process.stdout:
                        print(line.strip())
                    else:
                        print("ERROR:", line.strip())
                else:
                    outputs.remove(output)
            
            # Check if process has finished
            if process.poll() is not None and not outputs:
                break
        
        # Get return code
        returncode = process.wait()
        if returncode != 0:
            print(f"❌ Process exited with code {returncode}")
            return False
        return True
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    print("🚀 Submitting Custom CNN Job to Ray")
    ray_address = "ray://localhost:10001"

    if not check_required_files():
        return

    wandb_env = {
        'WANDB_API_KEY': os.getenv('WANDB_API_KEY'),
        'WANDB_PROJECT': os.getenv('WANDB_PROJECT'),
        'WANDB_ENTITY': os.getenv('WANDB_ENTITY')
    }

    # Check if any required env vars are missing
    missing_vars = [k for k, v in wandb_env.items() if not v]
    if missing_vars:
        print(f"❌ Missing required environment variables: {missing_vars}")
        return

    for key, val in wandb_env.items():
        print(f"{'✅' if val else '⚠️'} {key} is {'set' if val else 'not set'}")

    try:
        ray.init(address=ray_address)
        print(f"✅ Connected to Ray cluster at {ray_address}")
    except Exception as e:
        print(f"❌ Failed to connect to Ray cluster: {e}")
        return

    config = load_config()
    run_name = os.getenv('WANDB_RUN_NAME') or f"{config['run_name']}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    env_vars = {**{k: v for k, v in wandb_env.items() if v}, "WANDB_RUN_NAME": run_name}
    runtime_env = {"env_vars": env_vars}
    files = prepare_job_files()
    task = run_ray_job.options(runtime_env=runtime_env).remote(files)
    job_id = ray.get_runtime_context().job_id
    print(f"🕒 Waiting for task result... (Job ID: {job_id})")
    success = ray.get(task)

    if success:
        print("🎉 Training completed successfully")
        print("🌐 Check W&B dashboard at: https://wandb.ai")
    else:
        print("❌ Training failed")

    ray.shutdown()

if __name__ == "__main__":
    main()


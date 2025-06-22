#!/usr/bin/env python3

import os
import sys
import subprocess
from pathlib import Path

def setup_environment():
    print("🔍 Current working directory:", os.getcwd())
    print("🔍 Directory contents:", os.listdir())
    print("🔍 Environment variables:")
    for k, v in sorted(os.environ.items()):
        if k.startswith('WANDB_'):
            print(f"  {k}: {'[SET]' if v else '[NOT SET]'}")
    
    from dotenv import load_dotenv
    load_dotenv()
    print("🔍 Checking required environment variables...")
    required_vars = ['WANDB_API_KEY', 'WANDB_PROJECT', 'WANDB_ENTITY']
    for var in required_vars:
        if var in os.environ:
            print(f"✅ {var} is set")
        else:
            print(f"❌ {var} is not set")
            return False
    return True

def run_training():
    print("🚀 Starting custom CNN training...")
    try:
        print("🔍 Current working directory:", os.getcwd())
        print("🔍 Directory contents:", os.listdir())
        print("🔍 Training script exists:", os.path.exists("/data/model/cnn-cpu/train_custom_cnn.py"))
        print("🔍 Data file exists:", os.path.exists("/data/label_studio_export.json"))
        
        process = subprocess.Popen(
            [sys.executable, "/data/model/cnn-cpu/train_custom_cnn.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1
        )
        
        # Stream output in real-time
        while True:
            stdout_line = process.stdout.readline()
            stderr_line = process.stderr.readline()
            
            if stdout_line:
                print(stdout_line.strip())
            if stderr_line:
                print("ERROR:", stderr_line.strip())
            
            # Check if process has finished
            if process.poll() is not None:
                break
        
        # Get remaining output
        remaining_stdout, remaining_stderr = process.communicate()
        if remaining_stdout:
            print(remaining_stdout)
        if remaining_stderr:
            print("ERROR:", remaining_stderr)
        
        return process.returncode == 0
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    try:
        print("🤖 Ray Job: Custom CNN Training")
        if not setup_environment():
            print("❌ Environment setup failed")
            sys.exit(1)
            
        # Check if we're in the right directory
        print("🔍 Current working directory:", os.getcwd())
        print("🔍 Directory contents:", os.listdir())
        
        # Check if required files exist
        required_files = [
            "/data/model/cnn-cpu/train_custom_cnn.py",
            "/data/label_studio_export.json"
        ]
        for file in required_files:
            if not os.path.exists(file):
                print(f"❌ Required file not found: {file}")
                sys.exit(1)
            else:
                print(f"✅ Found required file: {file}")
        
        if not run_training():
            print("❌ Training failed")
            sys.exit(1)
            
        print("✅ Training completed successfully")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Unexpected error in main: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

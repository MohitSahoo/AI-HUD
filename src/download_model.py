import os
import urllib.request
import zipfile
import sys

def download_progress(count, block_size, total_size):
    """Show download progress"""
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write(f"\rDownloading YOLOv8 model: {percent}% complete")
    sys.stdout.flush()

def download_yolo_model():
    """Download YOLOv8 nano model if it doesn't exist"""
    model_path = "yolov8n.pt"
    
    if os.path.exists(model_path):
        print(f"Model already exists at {model_path}")
        return model_path
    
    print("Downloading YOLOv8 nano model...")
    url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
    
    try:
        urllib.request.urlretrieve(url, model_path, download_progress)
        print(f"\nModel downloaded successfully to {model_path}")
        return model_path
    except Exception as e:
        print(f"\nError downloading model: {e}")
        print("Please download the model manually from:")
        print(url)
        print(f"And save it to {os.path.abspath(model_path)}")
        return None

if __name__ == "__main__":
    download_yolo_model() 
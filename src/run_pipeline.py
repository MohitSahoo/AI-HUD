import os
import sys
import time
import subprocess

def run_command(command, description):
    """Run a command and print its output in real-time"""
    print(f"\n{'='*50}")
    print(f"STEP: {description}")
    print(f"{'='*50}")
    
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    # Print output in real-time
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    
    # Wait for the process to finish
    process.wait()
    
    if process.returncode != 0:
        print(f"\nERROR: {description} failed with return code {process.returncode}")
        return False
    
    print(f"\nSUCCESS: {description} completed successfully")
    return True

def main():
    print("Driver Drowsiness Detection Pipeline")
    print("===================================")
    
    # Step 1: Download the base model
    if not run_command("python download_model.py", "Download base YOLOv8 model"):
        return
    
    # Step 2: Train the model
    train = input("\nDo you want to train the model? (y/n): ").strip().lower()
    if train == 'y':
        if not run_command("python train_yolo_model.py", "Train YOLOv8 model"):
            return
    
    # Step 3: Run drowsiness detection
    run_detection = input("\nDo you want to run drowsiness detection? (y/n): ").strip().lower()
    if run_detection == 'y':
        if not run_command("python drowsiness_detection.py", "Run drowsiness detection"):
            return
    
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main() 
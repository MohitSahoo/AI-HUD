import os
import yaml
import shutil
from ultralytics import YOLO
from download_model import download_yolo_model

def prepare_dataset():
    """
    Prepare the dataset for YOLOv8 training by creating a proper directory structure
    and updating the data.yaml file.
    """
    # Define paths
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "Drowsiness Detection.v2-augmented-v1.yolov5-obb")
    dataset_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "drowsiness_dataset")
    
    # Create dataset directory if it doesn't exist
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Create train, val, and test directories
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(dataset_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, split, 'labels'), exist_ok=True)
    
    # Map the original directory names to new ones
    dir_mapping = {
        'train': 'train',
        'valid': 'val',
        'test': 'test'
    }
    
    # Process each split
    for orig_split, new_split in dir_mapping.items():
        orig_img_dir = os.path.join(base_dir, orig_split, 'images')
        orig_label_dir = os.path.join(base_dir, orig_split, 'labelTxt')
        
        new_img_dir = os.path.join(dataset_dir, new_split, 'images')
        new_label_dir = os.path.join(dataset_dir, new_split, 'labels')
        
        # Copy and convert images
        if os.path.exists(orig_img_dir):
            for img_file in os.listdir(orig_img_dir):
                if img_file.endswith(('.jpg', '.jpeg', '.png')):
                    src_path = os.path.join(orig_img_dir, img_file)
                    dst_path = os.path.join(new_img_dir, img_file)
                    shutil.copy(src_path, dst_path)
        
        # Convert and copy labels
        if os.path.exists(orig_label_dir):
            for label_file in os.listdir(orig_label_dir):
                if label_file.endswith('.txt'):
                    src_path = os.path.join(orig_label_dir, label_file)
                    dst_path = os.path.join(new_label_dir, label_file)
                    
                    # Convert rotated bounding box format to YOLO format
                    convert_rotated_bbox_to_yolo(src_path, dst_path)
    
    # Create data.yaml file
    with open(os.path.join(base_dir, 'data.yaml'), 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Update paths
    data_config['path'] = dataset_dir
    data_config['train'] = os.path.join(dataset_dir, 'train', 'images')
    data_config['val'] = os.path.join(dataset_dir, 'val', 'images')
    data_config['test'] = os.path.join(dataset_dir, 'test', 'images')
    
    # Write updated data.yaml
    with open(os.path.join(dataset_dir, 'data.yaml'), 'w') as f:
        yaml.dump(data_config, f)
    
    return os.path.join(dataset_dir, 'data.yaml')

def convert_rotated_bbox_to_yolo(src_path, dst_path):
    """
    Convert rotated bounding box format (x1 y1 x2 y2 x3 y3 x4 y4 class conf) 
    to YOLO format (class_id x_center y_center width height)
    """
    with open(src_path, 'r') as f:
        lines = f.readlines()
    
    yolo_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 9:  # Ensure we have enough parts
            # Extract coordinates and class
            x1, y1, x2, y2, x3, y3, x4, y4 = map(float, parts[:8])
            class_name = parts[8]
            
            # Map class name to class id (0 for awake, 1 for drowsy)
            class_id = 0 if class_name.lower() == 'awake' else 1
            
            # Calculate bounding box center, width, and height
            x_center = (x1 + x2 + x3 + x4) / 4 / 416  # Normalize by image width
            y_center = (y1 + y2 + y3 + y4) / 4 / 416  # Normalize by image height
            
            # Calculate width and height
            width = max(abs(x2 - x1), abs(x3 - x4)) / 416  # Normalize
            height = max(abs(y3 - y1), abs(y4 - y2)) / 416  # Normalize
            
            # Create YOLO format line
            yolo_line = f"{class_id} {x_center} {y_center} {width} {height}\n"
            yolo_lines.append(yolo_line)
    
    # Write to destination file
    with open(dst_path, 'w') as f:
        f.writelines(yolo_lines)

def train_yolo_model(data_yaml_path, epochs=10, batch_size=16):
    """
    Train a YOLOv8 model on the drowsiness detection dataset.
    """
    # Download or use existing YOLOv8 model
    model_path = download_yolo_model()
    if model_path is None:
        print("Error: Could not download or find the base YOLOv8 model.")
        return None
    
    # Load the model
    model = YOLO(model_path)
    
    # Train the model
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=416,
        patience=5,
        save=True
    )
    
    # Return the path to the best model
    return model.export(format='onnx')  # Export to ONNX format for faster inference

if __name__ == "__main__":
    print("Preparing dataset...")
    data_yaml_path = prepare_dataset()
    
    print(f"Training YOLOv8 model using {data_yaml_path}...")
    model_path = train_yolo_model(data_yaml_path)
    
    if model_path:
        print(f"Training complete. Model saved to {model_path}")
    else:
        print("Training failed. Please check the error messages above.") 
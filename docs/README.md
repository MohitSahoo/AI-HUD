# Driver Drowsiness Detection System

This system uses YOLO (You Only Look Once) and OpenCV to detect driver drowsiness in real-time using a webcam. It's designed to identify whether a driver is awake or drowsy based on their facial appearance.

## Features

- Real-time drowsiness detection using YOLO
- Training script to create a custom drowsiness detection model
- Visual feedback with bounding boxes and alerts
- Support for webcam input

## Requirements

- Python 3.7+
- Webcam
- Required Python packages (listed in requirements.txt)

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Download the base YOLOv8 model:
```bash
# This will be downloaded automatically when running the scripts
# but you can also download it manually from:
# https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

## Usage

### Training the Model

The repository includes a pre-annotated dataset for drowsiness detection. To train a custom YOLO model on this dataset:

```bash
python train_yolo_model.py
```

This will:
1. Prepare the dataset in the correct format for YOLOv8
2. Train a YOLOv8 model on the drowsiness detection dataset
3. Save the trained model to `runs/detect/train/weights/best.pt`

Training may take some time depending on your hardware. A GPU is recommended for faster training.

### Running Drowsiness Detection

After training (or even without training, using the base model), you can run the drowsiness detection:

```bash
python drowsiness_detection.py
```

The system will:
- Open your webcam
- Detect whether you appear awake or drowsy
- Display bounding boxes (green for awake, red for drowsy)
- Alert you if drowsiness is detected for a sustained period

Press 'q' to quit the application.

## How it Works

The system uses the following approach:
1. YOLOv8 model trained to detect "awake" and "drowsy" states
2. Real-time classification of the driver's state
3. Drowsiness alert based on sustained detection of the "drowsy" state

## Dataset

The dataset includes images of a person in a vehicle simulating "drowsy" and "awake" facial postures. It was originally provided by Augmented Startups through Roboflow.

## Customization

You can adjust the following parameters in the code:
- `DROWSY_CONSEC_FRAMES`: Number of consecutive frames of drowsiness to trigger alert (default: 20)
- Detection confidence threshold in `model(frame, conf=0.25)`

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
# 🚨 Real-time Drowsiness Detection System

A sophisticated computer vision system that detects drowsiness in real-time using your webcam. The system uses advanced machine learning techniques to monitor facial features and movement patterns to accurately detect when someone is becoming drowsy.

![Drowsiness Detection Demo](docs/demo.gif)

## ✨ Features

- 🔍 Real-time drowsiness detection using webcam
- 🎯 Accurate detection of multiple drowsiness indicators:
  - Head nodding
  - Head tilting
  - Eye closure
  - Reduced movement
  - Face position tracking
- 🎨 User-friendly Streamlit interface
- 🔊 Audio alerts when drowsiness is detected
- 🚗 traffic detection integration
- 🎯 High accuracy with minimal false positives
- ⚡ Fast and efficient processing

## 🛠️ Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/drowsiness-detection.git
cd drowsiness-detection
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Quick Start

Run the integrated system with both traffic and drowsiness detection:

```bash
streamlit run src/integrated_app.py
```

### Drowsiness Detection Only

Run just the drowsiness detection system:

```bash
python src/drowsiness_detection.py
```

## 📁 Project Structure

```
drowsiness-detection/
├── data/                  # Dataset and training data
├── docs/                  # Documentation and demos
├── models/               # Model weights and configurations
├── src/                  # Source code
│   ├── integrated_app.py           # Main Streamlit application
│   ├── drowsiness_detection.py     # Core detection logic
│   ├── train_yolo_model.py        # Model training script
│   └── download_model.py          # Model download utility
├── utils/               # Utility files and resources
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 🎯 How It Works

The system uses a combination of computer vision and machine learning techniques:

1. **Face Detection**: Uses YOLOv8 to detect faces in real-time
2. **Feature Analysis**: Tracks multiple facial features and movements
3. **Pattern Recognition**: Identifies drowsiness patterns through:
   - Head movement analysis
   - Face position tracking
   - Movement pattern detection
   - Stability monitoring
4. **Alert System**: Triggers visual and audio alerts when drowsiness is detected

## 📋 Requirements

- Python 3.8 or higher
- Webcam
- Dependencies (automatically installed with requirements.txt):
  - OpenCV
  - PyTorch
  - Ultralytics YOLOv8
  - Streamlit
  - Pygame
  - NumPy
  - SciPy

## ⚙️ Configuration

The system can be configured through various parameters in `src/integrated_app.py`:

- `DROWSY_CONSEC_FRAMES`: Number of consecutive frames needed to trigger alert
- `confidence_threshold`: Minimum confidence for detection
- `movement_threshold`: Sensitivity to movement
- `stability_threshold`: Required stability for detection

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics
- OpenCV community
- Streamlit team
- All contributors and users of this project


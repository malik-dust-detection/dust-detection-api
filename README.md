# 🏠 Dust Detection System

Real-time dust detection using ESP32-CAM and Flask API with TensorFlow.

## Features
- ESP32-CAM captures images
- Flask API processes images
- TensorFlow model classifies as clean/dusty
- Returns confidence score

## Files
- `app.py` - Flask application
- `model.h5` - Trained model
- `labels.txt` - Class labels (0=clean, 1=dusty)

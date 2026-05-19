# Real-Time Traffic Anomaly Detection

A deep learning system that detects traffic accidents in real time from video using a custom Transformer model trained on video clip sequences.

## How It Works

Video frames are buffered into 16-frame clips → passed through `AccidentTransformer` → classified with a rolling confidence window into one of three states:

| Output | Condition |
|---|---|
| ✅ Normal | Low accident count in prediction buffer |
| ⚠️ Collision Risk | Moderate anomaly signals detected |
| 🚨 Accident Detected | High confidence + sustained anomaly count |

## Project Structure

```
├── train.py              # Model training script
├── detect.py             # Real-time detection on video
├── extract_clips.py      # Clip extraction from raw footage
├── model.pth             # Pretrained model weights
├── src/
│   ├── model.py          # AccidentTransformer architecture
│   └── dataset.py        # VideoDataset loader
├── dataset/              # Training data
├── dataset_clips/        # Extracted train/val clips
└── *.mp4                 # Sample accident & traffic videos
```

## Setup & Run

```bash
pip install torch torchvision opencv-python numpy

# Train the model
python train.py

# Run detection on a video
python detect.py
```

To change the input video, edit this line in `detect.py`:
```python
cap = cv2.VideoCapture("your_video.mp4")  # or 0 for webcam
```

## Model Details

- Architecture: Custom Transformer (`AccidentTransformer`) for spatiotemporal video classification
- Input: 16-frame clips at 224×224 resolution
- Classes: Accident / Normal
- Training: 5 epochs, Adam optimizer (lr=1e-4), CrossEntropyLoss
- Confidence threshold: 0.75 (detections below this are treated as Normal)

## Requirements

- Python 3.8+
- PyTorch
- OpenCV
- NumPy

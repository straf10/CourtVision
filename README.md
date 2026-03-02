# Basketball Video Analysis

Analyze basketball footage with automated detection of players, ball, team assignment, and more. This repository integrates object tracking, zero-shot classification, and custom keypoint detection for a fully annotated basketball game experience.

Leveraging the convenience of [Roboflow](https://roboflow.com/) for dataset management and [Ultralytics](https://docs.ultralytics.com/) YOLO models for both training and inference, this project provides a robust framework for basketball video analysis.

Training notebooks are included to help you customize and fine-tune models to suit your specific needs, ensuring a seamless and efficient workflow.

## Features

- **Player and ball detection/tracking** using pretrained YOLOv12 models with ByteTrack.
- **Court keypoint detection** for visualizing important zones.
- **Team assignment** with jersey color classification.
- **Ball possession detection**, pass detection, and interception detection.
- **Easy stubbing** to skip repeated computation for fast iteration.
- **Various drawers** to overlay detected elements (ellipses, triangles, annotations) onto frames.

## Project Structure

```
basketball_analysis/
├── main.py                        # Entry point — runs the full pipeline
├── trackers/
│   ├── player_tracker.py          # Player detection & tracking (YOLO + ByteTrack)
│   └── ball_tracker.py            # Ball detection, tracking, filtering & interpolation
├── drawers/
│   ├── player_tracks_drawer.py    # Draws player ellipses and track IDs
│   ├── ball_tracks_drawer.py      # Draws ball position as triangles
│   └── utils.py                   # Shared drawing primitives
├── utils/
│   ├── video_utils.py             # read_video(), save_video()
│   ├── stubs_utils.py             # save_stub(), read_stub() for pickle caching
│   └── bbox_utils.py              # Bounding box helpers
├── training_notebooks/
│   ├── ball_detection_training.py               # YOLOv12-L training script
│   └── basketball_player_detection_training.ipynb  # YOLOv12-S Colab notebook
├── models/                        # YOLO .pt weights (gitignored)
├── stubs/                         # Cached tracks for fast re-runs
├── data/                          # Training outputs and metrics
├── input_videos/                  # Source footage (gitignored)
└── output_videos/                 # Annotated output (gitignored)
```

## Getting Started

### Prerequisites

- Python 3.10+
- A CUDA-capable GPU is recommended for inference and training

### Installation

```bash
git clone https://github.com/<your-username>/basketball_analysis.git
cd basketball_analysis
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

### Model Weights

Download or train your own YOLO weights and place them in the `models/` directory. The pipeline expects:

| File | Used by |
|------|---------|
| `models/best.pt` | `BallTracker` — YOLOv12-L trained on [Basketball-Players](https://universe.roboflow.com/roboflow-universe-projects/basketball-players-fy4c2) (Roboflow) |
| `models/player_detector_model.pt` | `PlayerTracker` — YOLOv12 player detection weights |

### Running the Pipeline

1. Place your input video at `input_videos/video_1.mp4`.
2. Run the pipeline:

```bash
python main.py
```

3. The annotated output is saved to `output_videos/output_video.avi`.

> **Tip:** On the first run, stub files are generated in `stubs/`. Subsequent runs load from stubs instantly. Delete the stub files when you want to re-run detection with a new model or video.

## Training

Training notebooks are provided in `training_notebooks/`:

| Notebook | Model | Environment |
|----------|-------|-------------|
| `ball_detection_training.py` | YOLOv12-L | Local GPU (RTX 3090) |
| `basketball_player_detection_training.ipynb` | YOLOv12-S | Google Colab |

Both use the **Basketball-Players-25** dataset from Roboflow. To download the dataset, create a `.env` file with your API key:

```
ROBOFLOW_API_KEY=your_key_here
```

### Latest Training Results (YOLOv12-L, imgsz=1280)

| Class | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| **All** | **0.909** | **0.897** | **0.895** | **0.664** |
| Ball | 1.000 | 0.534 | 0.752 | 0.506 |
| Hoop | 0.842 | 0.955 | 0.945 | 0.724 |
| Player | 0.969 | 0.870 | 0.941 | 0.666 |
| Ref | 0.933 | 0.909 | 0.922 | 0.705 |
| Shot Clock | 0.905 | 1.000 | 0.900 | 0.624 |
| Team Name | 0.893 | 0.900 | 0.880 | 0.709 |
| Team Points | 0.877 | 0.949 | 0.886 | 0.683 |
| Time Remaining | 0.860 | 1.000 | 0.920 | 0.700 |

## Dependencies

- [ultralytics](https://github.com/ultralytics/ultralytics) — YOLO model training and inference
- [supervision](https://github.com/roboflow/supervision) — ByteTrack object tracking
- [opencv-python](https://github.com/opencv/opencv-python) — Video I/O and frame manipulation
- [numpy](https://numpy.org/) / [pandas](https://pandas.pydata.org/) — Data processing and interpolation
- [roboflow](https://github.com/roboflow/roboflow-python) — Dataset management (training only)

## License

This project is for educational and research purposes.

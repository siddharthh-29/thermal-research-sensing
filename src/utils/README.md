# src/utils — YOLOv5-Face utilities

The files in this directory are adapted from the
[yolov5-face](https://github.com/IS2AI/AnyFace/tree/main/yolov5-face/utils)
utility module, included here to support development and testing of the face
landmark detection step (`scripts/mmslab_sim1_05_detect_face_landmarks.py`).

**Original source:** https://github.com/IS2AI/AnyFace/tree/main/yolov5-face/utils

| File | Purpose |
|------|---------|
| `datasets.py` | Image/video dataloaders and letterbox preprocessing |
| `general.py` | NMS, coordinate scaling, and general inference utilities |
| `torch_utils.py` | Device selection and model loading helpers |
| `metrics.py` | Detection evaluation metrics (mAP, IoU) |
| `plots.py` | Bounding box and landmark drawing utilities |
| `autoanchor.py` | Anchor generation for YOLOv5 training |
| `google_utils.py` | Model download helpers |

These utilities are only needed for running the face detector in step 05.
For the rest of the pipeline (ROI extraction, signal processing) they are
not used.

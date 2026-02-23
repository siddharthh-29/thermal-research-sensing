# Model Weights

This directory is a placeholder for the YOLOv5-Face model weights used by
`scripts/mmslab_sim1_05_detect_face_landmarks.py`.

## Required file

| File | Description |
|------|-------------|
| `yolov5s_face.pt` | YOLOv5-small face detector trained on thermal faces (TFW) |

## How to download

The weights are provided by the **Thermal Faces in the Wild (TFW)** project:

> https://github.com/IS2AI/TFW?tab=readme-ov-file

Follow the instructions in the TFW repository to download the pretrained
weights for thermal face detection. Place the file `yolov5s_face.pt` in
this directory.

## About the model

The TFW weights are trained on the
[SpeakingFaces corpus](https://github.com/IS2AI/SpeakingFaces) and fine-tuned
for LWIR thermal imagery. They are used together with the
[yolov5-face](https://github.com/deepcam-cn/yolov5-face) inference code
(located in `src/utils/`) to detect a face bounding box and 5 landmarks
(left eye, right eye, nose tip, left mouth corner, right mouth corner) in
each thermal frame.

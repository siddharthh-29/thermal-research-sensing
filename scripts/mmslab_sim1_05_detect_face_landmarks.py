"""
mmslab_sim1_05_detect_face_landmarks.py

Batch face detection on raw thermal .dat recordings using YOLOv5-Face with
TFW (Thermal Faces in the Wild) weights.

For each subject-task pair, exports:
  - CSV  <SUBJECT>-<TASK>-face.csv  with bounding box, 5 landmarks, and
    confidence score per frame.
  - (optional) annotated JPEG thumbnails for visual quality-check.

Pipeline step 5: Generate face annotation CSVs used by the ROI extraction
and signal synchronization stages.

Requirements:
  This script imports from the yolov5-face repository. Ensure the repo root
  is on sys.path (the script adds its parent directory automatically).
  Weights file: models/yolov5s_face.pt  (TFW thermal weights).

Configuration:
  Edit the CONFIGURATION block below before running.
"""

import os
import re
import csv
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

# Ensure the yolov5-face repo root (parent of scripts/) is importable.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, scale_coords
from utils.torch_utils import select_device

# =============================================================================
# Configuration
# =============================================================================
BASE_PATH = Path("/media/arritmic/HI-NRI-TINO-002/DATABASES/MULTIMODAL/Affective_States/SIMULATOR_STUDY_I")
RAW_DIR = BASE_PATH / "RawThermalData"

WEIGHTS = Path("../models/yolov5s_face.pt")  # relative to yolov5-face repo root

FPS = 7.5
WIDTH = 640
HEIGHT = 512

IMG_SIZE = 800
CONF_THRES = 0.25
IOU_THRES = 0.5

# Set to None to auto-discover all T### directories under RawThermalData.
SUBJECTS = ["T016", "T017", "T018", "T023", "T034", "T036", "T038", "T039"]
TASKS    = ["PD", "ND", "CD", "ED"]

# SIM1 task segment durations (seconds).
# PD is the 3-min practice drive; ND/CD/ED cover the first 4 load phases.
CALM_DURATION_S      = 180.0
LD_FIRST4_DURATION_S = 640.0   # P1(80s) + P2(160s) + P3(240s) + P4(160s)

TASK_SEGMENTS = {
    "PD": (0.0, CALM_DURATION_S),
    "ND": (0.0, LD_FIRST4_DURATION_S),
    "CD": (0.0, LD_FIRST4_DURATION_S),
    "ED": (0.0, LD_FIRST4_DURATION_S),
}

SAVE_DEBUG_IMAGES   = True   # save annotated JPEG thumbnails for visual QC
SAVE_EVERY_N_FRAMES = 100    # thumbnail interval (frames)

# CSVs and debug images are written alongside the .dat files in RawThermalData.
OUT_ROOT = Path("/media/arritmic/HI-NRI-TINO-002/DATABASES/MULTIMODAL/Affective_States/SIMULATOR_STUDY_I/RawThermalData")


# =============================================================================
# .inf timestamp parsing
# =============================================================================
_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def read_inf_timestamps(inf_path: Path) -> Tuple[Optional[int], Optional[int], Optional[int], np.ndarray]:
    """Parse a .inf header and return (n_frames, width, height, timestamps)."""
    try:
        text = inf_path.read_text(errors="ignore")
    except Exception:
        b = inf_path.read_bytes()
        text = b.decode("latin-1", errors="ignore")

    nums = _NUM_RE.findall(text)
    if len(nums) < 3:
        return None, None, None, np.array([], dtype=np.float64)

    n_frames = int(float(nums[0]))
    w = int(float(nums[1]))
    h = int(float(nums[2]))

    raw = np.array([float(x) for x in nums[3:]], dtype=np.float64)
    if raw.size == 0:
        return n_frames, w, h, np.array([], dtype=np.float64)

    # many .inf files store one timestamp per frame
    if raw.size >= n_frames:
        ts = raw[:n_frames]
    else:
        ts = raw

    return n_frames, w, h, ts


def normalize_ts_seconds(ts: np.ndarray, t0: float, t1: float) -> np.ndarray:
    """
    Heuristic: try identity, ms->s, min->s, and shifted variants.
    Pick the transform that maximizes overlap with [t0, t1].
    """
    ts = np.asarray(ts, dtype=np.float64)
    if ts.size == 0 or not np.isfinite(ts).any():
        return ts

    finite = np.isfinite(ts)
    ts_fin = ts[finite]
    ts_min = float(np.nanmin(ts_fin))

    def score(x: np.ndarray) -> int:
        m = np.isfinite(x)
        if not np.any(m):
            return 0
        return int(np.sum((x[m] >= t0) & (x[m] <= t1)))

    candidates = []
    candidates.append(ts)
    candidates.append(ts / 1000.0)
    candidates.append(ts * 60.0)
    candidates.append(ts - ts_min)
    candidates.append((ts / 1000.0) - (ts_min / 1000.0))
    candidates.append((ts * 60.0) - (ts_min * 60.0))

    best = candidates[0]
    best_sc = score(best)
    for c in candidates[1:]:
        sc = score(c)
        if sc > best_sc:
            best_sc = sc
            best = c
    return best


# =============================================================================
# Raw thermal reader
# =============================================================================
def infer_dtype_and_nframes(dat_path: Path, width: int, height: int) -> tuple[np.dtype, int]:
    size_bytes = dat_path.stat().st_size
    pixels_per_frame = width * height

    for dtype in (np.dtype("<u2"), np.dtype("<u4")):
        denom = pixels_per_frame * dtype.itemsize
        n = size_bytes / denom
        if abs(n - round(n)) < 1e-9 and n >= 1:
            return dtype, int(round(n))

    raise ValueError(f"Could not infer dtype/n_frames from {dat_path} (size={size_bytes} bytes)")


def thermal_frame_to_bgr_u8(frame: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    """Normalize raw thermal frame and apply inferno colormap → BGR uint8."""
    x = frame.astype(np.float32)
    lo = np.percentile(x, p_low)
    hi = np.percentile(x, p_high)
    if hi <= lo:
        u8 = np.zeros_like(x, dtype=np.uint8)
    else:
        x = (x - lo) / (hi - lo)
        x = np.clip(x, 0.0, 1.0)
        u8 = (x * 255.0).astype(np.uint8)

    return cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)


# =============================================================================
# Landmarks scaling and drawing
# =============================================================================
def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6, 8]] -= pad[0]
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]
    coords[:, :10] /= gain

    coords[:, 0].clamp_(0, img0_shape[1])
    coords[:, 1].clamp_(0, img0_shape[0])
    coords[:, 2].clamp_(0, img0_shape[1])
    coords[:, 3].clamp_(0, img0_shape[0])
    coords[:, 4].clamp_(0, img0_shape[1])
    coords[:, 5].clamp_(0, img0_shape[0])
    coords[:, 6].clamp_(0, img0_shape[1])
    coords[:, 7].clamp_(0, img0_shape[0])
    coords[:, 8].clamp_(0, img0_shape[1])
    coords[:, 9].clamp_(0, img0_shape[0])
    return coords


def draw_one_face(im0: np.ndarray, x1, y1, x2, y2, conf: float, lms: list[float]) -> np.ndarray:
    out = im0.copy()
    tl = max(1, int(round(0.002 * (out.shape[0] + out.shape[1]) / 2)) + 1)

    cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), thickness=tl, lineType=cv2.LINE_AA)

    for i in range(5):
        px = int(lms[2 * i])
        py = int(lms[2 * i + 1])
        cv2.circle(out, (px, py), tl + 1, (255, 0, 0), -1)

    cv2.putText(
        out,
        f"{conf:.3f}",
        (int(x1), max(0, int(y1) - 4)),
        0,
        tl / 3,
        (225, 255, 255),
        thickness=max(tl - 1, 1),
        lineType=cv2.LINE_AA,
    )
    return out


# =============================================================================
# Per-session export
# =============================================================================
@torch.no_grad()
def export_subject_task(model, device, subject: str, task: str):
    subject = subject.upper()
    task = task.upper()

    if task not in TASK_SEGMENTS:
        print(f"[WARN] task {task} not in TASK_SEGMENTS, skipping")
        return

    t0, dur = TASK_SEGMENTS[task]
    t1 = t0 + dur

    subj_dir = RAW_DIR / subject
    dat_path = subj_dir / f"{subject}-{task}.dat"
    inf_path = subj_dir / f"{subject}-{task}.inf"

    if not dat_path.exists():
        print(f"[WARN] missing dat: {dat_path}")
        return

    dtype, n_frames = infer_dtype_and_nframes(dat_path, WIDTH, HEIGHT)
    frames = np.memmap(dat_path, dtype=dtype, mode="r", shape=(n_frames, HEIGHT, WIDTH))

    # Optional timestamps from .inf (if present)
    ts_inf = np.array([], dtype=np.float64)
    if inf_path.exists():
        _, _, _, ts_raw = read_inf_timestamps(inf_path)
        if ts_raw.size > 0:
            ts_inf = normalize_ts_seconds(ts_raw, t0=float(t0), t1=float(t1))

    # Frame range
    i0 = int(np.floor(t0 * FPS))
    i1 = int(np.ceil(t1 * FPS))
    i0 = max(0, min(i0, n_frames - 1))
    i1 = max(0, min(i1, n_frames - 1))

    imgsz = check_img_size(IMG_SIZE, s=int(model.stride.max()))

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    csv_path = OUT_ROOT / f"{subject}/{subject}-{task}-face.csv"
    img_dir = OUT_ROOT / f"{subject}/{subject}-{task}-faces"
    if SAVE_DEBUG_IMAGES:
        img_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] {subject} {task} | dtype={dtype} | frames={n_frames} | segment [{t0:.1f}s, {t1:.1f}s] -> [{i0}, {i1}]")

    # When no face is detected, bbox/landmark/conf columns are left empty (NaN).
    header = [
        "subject", "task",
        "frame_idx",
        "time_s_nominal",
        "time_s_inf",
        "n_faces",
        "conf",
        "x1", "y1", "x2", "y2",
        "lm1_x", "lm1_y",
        "lm2_x", "lm2_y",
        "lm3_x", "lm3_y",
        "lm4_x", "lm4_y",
        "lm5_x", "lm5_y",
    ]

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for k, fi in enumerate(range(i0, i1 + 1)):
            t_nom = fi / FPS
            t_inf = float(ts_inf[fi]) if (ts_inf.size > fi and np.isfinite(ts_inf[fi])) else np.nan

            fr = frames[fi]
            im0 = thermal_frame_to_bgr_u8(fr)  # BGR uint8 (inferno)

            img = letterbox(im0, new_shape=imgsz)[0]          # BGR letterboxed
            img = img[:, :, ::-1].transpose(2, 0, 1)          # BGR→RGB, HWC→CHW
            img = np.ascontiguousarray(img)

            img_t = torch.from_numpy(img).to(device).float() / 255.0
            if img_t.ndimension() == 3:
                img_t = img_t.unsqueeze(0)

            pred = model(img_t)[0]
            pred = non_max_suppression_face(pred, CONF_THRES, IOU_THRES)

            det = pred[0]
            n_faces = 0

            conf = np.nan
            x1 = y1 = x2 = y2 = np.nan
            lms = [np.nan] * 10

            # Select best face (highest confidence)
            if det is not None and len(det):
                if det.shape[1] < 16:
                    raise RuntimeError("Weights do not output landmarks. Use a yolov5-face landmark weights file.")

                n_faces = int(det.shape[0])

                det[:, :4]   = scale_coords(img_t.shape[2:], det[:, :4], im0.shape).round()
                det[:, 5:15] = scale_coords_landmarks(img_t.shape[2:], det[:, 5:15], im0.shape).round()

                best_idx = int(torch.argmax(det[:, 4]).item())
                best = det[best_idx]

                x1, y1, x2, y2 = [float(v.item()) for v in best[:4]]
                conf = float(best[4].item())
                lms  = [float(v.item()) for v in best[5:15]]

                if SAVE_DEBUG_IMAGES and ((fi - i0) % SAVE_EVERY_N_FRAMES == 0):
                    vis = draw_one_face(im0, x1, y1, x2, y2, conf, lms)
                    cv2.putText(
                        vis,
                        f"{subject}-{task}  t={t_nom:7.3f}s  frame={fi}",
                        (10, 22),
                        0,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
                    out_name = img_dir / f"frame_{fi:06d}.jpg"
                    cv2.imwrite(str(out_name), vis)

            # Write row
            row = [
                subject, task,
                fi,
                f"{t_nom:.6f}",
                f"{t_inf:.6f}" if np.isfinite(t_inf) else "",
                n_faces,
                f"{conf:.6f}" if np.isfinite(conf) else "",
                f"{x1:.3f}" if np.isfinite(x1) else "",
                f"{y1:.3f}" if np.isfinite(y1) else "",
                f"{x2:.3f}" if np.isfinite(x2) else "",
                f"{y2:.3f}" if np.isfinite(y2) else "",
            ]

            for v in lms:
                row.append(f"{v:.3f}" if np.isfinite(v) else "")

            writer.writerow(row)

            if (k + 1) % 500 == 0:
                print(f"  processed {k + 1} / {i1 - i0 + 1} frames")

    print(f"[OK] wrote CSV: {csv_path}")
    if SAVE_DEBUG_IMAGES:
        print(f"[OK] debug images: {img_dir} (every {SAVE_EVERY_N_FRAMES} frames)")


@torch.no_grad()
def main():
    # Subject discovery if SUBJECTS is None
    if SUBJECTS is None:
        pat = re.compile(r"^T\d{3}$")
        subjects = sorted([p.name for p in RAW_DIR.iterdir() if p.is_dir() and pat.match(p.name)])
    else:
        subjects = [s.upper() for s in SUBJECTS]

    # Load model once
    device = select_device("")  # auto; set "cuda:0" if you want to force
    model = attempt_load(str(WEIGHTS), map_location=device)
    model.eval()

    print(f"[INFO] weights: {WEIGHTS}")
    print(f"[INFO] subjects: {len(subjects)}")
    print(f"[INFO] tasks: {TASKS}")
    print(f"[INFO] outputs: {OUT_ROOT.resolve()}")

    for subject in subjects:
        for task in TASKS:
            try:
                export_subject_task(model, device, subject, task)
            except KeyboardInterrupt:
                print("\n[STOP] interrupted by user")
                return
            except Exception as e:
                print(f"[WARN] failed {subject} {task}: {e}")

    print("[DONE] all exports finished")


if __name__ == "__main__":
    main()

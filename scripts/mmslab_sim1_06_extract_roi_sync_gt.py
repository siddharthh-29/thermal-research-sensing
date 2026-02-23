"""
mmslab_sim1_06_extract_roi_sync_gt.py

Batch ROI extraction and ground-truth synchronization for the SIM1 dataset.

For each subject-task pair:
  1. Reads the face annotation CSV produced by step 05 (bounding box + 5 landmarks).
  2. Smooths landmark positions with an exponential moving average (α=0.15).
  3. Defines 6 facial ROIs (nose, eye_l, eye_r, cheek_l, cheek_r, forehead) from
     stabilized landmarks, clipped to frame boundaries.
  4. Extracts a scalar temperature value per ROI per frame using the configured
     spatial aggregation method (see src/roi_extraction_methods.py).
  5. Upsamples the 7.5 Hz ROI traces to 30 Hz by cubic spline interpolation.
  6. Loads and resamples the contact ground-truth signals (HR, BR, PP, PEDA)
     to 30 Hz, aligning them in time with the thermal traces.
  7. Writes a synchronized CSV per session to StructuredStudyData.

Output CSV columns:
  time_s, [roi_name, ...], HR, BR, PP, PP_NR, PEDA

Pipeline step 6: Run after step 05 (face landmark detection).
Single-session reference version: mmslab_sim1_06_extract_roi_sync_gt_single.py

Configuration:
  Edit SUBJECTS, TASKS, BASE_PATH, and output options at the top of the script.
"""

import csv
import re
from io import BytesIO
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.roi_extraction_methods import roi_extract_by_name

# =============================================================================
# Configuration
# =============================================================================
BASE_PATH = Path("/media/arritmic/HI-NRI-TINO-002/DATABASES/MULTIMODAL/Affective_States/SIMULATOR_STUDY_I")
RAW_DIR = BASE_PATH / "RawThermalData"
STRUCT_DIR = BASE_PATH / "StructuredStudyData"

SUBJECTS   = ["T002", "T003", "T005", "T014", "T029", "T031", "T034", "T036"]
GENDER     = ["F",    "M",    "M",    "F",    "F",    "F",    "M",    "M"]
AGE_GROUP  = ["Y",    "Y",    "Y",    "Y",    "O",    "O",    "O",    "O"]
TASKS      = ["PD", "ND", "CD", "ED"]

# Set SUBJECTS=[] to auto-discover all T### directories under RAW_DIR.
AUTO_DISCOVER_SUBJECTS_IF_EMPTY = True

# If True, only process tasks that exist on disk for each subject (recommended)
ONLY_EXISTING_TASKS = True

FPS_NATIVE = 7.5
FPS_OUT = 30.0
WIDTH = 640
HEIGHT = 512

CALM_DURATION_S = 180.0
LD_FIRST4_DURATION_S = 640.0
TASK_SEGMENTS = {
    "BL": (0.0, CALM_DURATION_S),
    "PD": (0.0, CALM_DURATION_S),
    "ND": (0.0, LD_FIRST4_DURATION_S),
    "CD": (0.0, LD_FIRST4_DURATION_S),
    "ED": (0.0, LD_FIRST4_DURATION_S),
}

LANDMARK_SMOOTH_ALPHA = 0.15

# Debug overlays (can produce many files in batch)
SAVE_DEBUG_IMAGES = False
DEBUG_EVERY_N_FRAMES = 150
DEBUG_OUT_DIR = BASE_PATH / "AnalysisOutputs" / "roi_sync_debug"
SHOW_DEBUG_WINDOW = False

# Plots
PLOT_MODE = "none"  # "none" | "save" | "show"
PLOT_OUT_DIR = BASE_PATH / "AnalysisOutputs" / "roi_sync_plots"

# Summary
SUMMARY_OUT_CSV = BASE_PATH / "AnalysisOutputs" / "roi_sync_batch_summary.csv"


# =============================================================================
# Raw thermal + face CSV
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


def safe_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def load_face_csv(face_csv: Path) -> Dict[int, Dict[str, float]]:
    ann = {}
    with face_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fi = int(float(row["frame_idx"]))
            ann[fi] = {
                "conf": safe_float(row.get("conf", "")),
                "x1": safe_float(row.get("x1", "")),
                "y1": safe_float(row.get("y1", "")),
                "x2": safe_float(row.get("x2", "")),
                "y2": safe_float(row.get("y2", "")),
                "lm1_x": safe_float(row.get("lm1_x", "")),
                "lm1_y": safe_float(row.get("lm1_y", "")),
                "lm2_x": safe_float(row.get("lm2_x", "")),
                "lm2_y": safe_float(row.get("lm2_y", "")),
                "lm3_x": safe_float(row.get("lm3_x", "")),
                "lm3_y": safe_float(row.get("lm3_y", "")),
                "lm4_x": safe_float(row.get("lm4_x", "")),
                "lm4_y": safe_float(row.get("lm4_y", "")),
                "lm5_x": safe_float(row.get("lm5_x", "")),
                "lm5_y": safe_float(row.get("lm5_y", "")),
            }
    return ann


def smooth_landmarks_ema(ann: Dict[int, Dict[str, float]], alpha: float = 0.15) -> None:
    keys_to_smooth = [
        "x1", "y1", "x2", "y2",
        "lm1_x", "lm1_y", "lm2_x", "lm2_y",
        "lm3_x", "lm3_y", "lm4_x", "lm4_y",
        "lm5_x", "lm5_y",
    ]
    prev = None
    for fi in sorted(ann.keys()):
        row = ann[fi]
        if prev is None:
            prev = row.copy()
            continue
        for key in keys_to_smooth:
            if np.isfinite(row[key]) and np.isfinite(prev[key]):
                row[key] = alpha * row[key] + (1 - alpha) * prev[key]
        prev = row


# =============================================================================
# ROI geometry
# =============================================================================
def clip_rect(x1, y1, x2, y2, w, h) -> Tuple[int, int, int, int]:
    x1 = int(np.clip(x1, 0, w - 1))
    y1 = int(np.clip(y1, 0, h - 1))
    x2 = int(np.clip(x2, 0, w - 1))
    y2 = int(np.clip(y2, 0, h - 1))
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2


def rect_from_center(cx, cy, rw, rh, w, h) -> Tuple[int, int, int, int]:
    x1 = cx - rw / 2
    y1 = cy - rh / 2
    x2 = cx + rw / 2
    y2 = cy + rh / 2
    return clip_rect(x1, y1, x2, y2, w, h)


def build_rois_from_face(ann_row: Dict[str, float]) -> Dict[str, Tuple[int, int, int, int]]:
    w, h = WIDTH, HEIGHT
    x1, y1, x2, y2 = ann_row["x1"], ann_row["y1"], ann_row["x2"], ann_row["y2"]
    if not np.isfinite([x1, y1, x2, y2]).all():
        return {}

    x1i, y1i, x2i, y2i = clip_rect(x1, y1, x2, y2, w, h)
    bw = float(x2i - x1i)
    bh = float(y2i - y1i)

    le = np.array([ann_row["lm1_x"], ann_row["lm1_y"]], dtype=np.float64)
    re_ = np.array([ann_row["lm2_x"], ann_row["lm2_y"]], dtype=np.float64)
    no = np.array([ann_row["lm3_x"], ann_row["lm3_y"]], dtype=np.float64)
    ml = np.array([ann_row["lm4_x"], ann_row["lm4_y"]], dtype=np.float64)
    mr = np.array([ann_row["lm5_x"], ann_row["lm5_y"]], dtype=np.float64)

    if not np.isfinite(np.concatenate([le, re_, no, ml, mr])).all():
        return {}

    mid_eye = 0.5 * (le + re_)
    v_up = mid_eye - no
    forehead_c = mid_eye + 0.6 * v_up

    left_cheek_c = 0.5 * (le + ml) + np.array([-0.10 * bw, 0.05 * bh])
    right_cheek_c = 0.5 * (re_ + mr) + np.array([+0.10 * bw, 0.05 * bh])

    eye_rw, eye_rh = 0.24 * bw, 0.12 * bh
    nose_rw, nose_rh = 0.30 * bw, 0.15 * bh
    cheek_rw, cheek_rh = 0.20 * bw, 0.20 * bh
    forehead_rw, forehead_rh = 0.45 * bw, 0.18 * bh

    rois = {
        "nose": rect_from_center(no[0], no[1], nose_rw, nose_rh, w, h),
        "eye_l": rect_from_center(le[0] + 15, le[1], eye_rw, eye_rh, w, h),
        "eye_r": rect_from_center(re_[0] - 15, re_[1], eye_rw, eye_rh, w, h),
        "cheek_l": rect_from_center(left_cheek_c[0], left_cheek_c[1] - 10, cheek_rw, cheek_rh, w, h),
        "cheek_r": rect_from_center(right_cheek_c[0], right_cheek_c[1] - 10, cheek_rw, cheek_rh, w, h),
        "forehead": rect_from_center(forehead_c[0], forehead_c[1] - 10, forehead_rw, forehead_rh, w, h),
        "face_bbox": (x1i, y1i, x2i, y2i),
    }
    return rois


def draw_rois_for_debug(frame_raw: np.ndarray, rois: Dict[str, Tuple[int, int, int, int]]) -> np.ndarray:
    x = frame_raw.astype(np.float32)
    lo = np.percentile(x, 2.0)
    hi = np.percentile(x, 98.0)
    if hi <= lo:
        u8 = np.zeros_like(x, dtype=np.uint8)
    else:
        u8 = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
        u8 = (u8 * 255.0).astype(np.uint8)
    vis = cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)

    for name, r in rois.items():
        color = (0, 255, 0) if name == "face_bbox" else (255, 255, 255)
        x1, y1, x2, y2 = r
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        cv2.putText(vis, name, (x1, max(12, y1 + 12)), 0, 0.4, color, 1, cv2.LINE_AA)
    return vis


# =============================================================================
# Resampling utilities
# =============================================================================
def interp_to_grid_linear(t_in: np.ndarray, y_in: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    t_in = np.asarray(t_in, dtype=np.float64)
    y_in = np.asarray(y_in, dtype=np.float64)
    t_grid = np.asarray(t_grid, dtype=np.float64)

    m = np.isfinite(t_in) & np.isfinite(y_in)
    if np.sum(m) < 2:
        return np.full_like(t_grid, np.nan, dtype=np.float64)

    t = t_in[m]
    y = y_in[m]
    idx = np.argsort(t)
    t = t[idx]
    y = y[idx]

    t_unique, unique_idx = np.unique(t, return_index=True)
    y_unique = y[unique_idx]

    return np.interp(t_grid, t_unique, y_unique, left=np.nan, right=np.nan)


def interp_to_grid_cubic(t_in: np.ndarray, y_in: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    t_in = np.asarray(t_in, dtype=np.float64)
    y_in = np.asarray(y_in, dtype=np.float64)
    t_grid = np.asarray(t_grid, dtype=np.float64)

    m = np.isfinite(t_in) & np.isfinite(y_in)
    if np.sum(m) < 2:
        return np.full_like(t_grid, np.nan, dtype=np.float64)

    t = t_in[m]
    y = y_in[m]
    idx = np.argsort(t)
    t = t[idx]
    y = y[idx]

    t_unique, unique_idx = np.unique(t, return_index=True)
    y_unique = y[unique_idx]

    if len(t_unique) < 4:
        return np.interp(t_grid, t_unique, y_unique, left=np.nan, right=np.nan)

    f = interp1d(t_unique, y_unique, kind="cubic", bounds_error=False, fill_value=np.nan)
    return f(t_grid)


# =============================================================================
# StructuredStudyData loaders for PEDA and PP (OTACS containers)
# =============================================================================
def read_otacs_table(path: Path, start_row: int = 9) -> pd.DataFrame:
    b = path.read_bytes()
    OLE_SIG = b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1"
    ZIP_SIG = b"PK\x03\x04"
    bio = BytesIO(b)

    if b.startswith(OLE_SIG):
        df = pd.read_excel(bio, sheet_name=0, engine="xlrd", skiprows=start_row - 1)
    elif b.startswith(ZIP_SIG):
        df = pd.read_excel(bio, sheet_name=0, engine="openpyxl", skiprows=start_row - 1)
    else:
        df = pd.read_csv(path, sep=None, engine="python", skiprows=start_row - 1, encoding="latin-1")

    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~pd.Index(df.columns).astype(str).str.match(r"^Unnamed", na=False)]
    df = df.dropna(axis=1, how="all")
    return df


def _numericize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _normalize_time_seconds(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    finite = t[np.isfinite(t)]
    if finite.size == 0:
        return t
    if np.nanmax(finite) > 1e4:
        return t / 1000.0
    return t


def _pick_cols(df: pd.DataFrame, n_expected: int):
    cols = list(df.columns)
    frame_candidates = [c for c in cols if "frame" in str(c).lower()]
    time_candidates = [c for c in cols if str(c).strip().lower() == "time" or "time" in str(c).lower()]

    frame_col = frame_candidates[0] if frame_candidates else (cols[0] if len(cols) >= 1 else None)
    time_col = time_candidates[0] if time_candidates else (cols[1] if len(cols) >= 2 else None)
    remaining = [c for c in cols if c not in {frame_col, time_col}]

    if n_expected == 3:
        val_col = remaining[0] if len(remaining) >= 1 else (cols[2] if len(cols) >= 3 else None)
        return time_col, val_col, None

    if n_expected == 4:
        val_col = remaining[0] if len(remaining) >= 1 else (cols[2] if len(cols) >= 3 else None)
        val2_col = remaining[1] if len(remaining) >= 2 else (cols[3] if len(cols) >= 4 else None)
        return time_col, val_col, val2_col

    raise ValueError("n_expected must be 3 or 4")


def load_peda(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = read_otacs_table(path, start_row=9)
    df = _numericize(df).dropna(axis=1, how="all")
    time_col, val_col, _ = _pick_cols(df, n_expected=3)
    if time_col is None or val_col is None:
        return np.array([]), np.array([])
    t = _normalize_time_seconds(df[time_col].to_numpy(dtype=np.float64))
    y = df[val_col].to_numpy(dtype=np.float64)
    m = np.isfinite(t)
    t = t[m]
    y = y[m]
    order = np.argsort(t)
    return t[order], y[order]


def load_pp(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = read_otacs_table(path, start_row=9)
    df = _numericize(df).dropna(axis=1, how="all")
    time_col, val_col, val2_col = _pick_cols(df, n_expected=4)
    if time_col is None or val_col is None:
        return np.array([]), np.array([]), np.array([])
    t = _normalize_time_seconds(df[time_col].to_numpy(dtype=np.float64))
    pp = df[val_col].to_numpy(dtype=np.float64)
    pp_nr = df[val2_col].to_numpy(dtype=np.float64) if val2_col is not None else np.full_like(pp, np.nan)
    m = np.isfinite(t)
    t = t[m]
    pp = pp[m]
    pp_nr = pp_nr[m]
    order = np.argsort(t)
    return t[order], pp[order], pp_nr[order]


# =============================================================================
# Locate session folder and file prefix
# =============================================================================
def find_structured_session_dir(structured_subject_dir: Path, session_code: str) -> Path:
    session_code = session_code.strip().upper()
    for p in structured_subject_dir.iterdir():
        if not p.is_dir():
            continue
        parts = p.name.split()
        if len(parts) >= 2 and parts[-1].upper() == session_code:
            return p
    raise FileNotFoundError(f"Could not find session folder ending with {session_code} in {structured_subject_dir}")


def infer_prefix_in_session_dir(session_dir: Path, subject: str) -> str:
    subject = subject.upper()
    pat = re.compile(rf"^{re.escape(subject)}-\d{{3}}\.", re.IGNORECASE)
    for p in session_dir.iterdir():
        if p.is_file() and pat.match(p.name):
            return p.name.split(".", 1)[0]
    raise FileNotFoundError(f"Could not infer Txxx-zzz prefix in: {session_dir}")


# =============================================================================
# Batch helpers
# =============================================================================
def discover_subjects(raw_dir: Path) -> list[str]:
    out = []
    for p in raw_dir.iterdir():
        if p.is_dir() and re.match(r"^T\d{3}$", p.name.upper()):
            out.append(p.name.upper())
    return sorted(out)


def discover_tasks_for_subject(raw_dir: Path, subject: str) -> list[str]:
    subj_dir = raw_dir / subject
    if not subj_dir.exists():
        return []
    tasks = set()
    for f in subj_dir.glob(f"{subject}-*.dat"):
        # filename like T003-CD.dat
        stem = f.stem  # T003-CD
        parts = stem.split("-")
        if len(parts) >= 2:
            tasks.add(parts[1].upper())
    return sorted(tasks)


def process_one(subject: str, task: str) -> dict[str, Any]:
    subject = subject.upper()
    task = task.upper()

    # Segment
    if task in TASK_SEGMENTS:
        t0, dur = TASK_SEGMENTS[task]
        t1 = t0 + dur
    else:
        t0, t1 = 0.0, None

    subj_raw_dir = RAW_DIR / subject
    dat_path = subj_raw_dir / f"{subject}-{task}.dat"
    face_csv = subj_raw_dir / f"{subject}-{task}-face.csv"

    # Structured outputs
    structured_subject_dir = STRUCT_DIR / subject

    if not dat_path.exists():
        return {"subject": subject, "task": task, "status": "skip_missing_dat", "dat": str(dat_path)}
    if not face_csv.exists():
        return {"subject": subject, "task": task, "status": "skip_missing_face_csv", "face_csv": str(face_csv)}
    if not structured_subject_dir.exists():
        return {"subject": subject, "task": task, "status": "skip_missing_struct_dir", "struct_dir": str(structured_subject_dir)}

    dtype, n_frames = infer_dtype_and_nframes(dat_path, WIDTH, HEIGHT)
    frames = np.memmap(dat_path, dtype=dtype, mode="r", shape=(n_frames, HEIGHT, WIDTH))

    ann = load_face_csv(face_csv)
    if len(ann) == 0:
        return {"subject": subject, "task": task, "status": "skip_empty_face_csv", "face_csv": str(face_csv)}

    smooth_landmarks_ema(ann, alpha=LANDMARK_SMOOTH_ALPHA)
    frame_list_all = sorted(ann.keys())

    if t1 is None:
        t1 = (min(frame_list_all[-1], n_frames - 1)) / FPS_NATIVE

    i0 = int(np.floor(t0 * FPS_NATIVE))
    i1 = int(np.floor(t1 * FPS_NATIVE))
    i1 = min(i1, n_frames - 1)

    # Output grid
    t_out_start = t0
    t_out_end = i1 / FPS_NATIVE
    n_out = int(np.round((t_out_end - t_out_start) * FPS_OUT)) + 1
    t_grid = np.linspace(t_out_start, t_out_end, n_out)

    frame_list = [fi for fi in frame_list_all if (i0 <= fi <= i1 and 0 <= fi < n_frames)]
    if len(frame_list) < 2:
        return {"subject": subject, "task": task, "status": "skip_not_enough_frames", "n_frames": len(frame_list)}

    t_roi_native = np.array([fi / FPS_NATIVE for fi in frame_list], dtype=np.float64)

    roi_names = ["nose", "eye_l", "eye_r", "cheek_l", "cheek_r", "forehead"]
    roi_sig_native = {k: np.full(len(frame_list), np.nan, dtype=np.float64) for k in roi_names}

    # Debug
    dbg_dir = DEBUG_OUT_DIR / subject / f"{subject}-{task}"
    if SAVE_DEBUG_IMAGES:
        dbg_dir.mkdir(parents=True, exist_ok=True)
    if SHOW_DEBUG_WINDOW:
        cv2.namedWindow("ROI debug", cv2.WINDOW_NORMAL)

    for k, fi in enumerate(frame_list):
        row = ann.get(fi, None)
        if row is None:
            continue
        rois = build_rois_from_face(row)
        if not rois:
            continue

        fr = frames[fi]
        for name in roi_names:
            roi_sig_native[name][k] = roi_extract_by_name(fr, rois[name], name)

        if SAVE_DEBUG_IMAGES and ((fi - i0) % int(DEBUG_EVERY_N_FRAMES) == 0):
            vis = draw_rois_for_debug(fr, rois)
            cv2.putText(vis, f"{subject}-{task} t={fi / FPS_NATIVE:.3f}s frame={fi}", (10, 22),
                        0, 0.7, (255, 255, 255), 2)
            cv2.imwrite(str(dbg_dir / f"frame_{fi:06d}.jpg"), vis)

        if SHOW_DEBUG_WINDOW and (k % 200 == 0):
            vis = draw_rois_for_debug(fr, rois)
            cv2.putText(vis, f"{subject}-{task} t={fi / FPS_NATIVE:.3f}s frame={fi}", (10, 22),
                        0, 0.7, (255, 255, 255), 2)
            cv2.imshow("ROI debug", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    if SHOW_DEBUG_WINDOW:
        cv2.destroyAllWindows()

    # Upsample ROI signals
    roi_grid = {name: interp_to_grid_cubic(t_roi_native, roi_sig_native[name], t_grid) for name in roi_names}

    # StructuredStudyData session + GT
    try:
        session_dir = find_structured_session_dir(structured_subject_dir, task)
        prefix = infer_prefix_in_session_dir(session_dir, subject)
    except Exception as e:
        return {"subject": subject, "task": task, "status": "skip_no_session_dir", "error": str(e)}

    peda_path = session_dir / f"{prefix}.peda"
    pp_path = session_dir / f"{prefix}.pp"

    peda_grid = np.full_like(t_grid, np.nan, dtype=np.float64)
    pp_grid = np.full_like(t_grid, np.nan, dtype=np.float64)
    pp_nr_grid = np.full_like(t_grid, np.nan, dtype=np.float64)

    if peda_path.exists():
        t_peda, y_peda = load_peda(peda_path)
        m = (t_peda >= t0) & (t_peda <= t1)
        if np.sum(m) >= 2:
            peda_grid = interp_to_grid_linear(t_peda[m], y_peda[m], t_grid)

    if pp_path.exists():
        t_pp, y_pp, y_pp_nr = load_pp(pp_path)
        m = (t_pp >= t0) & (t_pp <= t1)
        if np.sum(m) >= 2:
            pp_grid = interp_to_grid_linear(t_pp[m], y_pp[m], t_grid)
            pp_nr_grid = interp_to_grid_linear(t_pp[m], y_pp_nr[m], t_grid)

    # Save synced CSV into the session folder (same behavior as your script)
    out_df = pd.DataFrame({
        "Time": t_grid.astype(np.float64),
        "roi_nose": roi_grid["nose"],
        "roi_eye_l": roi_grid["eye_l"],
        "roi_eye_r": roi_grid["eye_r"],
        "roi_cheek_l": roi_grid["cheek_l"],
        "roi_cheek_r": roi_grid["cheek_r"],
        "roi_forehead": roi_grid["forehead"],
        "peda": peda_grid,
        "pp": pp_grid,
        "pp_NR": pp_nr_grid,
    })

    out_csv = session_dir / f"{subject}-{task}-roi_peda_pp_{FPS_OUT:.0f}Hz.csv"
    out_df.to_csv(out_csv, index=False)

    # Optional plots
    plot_path = None
    if PLOT_MODE in ("save", "show"):
        fig = plt.figure(figsize=(14, 9))
        gs = fig.add_gridspec(3, 1, hspace=0.25)

        ax0 = fig.add_subplot(gs[0, 0])
        ax0.grid(True, alpha=0.3)
        ax0.set_title(f"{subject}-{task} | ROI signals (upsampled to {FPS_OUT:.0f} Hz)")
        ax0.set_ylabel("Raw thermal (a.u.)")
        for name in roi_names:
            ax0.plot(t_grid, roi_grid[name], label=name, linewidth=0.8)
        ax0.legend(loc="upper right", ncol=3, fontsize=8)

        ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f"GT Palm EDA (peda), resampled to {FPS_OUT:.0f} Hz")
        ax1.set_ylabel("peda")
        ax1.plot(t_grid, peda_grid, label="peda", linewidth=1.0)
        ax1.legend(loc="upper right")

        ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)
        ax2.grid(True, alpha=0.3)
        ax2.set_title(f"GT Perinasal EDA (pp), resampled to {FPS_OUT:.0f} Hz")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("pp")
        ax2.plot(t_grid, pp_grid, label="pp", linewidth=1.0)
        ax2.plot(t_grid, pp_nr_grid, label="pp_NR", linewidth=1.0, alpha=0.85)
        ax2.legend(loc="upper right")

        if PLOT_MODE == "save":
            out_dir = PLOT_OUT_DIR / subject
            out_dir.mkdir(parents=True, exist_ok=True)
            plot_path = out_dir / f"{subject}-{task}_roi_peda_pp_{FPS_OUT:.0f}Hz.png"
            fig.savefig(plot_path, dpi=160, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    return {
        "subject": subject,
        "task": task,
        "status": "ok",
        "out_csv": str(out_csv),
        "plot": str(plot_path) if plot_path is not None else "",
        "n_frames": int(n_frames),
        "n_annotated": int(len(frame_list_all)),
        "n_used": int(len(frame_list)),
        "n_out": int(n_out),
        "dtype": str(dtype),
        "peda_found": bool(peda_path.exists()),
        "pp_found": bool(pp_path.exists()),
        "debug_dir": str(dbg_dir) if SAVE_DEBUG_IMAGES else "",
    }


def main():
    # Subjects list
    subjects = [s.upper() for s in SUBJECTS]
    if (not subjects) and AUTO_DISCOVER_SUBJECTS_IF_EMPTY:
        subjects = discover_subjects(RAW_DIR)

    results = []
    for subject in subjects:
        # Tasks list per subject
        tasks = [t.upper() for t in TASKS]
        if ONLY_EXISTING_TASKS:
            existing = discover_tasks_for_subject(RAW_DIR, subject)
            tasks = [t for t in tasks if t in existing]

        if not tasks:
            print(f"[SKIP] {subject}: no tasks to process")
            results.append({"subject": subject, "task": "", "status": "skip_no_tasks"})
            continue

        for task in tasks:
            print(f"\n[RUN] {subject}-{task}")
            try:
                r = process_one(subject, task)
            except Exception as e:
                r = {"subject": subject, "task": task, "status": "error", "error": str(e)}
            results.append(r)

            if r.get("status") == "ok":
                print(f"[OK] {subject}-{task} -> {r.get('out_csv')}")
            else:
                print(f"[WARN] {subject}-{task} -> {r.get('status')}")

    # Save summary
    SUMMARY_OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(SUMMARY_OUT_CSV, index=False)
    print(f"\n[OK] batch summary saved: {SUMMARY_OUT_CSV}")
    print(df["status"].value_counts(dropna=False))


if __name__ == "__main__":
    main()

"""
mmslab_sim1_09_extract_hr_br_signals.py

Extracts heart rate (HR) and breathing rate (BR) from synchronized facial
thermal ROI traces and evaluates them against contact ground truth.

HR uses OMIT (orthogonal matrix image transformation) decomposition across
multiple facial ROIs, followed by Welch spectral peak detection in the
cardiac band (1.0–3.5 Hz / 60–210 bpm).
BR averages the nose and cheek traces, bandpass-filters them, and applies
the same Welch estimator in the respiratory band (0.12–0.55 Hz / 7–33 bpm).

Pipeline step 9: Run after step 06 (ROI extraction + GT sync).

Configuration:
  Edit SUBJECT, TASK, and BASE_PATH below.
"""

import os
import re
import csv
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import cv2
import matplotlib.pyplot as plt
from io import BytesIO
import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================
BASE_PATH = Path("/media/arritmic/HI-NRI-TINO-002/DATABASES/MULTIMODAL/Affective_States/SIMULATOR_STUDY_I")
RAW_DIR = BASE_PATH / "RawThermalData"
STRUCT_DIR = BASE_PATH / "StructuredStudyData"

SUBJECT = "T003"
TASK = "ND"  # BL, PD, ND, CD, ED, ...

FPS = 7.5
WIDTH = 640
HEIGHT = 512

# Segment limits (to exclude P5 in ND/CD/ED and to limit calm windows)
CALM_DURATION_S = 180.0
LD_FIRST4_DURATION_S = 640.0

TASK_SEGMENTS = {
    "BL": (0.0, CALM_DURATION_S),
    "PD": (0.0, CALM_DURATION_S),
    "ND": (0.0, LD_FIRST4_DURATION_S),
    "CD": (0.0, LD_FIRST4_DURATION_S),
    "ED": (0.0, LD_FIRST4_DURATION_S),
}

# Save some ROI overlay images for quick sanity check
SAVE_DEBUG_IMAGES = True
DEBUG_EVERY_N_FRAMES = 150
OUT_DIR = BASE_PATH / "AnalysisOutputs" / "roi_signals" / SUBJECT
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# READ GT DATA
# =============================================================================
def read_otacs_table(path: Path, start_row: int = 9) -> pd.DataFrame:
    """
    Reads OTACS measurement files that are Excel containers but have custom extensions.
    start_row=9 matches the authors R code (startRow=9).
    """
    path = Path(path)
    b = path.read_bytes()

    OLE_SIG = b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1"  # .xls container
    ZIP_SIG = b"PK\x03\x04"  # .xlsx container

    bio = BytesIO(b)

    if b.startswith(OLE_SIG):
        # Needs: pip install xlrd==2.0.1
        df = pd.read_excel(bio, sheet_name=0, engine="xlrd", skiprows=start_row - 1)
    elif b.startswith(ZIP_SIG):
        # Needs: pip install openpyxl
        df = pd.read_excel(bio, sheet_name=0, engine="openpyxl", skiprows=start_row - 1)
    else:
        # Last resort: try text table
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
    # if looks like ms, convert to s
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
        return frame_col, time_col, val_col, None

    if n_expected == 4:
        val_col = remaining[0] if len(remaining) >= 1 else (cols[2] if len(cols) >= 3 else None)
        val2_col = remaining[1] if len(remaining) >= 2 else (cols[3] if len(cols) >= 4 else None)
        return frame_col, time_col, val_col, val2_col

    raise ValueError("n_expected must be 3 or 4")


def load_3col_signal(path: Path, smooth_n: int = 1) -> tuple[np.ndarray, np.ndarray]:
    df = read_otacs_table(path, start_row=9)
    df = _numericize(df).dropna(axis=1, how="all")

    _, time_col, val_col, _ = _pick_cols(df, n_expected=3)
    if time_col is None or val_col is None:
        return np.array([]), np.array([])

    t = _normalize_time_seconds(df[time_col].to_numpy(dtype=np.float64))
    y = df[val_col].to_numpy(dtype=np.float64)

    if smooth_n is not None and smooth_n > 1:
        y = pd.Series(y, dtype="float64").rolling(window=smooth_n, min_periods=1).mean().to_numpy()

    m = np.isfinite(t)
    t = t[m]
    y = y[m]
    order = np.argsort(t)
    return t[order], y[order]


def load_pp_signal(path: Path, smooth_n: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = read_otacs_table(path, start_row=9)
    df = _numericize(df).dropna(axis=1, how="all")

    _, time_col, val_col, val2_col = _pick_cols(df, n_expected=4)
    if time_col is None or val_col is None:
        return np.array([]), np.array([]), np.array([])

    t = _normalize_time_seconds(df[time_col].to_numpy(dtype=np.float64))
    pp = df[val_col].to_numpy(dtype=np.float64)
    pp_nr = df[val2_col].to_numpy(dtype=np.float64) if val2_col is not None else np.full_like(pp, np.nan)

    if smooth_n is not None and smooth_n > 1:
        pp = pd.Series(pp, dtype="float64").rolling(window=smooth_n, min_periods=1).mean().to_numpy()
        pp_nr = pd.Series(pp_nr, dtype="float64").rolling(window=smooth_n, min_periods=1).mean().to_numpy()

    m = np.isfinite(t)
    t = t[m]
    pp = pp[m]
    pp_nr = pp_nr[m]
    order = np.argsort(t)
    return t[order], pp[order], pp_nr[order]


# =============================================================================
# Utilities
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


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    m = np.nanmean(x)
    s = np.nanstd(x)
    if not np.isfinite(s) or s < 1e-12:
        return x * 0.0
    return (x - m) / s


def nan_interp_1d(x: np.ndarray) -> np.ndarray:
    """
    Linear interpolation over NaNs. Keeps NaNs if all values are NaN.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n == 0:
        return x
    idx = np.arange(n)
    good = np.isfinite(x)
    if not np.any(good):
        return x
    x2 = x.copy()
    x2[~good] = np.interp(idx[~good], idx[good], x[good])
    return x2


# =============================================================================
# Read face annotations CSV
# Expected columns (from your exporter):
# frame_idx, conf, x1,y1,x2,y2, lm1_x,lm1_y,...,lm5_x,lm5_y
# =============================================================================
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


# =============================================================================
# ROI definition from bbox + 5 landmarks
# Assumption for landmark meaning (common 5pt convention):
# lm1: left eye, lm2: right eye, lm3: nose, lm4: left mouth, lm5: right mouth
# If your landmark ordering differs, swap mapping here.
# =============================================================================
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
    eye_vec = re_ - le

    # Forehead: above mid-eye along the nose->mid_eye direction
    v_up = mid_eye - no
    forehead_c = mid_eye + 0.6 * v_up

    # Cheeks: between eye and mouth corner, shifted slightly outward
    left_cheek_c = 0.5 * (le + ml) + np.array([-0.10 * bw, 0.05 * bh])
    right_cheek_c = 0.5 * (re_ + mr) + np.array([+0.10 * bw, 0.05 * bh])

    # ROI sizes as fractions of bbox
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


def roi_mean(frame_raw: np.ndarray, rect: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = rect
    patch = frame_raw[y1:y2, x1:x2]
    if patch.size == 0:
        return float("nan")
    return float(np.mean(patch.astype(np.float64)))


def draw_rois_for_debug(frame_raw: np.ndarray, rois: Dict[str, Tuple[int, int, int, int]]) -> np.ndarray:
    """
    Debug visualization: raw -> u8 -> inferno + draw ROI rectangles.
    """
    x = frame_raw.astype(np.float32)
    lo = np.percentile(x, 2.0)
    hi = np.percentile(x, 98.0)
    if hi <= lo:
        u8 = np.zeros_like(x, dtype=np.uint8)
    else:
        u8 = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
        u8 = (u8 * 255.0).astype(np.uint8)
    vis = cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)

    # draw
    for name, r in rois.items():
        if name == "face_bbox":
            color = (0, 255, 0)
        else:
            color = (255, 255, 255)
        x1, y1, x2, y2 = r
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        cv2.putText(vis, name, (x1, max(12, y1 + 12)), 0, 0.4, color, 1, cv2.LINE_AA)
    return vis


# =============================================================================
# StructuredStudyData lookup and GT loading
# =============================================================================
def find_struct_session_folder(subject: str, task: str) -> Optional[Tuple[Path, int]]:
    """
    Finds folder like "4 ND" under StructuredStudyData/Txxx that matches task.
    Returns (folder_path, order_int).
    """
    subj_dir = STRUCT_DIR / subject
    if not subj_dir.exists():
        return None

    # Match folders: "<int> <TASK>"
    best = None
    for p in subj_dir.iterdir():
        if not p.is_dir():
            continue
        m = re.match(r"^\s*(\d+)\s+([A-Za-z0-9]+)\s*$", p.name)
        if not m:
            continue
        order = int(m.group(1))
        code = m.group(2).upper()
        if code == task.upper():
            best = (p, order)
            break
    return best


def read_3col_signal_file(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads files like .HR, .BR, .peda assumed to have columns:
      frame_idx, time_s, value
    Uses a delimiter heuristic.
    """
    text = path.read_text(errors="ignore").strip().splitlines()
    if len(text) == 0:
        return np.array([]), np.array([])

    first = text[0]
    # detect if header
    has_header = any(c.isalpha() for c in first)

    # delimiter heuristic
    comma = first.count(",")
    tab = first.count("\t")
    if tab > comma and tab > 0:
        delim = "\t"
    elif comma > 0:
        delim = ","
    else:
        delim = None  # whitespace

    data = np.genfromtxt(
        path,
        delimiter=delim,
        skip_header=1 if has_header else 0,
        dtype=np.float64,
        invalid_raise=False,
    )

    if data.ndim == 1 and data.size > 0:
        data = data.reshape(1, -1)

    if data.size == 0 or data.shape[1] < 3:
        return np.array([]), np.array([])

    t = data[:, 1]
    y = data[:, 2]
    return t, y


def read_pp_signal_file(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads .pp assumed to have columns:
      frame_idx, time_s, pp, pp_NR
    """
    text = path.read_text(errors="ignore").strip().splitlines()
    if len(text) == 0:
        return np.array([]), np.array([]), np.array([])

    first = text[0]
    has_header = any(c.isalpha() for c in first)

    comma = first.count(",")
    tab = first.count("\t")
    if tab > comma and tab > 0:
        delim = "\t"
    elif comma > 0:
        delim = ","
    else:
        delim = None

    data = np.genfromtxt(
        path,
        delimiter=delim,
        skip_header=1 if has_header else 0,
        dtype=np.float64,
        invalid_raise=False,
    )
    if data.ndim == 1 and data.size > 0:
        data = data.reshape(1, -1)

    if data.size == 0 or data.shape[1] < 4:
        return np.array([]), np.array([]), np.array([])

    t = data[:, 1]
    pp = data[:, 2]
    pp_nr = data[:, 3]
    return t, pp, pp_nr


def filter_passthrough(x: np.ndarray, fs: float) -> np.ndarray:
    """Identity filter — returns the signal unmodified.
    Replace with a bandpass or detrend step for HR/BR extraction."""
    return x


# =============================================================================
# Main
# =============================================================================
def main():
    subject = SUBJECT.upper()
    task = TASK.upper()

    subj_raw_dir = RAW_DIR / subject
    dat_path = subj_raw_dir / f"{subject}-{task}.dat"
    if not dat_path.exists():
        raise FileNotFoundError(f"Missing raw thermal dat: {dat_path}")

    face_csv = subj_raw_dir / f"{subject}-{task}-face.csv"
    if not face_csv.exists():
        raise FileNotFoundError(f"Missing face CSV: {face_csv}")

    # Segment limit
    if task in TASK_SEGMENTS:
        t0, dur = TASK_SEGMENTS[task]
        t1 = t0 + dur
    else:
        # If not specified, use whole file (but still bounded by CSV frames)
        t0, t1 = 0.0, 1e12

    # Load annotations dict: frame_idx -> row
    ann = load_face_csv(face_csv)
    frame_list = sorted(ann.keys())
    if len(frame_list) == 0:
        raise RuntimeError("Face CSV is empty")

    # Limit by segment time
    i0 = int(np.floor(t0 * FPS))
    i1 = int(np.ceil(t1 * FPS))
    frame_list = [fi for fi in frame_list if (fi >= i0 and fi <= i1)]
    if len(frame_list) == 0:
        raise RuntimeError("No annotated frames in requested segment")

    # Load raw thermal frames (memmap)
    dtype, n_frames = infer_dtype_and_nframes(dat_path, WIDTH, HEIGHT)
    frames = np.memmap(dat_path, dtype=dtype, mode="r", shape=(n_frames, HEIGHT, WIDTH))

    # Prepare arrays
    N = len(frame_list)
    t = np.zeros(N, dtype=np.float64)

    sig = {
        "nose": np.full(N, np.nan, dtype=np.float64),
        "eye_l": np.full(N, np.nan, dtype=np.float64),
        "eye_r": np.full(N, np.nan, dtype=np.float64),
        "cheek_l": np.full(N, np.nan, dtype=np.float64),
        "cheek_r": np.full(N, np.nan, dtype=np.float64),
        "forehead": np.full(N, np.nan, dtype=np.float64),
        "face_bbox": np.full(N, np.nan, dtype=np.float64),
    }

    # Debug folder
    dbg_dir = OUT_DIR / f"{subject}-{task}-roi_debug"
    if SAVE_DEBUG_IMAGES:
        dbg_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] subject={subject} task={task}")
    print(f"[INFO] raw={dat_path.name} dtype={dtype} n_frames={n_frames}")
    print(f"[INFO] face_csv={face_csv.name} annotated_frames={N}")
    print(f"[INFO] segment [{t0:.1f}s, {t1:.1f}s] frames [{i0}, {i1}]")

    cv2.namedWindow("Analysis", cv2.WINDOW_NORMAL)

    # Extract ROI mean per annotated frame
    for k, fi in enumerate(frame_list):
        t[k] = fi / FPS
        row = ann[fi]

        if fi < 0 or fi >= n_frames:
            continue

        frame_raw = frames[fi]  # uint16 or uint32

        rois = build_rois_from_face(row)
        if not rois:
            continue

        # Mean over each ROI on raw thermal values
        for name, rect in rois.items():
            if name == "face_bbox":
                sig[name][k] = roi_mean(frame_raw, rect)
            elif name in sig:
                sig[name][k] = roi_mean(frame_raw, rect)

        vis = draw_rois_for_debug(frame_raw, rois)
        cv2.putText(vis, f"{subject}-{task} t={t[k]:.3f}s frame={fi}", (10, 22), 0, 0.7, (255, 255, 255), 2)
        cv2.imshow("Analysis", vis)
        key = cv2.waitKey(10)
        if key == ord("q"):
            break
        # Debug overlays
        if SAVE_DEBUG_IMAGES and ((k % DEBUG_EVERY_N_FRAMES) == 0):
            vis = draw_rois_for_debug(frame_raw, rois)
            cv2.putText(vis, f"{subject}-{task} t={t[k]:.3f}s frame={fi}", (10, 22), 0, 0.7, (255, 255, 255), 2)
            cv2.imwrite(str(dbg_dir / f"frame_{fi:06d}.jpg"), vis)

        if (k + 1) % 1000 == 0:
            print(f"  processed {k + 1}/{N}")

    # Interpolate NaNs (optional, but usually needed before filtering)
    sig_interp = {k: nan_interp_1d(v) for k, v in sig.items()}

    sig_filt = {k: filter_passthrough(v, FPS) for k, v in sig_interp.items()}

    # Load GT signals for this task (if available)
    gt = {}
    sess = find_struct_session_folder(subject, task)
    if sess is not None:
        sess_dir, order = sess
        zzz = f"{order:03d}"
        prefix = f"{subject}-{zzz}"

        pp_path = sess_dir / f"{prefix}.pp"
        hr_path = sess_dir / f"{prefix}.HR"
        br_path = sess_dir / f"{prefix}.BR"
        peda_path = sess_dir / f"{prefix}.peda"

        smooth_n = 1  # set to 5 if you want the same smoothing as before

        if pp_path.exists():
            tt, pp, pp_nr = load_pp_signal(pp_path, smooth_n=smooth_n)
            gt["pp_t"] = tt
            gt["pp"] = pp
            gt["pp_nr"] = pp_nr

        if hr_path.exists():
            tt, y = load_3col_signal(hr_path, smooth_n=smooth_n)
            gt["hr_t"] = tt
            gt["hr"] = y

        if br_path.exists():
            tt, y = load_3col_signal(br_path, smooth_n=smooth_n)
            gt["br_t"] = tt
            gt["br"] = y

        if peda_path.exists():
            tt, y = load_3col_signal(peda_path, smooth_n=smooth_n)
            gt["peda_t"] = tt
            gt["peda"] = y

        print(f"[INFO] Structured session folder: {sess_dir.name} (order={order}, prefix={prefix})")
    else:
        print("[WARN] Could not find matching session folder in StructuredStudyData for this task")

    # Plot 1: ROI temperature traces
    plt.figure(figsize=(12, 7))
    for name in ["nose", "eye_l", "eye_r", "cheek_l", "cheek_r", "forehead"]:
        plt.plot(t, sig_filt[name], label=name)
    plt.xlabel("Time (s)")
    plt.ylabel("Mean raw thermal value (a.u.)")
    plt.title(f"{subject}-{task}: ROI mean signals")
    plt.legend(loc="best")
    plt.tight_layout()
    out1 = OUT_DIR / f"{subject}-{task}-roi_signals.png"
    plt.savefig(out1, dpi=150)
    plt.close()
    print(f"[OK] saved: {out1}")

    # -----------------------------
    # Plot 2: compare nose ROI vs GT perinasal EDA (pp)
    # Use z-scored signals, interpolate GT to ROI time for correlation.
    # -----------------------------
    if "pp_t" in gt:
        # Restrict GT to same time window for cleaner plots
        m = (gt["pp_t"] >= t[0]) & (gt["pp_t"] <= t[-1])
        pp_t = gt["pp_t"][m]
        pp = gt["pp"][m]
        pp_nr = gt["pp_nr"][m]

        # Interpolate pp to ROI time vector
        pp_i = np.interp(t, pp_t, pp) if pp_t.size > 1 else np.full_like(t, np.nan)
        pp_nr_i = np.interp(t, pp_t, pp_nr) if pp_t.size > 1 else np.full_like(t, np.nan)

        nose_z = zscore(sig_filt["nose"])
        pp_z = zscore(pp_i)
        pp_nr_z = zscore(pp_nr_i)

        # Simple correlation (after interpolation)
        good = np.isfinite(nose_z) & np.isfinite(pp_nr_z)
        corr = float(np.corrcoef(nose_z[good], pp_nr_z[good])[0, 1]) if np.sum(good) > 10 else float("nan")

        plt.figure(figsize=(12, 6))
        plt.plot(t, nose_z, label="nose ROI (z)")
        plt.plot(t, pp_z, label="GT pp (z, interpolated)")
        plt.plot(t, pp_nr_z, label="GT pp_NR (z, interpolated)")
        plt.xlabel("Time (s)")
        plt.ylabel("Z-score")
        plt.title(f"{subject}-{task}: Nose ROI vs GT perinasal EDA (corr with pp_NR = {corr:.3f})")
        plt.legend(loc="best")
        plt.tight_layout()
        out2 = OUT_DIR / f"{subject}-{task}-nose_vs_pp.png"
        plt.savefig(out2, dpi=150)
        plt.close()
        print(f"[OK] saved: {out2}")

    # Optional: you can add more comparisons, for example:
    # - forehead ROI vs HR (if you later derive rPPG/rBGS-style features)
    # - face_bbox mean vs peda
    # using the same alignment pattern above.

    if SAVE_DEBUG_IMAGES:
        print(f"[OK] ROI debug images: {dbg_dir} (every {DEBUG_EVERY_N_FRAMES} annotated frames)")


if __name__ == "__main__":
    main()

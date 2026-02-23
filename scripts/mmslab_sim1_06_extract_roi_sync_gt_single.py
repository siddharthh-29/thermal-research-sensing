"""
mmslab_sim1_06_extract_roi_sync_gt_single.py

Single-session version of step 06 (ROI extraction + ground-truth synchronization).
Processes one SUBJECT/TASK at a time.

For batch processing across multiple subjects and tasks, use:
  mmslab_sim1_06_extract_roi_sync_gt.py

Configuration:
  Edit SUBJECT, TASK, and BASE_PATH at the top of the script.
"""

import csv
import re
from io import BytesIO
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

# =============================================================================
# Configuration
# =============================================================================
BASE_PATH = Path("/media/arritmic/HI-NRI-TINO-002/DATABASES/MULTIMODAL/Affective_States/SIMULATOR_STUDY_I")
RAW_DIR = BASE_PATH / "RawThermalData"
STRUCT_DIR = BASE_PATH / "StructuredStudyData"

SUBJECT = "T003"
TASK = "PD"  # BL, PD, ND, CD, ED, ...

FPS = 7.5
WIDTH = 640
HEIGHT = 512

# Segment limits
CALM_DURATION_S = 180.0
LD_FIRST4_DURATION_S = 640.0
TASK_SEGMENTS = {
    "BL": (0.0, CALM_DURATION_S),
    "PD": (0.0, CALM_DURATION_S),
    "ND": (0.0, LD_FIRST4_DURATION_S),
    "CD": (0.0, LD_FIRST4_DURATION_S),
    "ED": (0.0, LD_FIRST4_DURATION_S),
}

# Debug: save overlay images sometimes
SAVE_DEBUG_IMAGES = True
DEBUG_EVERY_N_FRAMES = 150  # on the uniform 7.5 Hz grid
DEBUG_OUT_DIR = BASE_PATH / "AnalysisOutputs" / "roi_sync_debug"
SHOW_DEBUG_WINDOW = False  # set True if you want cv2.imshow


# -----------------------------
# Raw thermal + face CSV
# -----------------------------
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
    """
    Expected columns:
      frame_idx, conf, x1,y1,x2,y2, lm1_x,lm1_y,...,lm5_x,lm5_y
    """
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


# -----------------------------
# ROI geometry
# Assumption (typical 5pt order): lm1 left-eye, lm2 right-eye, lm3 nose, lm4 left-mouth, lm5 right-mouth
# If your ordering differs, swap mapping here.
# -----------------------------
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

    eye_rw, eye_rh = 0.22 * bw, 0.12 * bh
    nose_rw, nose_rh = 0.30 * bw, 0.20 * bh
    cheek_rw, cheek_rh = 0.15 * bw, 0.20 * bh
    forehead_rw, forehead_rh = 0.45 * bw, 0.18 * bh

    rois = {
        "nose": rect_from_center(no[0], no[1] + 20, nose_rw, nose_rh, w, h),
        "eye_l": rect_from_center(le[0] + 15, le[1], eye_rw, eye_rh, w, h),
        "eye_r": rect_from_center(re_[0] - 15, re_[1], eye_rw, eye_rh, w, h),
        "cheek_l": rect_from_center(left_cheek_c[0], left_cheek_c[1], cheek_rw, cheek_rh, w, h),
        "cheek_r": rect_from_center(right_cheek_c[0], right_cheek_c[1], cheek_rw, cheek_rh, w, h),
        "forehead": rect_from_center(forehead_c[0], forehead_c[1] - 20, forehead_rw, forehead_rh, w, h),
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


# -----------------------------
# Resampling utilities (to 7.5 Hz grid)
# -----------------------------
def interp_1d_to_grid(t_in: np.ndarray, y_in: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
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


# -----------------------------
# StructuredStudyData loaders for PEDA and PP (OTACS containers)
# -----------------------------
def read_otacs_table(path: Path, start_row: int = 9) -> pd.DataFrame:
    b = path.read_bytes()

    OLE_SIG = b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1"  # .xls
    ZIP_SIG = b"PK\x03\x04"  # .xlsx

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
    if np.nanmax(finite) > 1e4:  # ms
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


# -----------------------------
# Locate session folder and file prefix (Txxx-zzz)
# -----------------------------
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
            return p.name.split(".", 1)[0]  # Txxx-zzz
    raise FileNotFoundError(f"Could not infer Txxx-zzz prefix in: {session_dir}")


# -----------------------------
# Main
# -----------------------------
def main():
    subject = SUBJECT.upper()
    task = TASK.upper()

    # Segment
    if task in TASK_SEGMENTS:
        t0, dur = TASK_SEGMENTS[task]
        t1 = t0 + dur
    else:
        t0, t1 = 0.0, None  # will be clamped later

    # Paths
    subj_raw_dir = RAW_DIR / subject
    dat_path = subj_raw_dir / f"{subject}-{task}.dat"
    face_csv = subj_raw_dir / f"{subject}-{task}-face.csv"

    if not dat_path.exists():
        raise FileNotFoundError(f"Missing raw thermal dat: {dat_path}")
    if not face_csv.exists():
        raise FileNotFoundError(f"Missing face CSV: {face_csv}")

    dtype, n_frames = infer_dtype_and_nframes(dat_path, WIDTH, HEIGHT)
    frames = np.memmap(dat_path, dtype=dtype, mode="r", shape=(n_frames, HEIGHT, WIDTH))

    ann = load_face_csv(face_csv)

    # --- Smooth landmark trajectories to reduce ROI jitter ---
    alpha = 0.15
    prev = None
    for fi in sorted(ann.keys()):
        row = ann[fi]
        if prev is None:
            prev = row.copy()
            continue
        for key in ['x1', 'y1', 'x2', 'y2', 'lm1_x', 'lm1_y', 'lm2_x', 'lm2_y',
                    'lm3_x', 'lm3_y', 'lm4_x', 'lm4_y', 'lm5_x', 'lm5_y']:
            if np.isfinite(row[key]) and np.isfinite(prev[key]):
                row[key] = alpha * row[key] + (1 - alpha) * prev[key]
        prev = row
    # -----------------------------------------------------------

    frame_list_all = sorted(ann.keys())
    if len(frame_list_all) == 0:
        raise RuntimeError("Face CSV is empty")




    # Clamp segment to available frames
    if t1 is None:
        t1 = (min(frame_list_all[-1], n_frames - 1)) / FPS

    i0 = int(np.floor(t0 * FPS))
    i1 = int(np.floor(t1 * FPS))

    # Build uniform 7.5 Hz grid (frame indices) for the segment
    i1 = min(i1, n_frames - 1)
    grid_frames = np.arange(i0, i1 + 1, dtype=np.int64)
    t_grid = grid_frames.astype(np.float64) / FPS

    print(f"[INFO] subject={subject} task={task}")
    print(f"[INFO] raw={dat_path.name} dtype={dtype} n_frames={n_frames}")
    print(f"[INFO] face_csv={face_csv.name} annotated_frames_total={len(frame_list_all)}")
    print(f"[INFO] segment [{t0:.1f}s, {t1:.1f}s] -> frames [{i0}, {i1}] -> grid_n={len(grid_frames)} at {FPS} Hz")

    # Extract ROI means only on frames that have annotations within [i0, i1]
    frame_list = [fi for fi in frame_list_all if (i0 <= fi <= i1 and 0 <= fi < n_frames)]
    if len(frame_list) < 2:
        raise RuntimeError("Not enough annotated frames in the selected segment")

    t_roi = np.array([fi / FPS for fi in frame_list], dtype=np.float64)

    roi_names = ["nose", "eye_l", "eye_r", "cheek_l", "cheek_r", "forehead"]
    roi_sig = {k: np.full(len(frame_list), np.nan, dtype=np.float64) for k in roi_names}

    # Debug output folder
    dbg_dir = DEBUG_OUT_DIR / subject / f"{subject}-{task}"
    if SAVE_DEBUG_IMAGES:
        dbg_dir.mkdir(parents=True, exist_ok=True)

    if SHOW_DEBUG_WINDOW:
        cv2.namedWindow("ROI debug", cv2.WINDOW_NORMAL)

    for k, fi in enumerate(frame_list):
        row = ann[fi]
        rois = build_rois_from_face(row)
        if not rois:
            continue

        fr = frames[fi]
        for name in roi_names:
            roi_sig[name][k] = roi_mean(fr, rois[name])

        # optional debug
        if SAVE_DEBUG_IMAGES and ((fi - i0) % int(DEBUG_EVERY_N_FRAMES) == 0):
            vis = draw_rois_for_debug(fr, rois)
            cv2.putText(vis, f"{subject}-{task} t={fi / FPS:.3f}s frame={fi}", (10, 22),
                        0, 0.7, (255, 255, 255), 2)
            cv2.imwrite(str(dbg_dir / f"frame_{fi:06d}.jpg"), vis)

        if SHOW_DEBUG_WINDOW and (k % 200 == 0):
            vis = draw_rois_for_debug(fr, rois)
            cv2.putText(vis, f"{subject}-{task} t={fi / FPS:.3f}s frame={fi}", (10, 22),
                        0, 0.7, (255, 255, 255), 2)
            cv2.imshow("ROI debug", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        if (k + 1) % 1000 == 0:
            print(f"  extracted {k + 1}/{len(frame_list)} annotated frames")

    if SHOW_DEBUG_WINDOW:
        cv2.destroyAllWindows()

    # Resample ROI signals to the uniform 7.5 Hz grid
    roi_grid = {}
    for name in roi_names:
        roi_grid[name] = interp_1d_to_grid(t_roi, roi_sig[name], t_grid)

    # -----------------------------
    # Load GT PEDA and PP from StructuredStudyData, then resample to 7.5 Hz grid
    # -----------------------------
    structured_subject_dir = STRUCT_DIR / subject
    session_dir = find_structured_session_dir(structured_subject_dir, task)
    prefix = infer_prefix_in_session_dir(session_dir, subject)  # Txxx-zzz

    peda_path = session_dir / f"{prefix}.peda"
    pp_path = session_dir / f"{prefix}.pp"

    peda_grid = np.full_like(t_grid, np.nan, dtype=np.float64)
    pp_grid = np.full_like(t_grid, np.nan, dtype=np.float64)
    pp_nr_grid = np.full_like(t_grid, np.nan, dtype=np.float64)

    if peda_path.exists():
        t_peda, y_peda = load_peda(peda_path)
        # clip to segment, then interpolate
        m = (t_peda >= t0) & (t_peda <= t1)
        if np.sum(m) >= 2:
            peda_grid = interp_1d_to_grid(t_peda[m], y_peda[m], t_grid)
        print(f"[INFO] loaded PEDA: {peda_path.name} n={len(t_peda)}")

    else:
        print(f"[WARN] missing PEDA: {peda_path}")

    if pp_path.exists():
        t_pp, y_pp, y_pp_nr = load_pp(pp_path)
        m = (t_pp >= t0) & (t_pp <= t1)
        if np.sum(m) >= 2:
            pp_grid = interp_1d_to_grid(t_pp[m], y_pp[m], t_grid)
            pp_nr_grid = interp_1d_to_grid(t_pp[m], y_pp_nr[m], t_grid)
        print(f"[INFO] loaded PP: {pp_path.name} n={len(t_pp)}")

    else:
        print(f"[WARN] missing PP: {pp_path}")

    # -----------------------------
    # Save synchronized CSV (ROIs + PEDA + PP)
    # -----------------------------
    out_df = pd.DataFrame({
        "frame_idx": grid_frames.astype(np.int64),
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

    out_csv = session_dir / f"{subject}-{task}-roi_peda_pp_{FPS:.1f}Hz.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"[OK] saved synced CSV: {out_csv}")

    # -----------------------------
    # Visualization (only ROIs + PEDA + PP)
    # -----------------------------
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(3, 1, hspace=0.25)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.grid(True, alpha=0.3)
    ax0.set_title(f"{subject}-{task} | ROI mean signals (resampled to {FPS} Hz)")
    ax0.set_ylabel("Raw thermal (a.u.)")
    ax0.plot(t_grid, roi_grid["nose"], label="nose", linewidth=1.0)
    ax0.plot(t_grid, roi_grid["eye_l"], label="eye_l", linewidth=1.0)
    ax0.plot(t_grid, roi_grid["eye_r"], label="eye_r", linewidth=1.0)
    ax0.plot(t_grid, roi_grid["cheek_l"], label="cheek_l", linewidth=1.0)
    ax0.plot(t_grid, roi_grid["cheek_r"], label="cheek_r", linewidth=1.0)
    ax0.plot(t_grid, roi_grid["forehead"], label="forehead", linewidth=1.0)
    ax0.legend(loc="upper right", ncol=3, fontsize=8)

    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("GT Palm EDA (peda), resampled to 7.5 Hz")
    ax1.set_ylabel("peda")
    ax1.plot(t_grid, peda_grid, label="peda", linewidth=1.0)
    ax1.legend(loc="upper right")

    ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)
    ax2.grid(True, alpha=0.3)
    ax2.set_title("GT Perinasal EDA (pp), resampled to 7.5 Hz")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("pp")
    ax2.plot(t_grid, pp_grid, label="pp", linewidth=1.0)
    ax2.plot(t_grid, pp_nr_grid, label="pp_NR", linewidth=1.0, alpha=0.85)
    ax2.legend(loc="upper right")

    plt.show()

    if SAVE_DEBUG_IMAGES:
        print(f"[OK] debug ROI overlays: {dbg_dir}")


if __name__ == "__main__":
    main()

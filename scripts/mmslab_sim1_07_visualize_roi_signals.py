"""
mmslab_sim1_06_visualize_roi_signals.py

Visualizes synchronized ROI temperature traces alongside contact ground-truth
signals (HR, BR, PEDA, PP) for a given subject and task.

Pipeline step 6: Quality-check the synchronized CSV before signal extraction.

Configuration:
  Edit SUBJECT, TASK, BASE_PATH, and output options at the top of the script.
"""

import os
import re
from pathlib import Path
from io import BytesIO

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Publication-quality plot defaults
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 12,  # General font size reduced
    'axes.titlesize': 14,  # Subplot titles
    'axes.labelsize': 12,  # Axis labels (x, y)
    'xtick.labelsize': 10,  # X-axis tick labels
    'ytick.labelsize': 10,  # Y-axis tick labels (Reduced as requested)
    'legend.fontsize': 10,  # Legend text
    'figure.titlesize': 20  # Main title
})

# -------------------------------------------------
# Robust INF parsing (raw thermal)
# -------------------------------------------------
_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _extract_nums_from_text(text: str):
    return _NUM_RE.findall(text)


def _infer_header_from_dat_size(
        dat_path: str,
        wh_candidates=((640, 512), (512, 640), (320, 240), (240, 320), (640, 480), (480, 640)),
):
    size_bytes = os.path.getsize(dat_path)
    for (w, h) in wh_candidates:
        for bps in (2, 4, 1):
            denom = w * h * bps
            if denom <= 0: continue
            n = size_bytes / denom
            if abs(n - round(n)) < 1e-9 and n > 1:
                return int(round(n)), w, h
    raise ValueError(f"Could not infer header from dat size for {dat_path}. size={size_bytes} bytes")


def read_inf_any(inf_path: str, dat_path: str | None = None):
    b = Path(inf_path).read_bytes()
    encodings = ("utf-8-sig", "utf-8", "utf-16", "utf-16le", "utf-16be", "latin-1")
    nums = []
    for enc in encodings:
        try:
            text = b.decode(enc, errors="ignore")
            nums = _extract_nums_from_text(text)
            if len(nums) >= 3: break
        except Exception:
            continue

    if len(nums) >= 3:
        n_frames = int(float(nums[0]))
        width = int(float(nums[1]))
        height = int(float(nums[2]))
        raw = np.array([float(x) for x in nums[3:]], dtype=np.float64)
        if raw.size == 0:
            ts = raw
        elif raw.size == n_frames:
            ts = raw
        elif raw.size % n_frames == 0:
            k = raw.size // n_frames
            ts = raw.reshape(n_frames, k)[:, -1]
        else:
            ts = raw[:n_frames]
            if ts.size < n_frames:
                ts = np.pad(ts, (0, n_frames - ts.size), constant_values=np.nan)
        return n_frames, width, height, ts

    if len(b) >= 12:
        hdr = np.frombuffer(b[:12], dtype="<i4", count=3)
        n_frames, width, height = int(hdr[0]), int(hdr[1]), int(hdr[2])
        if 1 < n_frames < 10_000_000 and 1 < width < 5000 and 1 < height < 5000:
            rest = b[12:]
            if len(rest) >= n_frames * 8:
                ts = np.frombuffer(rest[: n_frames * 8], dtype="<f8").astype(np.float64)
            elif len(rest) >= n_frames * 4:
                ts = np.frombuffer(rest[: n_frames * 4], dtype="<f4").astype(np.float64)
            else:
                ts = np.array([], dtype=np.float64)
            return n_frames, width, height, ts

    if dat_path is not None and os.path.exists(dat_path):
        n_frames, width, height = _infer_header_from_dat_size(dat_path)
        ts = np.array([], dtype=np.float64)
        return n_frames, width, height, ts

    raise ValueError(f"Could not parse header from {inf_path}")


def normalize_timestamps_to_seconds(ts: np.ndarray) -> np.ndarray:
    if ts is None: return np.array([], dtype=np.float64)
    ts = np.asarray(ts, dtype=np.float64)
    if ts.size == 0: return ts
    finite = ts[np.isfinite(ts)]
    if finite.size == 0: return ts
    if np.nanmax(finite) > 1e4: return ts / 1000.0
    return ts


def guess_dtype_from_size(dat_path: str, n_frames: int, width: int, height: int) -> np.dtype:
    expected_samples = n_frames * width * height
    size_bytes = os.path.getsize(dat_path)
    bps = size_bytes / expected_samples
    if abs(bps - 2.0) < 0.1: return np.dtype("<u2")
    if abs(bps - 1.0) < 0.1: return np.dtype("u1")
    if abs(bps - 4.0) < 0.1: return np.dtype("<u4")
    raise ValueError(f"Unexpected bytes per sample: {bps:.4f}. File size={size_bytes} bytes.")


def load_thermal_dat(dat_path: str, inf_path: str, use_memmap: bool = True):
    n_frames, width, height, ts = read_inf_any(inf_path, dat_path=dat_path)
    dtype = guess_dtype_from_size(dat_path, n_frames, width, height)
    shape = (n_frames, height, width)
    if use_memmap:
        frames = np.memmap(dat_path, dtype=dtype, mode="r", shape=shape)
    else:
        frames_1d = np.fromfile(dat_path, dtype=dtype)
        if frames_1d.size != n_frames * width * height:
            raise ValueError(f"Sample count mismatch. Expected {n_frames * width * height}, got {frames_1d.size}")
        frames = frames_1d.reshape(shape)
    return frames, ts, (width, height), dtype


def normalize_for_display(img: np.ndarray, p_low=2, p_high=98) -> np.ndarray:
    x = img.astype(np.float32)
    lo = np.percentile(x, p_low)
    hi = np.percentile(x, p_high)
    if hi <= lo: return np.zeros_like(x, dtype=np.float32)
    x = (x - lo) / (hi - lo)
    return np.clip(x, 0.0, 1.0)


# -------------------------------------------------
# StructuredStudyData helpers
# -------------------------------------------------
def norm_session_code(code: str) -> str:
    code = code.strip().upper()
    return "BL" if code == "B" else code


def find_session_file(session_dir: Path, prefix: str, ext: str) -> Path | None:
    pref = f"{prefix}.{ext}".lower()
    for p in session_dir.iterdir():
        if p.is_file() and p.name.lower().startswith(pref): return p
    return None


def find_structured_session_dir(structured_subject_dir: Path, session_code: str) -> Path:
    session_code = norm_session_code(session_code)
    if not structured_subject_dir.is_dir():
        raise FileNotFoundError(f"Structured subject directory not found: {structured_subject_dir}")
    for p in structured_subject_dir.iterdir():
        if not p.is_dir(): continue
        parts = p.name.split()
        if len(parts) >= 2 and parts[-1].upper() == session_code: return p
    raise FileNotFoundError(f"Could not find session folder ending with {session_code} in {structured_subject_dir}")


def infer_prefix_in_session_dir(session_dir: Path, subject: str) -> str:
    subject = subject.upper()
    pat = re.compile(rf"^{re.escape(subject)}-\d{{3}}\.", re.IGNORECASE)
    for p in session_dir.iterdir():
        if p.is_file() and pat.match(p.name): return p.name.split(".", 1)[0]
    raise FileNotFoundError(f"Could not infer Txxx-zzz prefix in: {session_dir}")


# -------------------------------------------------
# OTACS Excel reader
# -------------------------------------------------
def read_otacs_table(path: Path, start_row: int = 9) -> pd.DataFrame:
    path = Path(path)
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
    for c in out.columns: out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _normalize_time_col_seconds(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    finite = t[np.isfinite(t)]
    if finite.size == 0: return t
    if np.nanmax(finite) > 1e4: return t / 1000.0
    return t


def clip_df_time(df: pd.DataFrame, t0: float, t1: float) -> pd.DataFrame:
    if df is None or len(df) == 0 or "Time" not in df.columns:
        return df.iloc[0:0].copy() if df is not None else pd.DataFrame()
    out = df.copy()
    out["Time"] = pd.to_numeric(out["Time"], errors="coerce")
    out = out.dropna(subset=["Time"])
    out = out.sort_values("Time").drop_duplicates(subset=["Time"], keep="first").reset_index(drop=True)
    out = out[(out["Time"] >= t0) & (out["Time"] <= t1)].reset_index(drop=True)
    return out


def sma(x: np.ndarray, n: int = 5) -> np.ndarray:
    if n is None or n <= 1: return np.asarray(x, dtype=np.float64)
    return pd.Series(x, dtype="float64").rolling(window=n, min_periods=1).mean().to_numpy()


def clean_range_to_nan(x: np.ndarray, low: float, high: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).copy()
    x[(x <= low) | (x >= high)] = np.nan
    return x


def _pick_cols_by_position_or_name(df: pd.DataFrame, n_expected: int):
    cols = list(df.columns)
    frame_candidates = [c for c in cols if "frame" in str(c).lower()]
    time_candidates = [c for c in cols if str(c).strip().lower() == "time" or "time" in str(c).lower()]
    frame_col = frame_candidates[0] if frame_candidates else None
    time_col = time_candidates[0] if time_candidates else None
    if time_col is None and len(cols) >= 2: time_col = cols[1]
    if frame_col is None and len(cols) >= 1: frame_col = cols[0]
    remaining = [c for c in cols if c not in {frame_col, time_col}]
    if n_expected == 3:
        val_col = remaining[0] if len(remaining) >= 1 else (cols[2] if len(cols) >= 3 else None)
        return frame_col, time_col, val_col, None
    if n_expected == 4:
        val_col = remaining[0] if len(remaining) >= 1 else (cols[2] if len(cols) >= 3 else None)
        val2_col = remaining[1] if len(remaining) >= 2 else (cols[3] if len(cols) >= 4 else None)
        return frame_col, time_col, val_col, val2_col
    raise ValueError("n_expected must be 3 or 4")


def load_hr_br(path: Path, low: float, high: float, smooth_n: int = 5) -> pd.DataFrame:
    df = read_otacs_table(path, start_row=9)
    df = _numericize(df).dropna(axis=1, how="all")
    frame_col, time_col, val_col, _ = _pick_cols_by_position_or_name(df, n_expected=3)
    if time_col is None or val_col is None: return pd.DataFrame(columns=["Time", "Value"])
    t = _normalize_time_col_seconds(df[time_col].to_numpy(dtype=np.float64))
    v = df[val_col].to_numpy(dtype=np.float64)
    v = clean_range_to_nan(v, low, high)
    v = sma(v, smooth_n)
    out = pd.DataFrame({"Time": t, "Value": v}).dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)
    return out


def load_peda_raw(path: Path, smooth_n: int = 5) -> pd.DataFrame:
    df = read_otacs_table(path, start_row=9)
    df = _numericize(df).dropna(axis=1, how="all")
    frame_col, time_col, val_col, _ = _pick_cols_by_position_or_name(df, n_expected=3)
    if time_col is None or val_col is None: return pd.DataFrame(columns=["Time", "Value"])
    t = _normalize_time_col_seconds(df[time_col].to_numpy(dtype=np.float64))
    v = df[val_col].to_numpy(dtype=np.float64)
    v = clean_range_to_nan(v, 28, 628)
    v = sma(v, smooth_n)
    out = pd.DataFrame({"Time": t, "Value": v}).dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)
    return out


def load_pp_raw(path: Path, smooth_n: int = 5) -> pd.DataFrame:
    df = read_otacs_table(path, start_row=9)
    df = _numericize(df).dropna(axis=1, how="all")
    frame_col, time_col, val_col, val2_col = _pick_cols_by_position_or_name(df, n_expected=4)
    if time_col is None or val_col is None: return pd.DataFrame(columns=["Time", "pp", "pp_NR"])
    t = _normalize_time_col_seconds(df[time_col].to_numpy(dtype=np.float64))
    pp = df[val_col].to_numpy(dtype=np.float64)
    pp = sma(pp, smooth_n)
    out = pd.DataFrame({"Time": t, "pp": pp})
    if val2_col is not None:
        pp_nr = df[val2_col].to_numpy(dtype=np.float64)
        out["pp_NR"] = sma(pp_nr, smooth_n)
    else:
        out["pp_NR"] = np.nan
    out = out.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)
    return out


def read_video_frames_at_times(video_path: Path, times_s: np.ndarray):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): raise RuntimeError(f"Could not open video: {video_path}")
    frames = []
    for t in times_s:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            frames.append(None)
            continue
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


# -------------------------------------------------
# Visualize
# -------------------------------------------------
def visualize_segment(
        base_path: str,
        subject: str,
        session_code: str,
        start_s: float = 60.0,
        duration_s: float = 180.0,
        n_sample_frames: int = 5,
        use_raw_thermal: bool = True,
        use_structured_videos: bool = True,
        smooth_n: int = 5,
        out_path: str | None = "sim1_segment_overview.png",
):
    base = Path(base_path)
    subject = subject.upper()
    session_code = norm_session_code(session_code)

    raw_dir = base / "RawThermalData" / subject
    structured_dir = base / "StructuredStudyData" / subject
    session_dir = find_structured_session_dir(structured_dir, session_code)
    prefix = infer_prefix_in_session_dir(session_dir, subject)

    avi1 = session_dir / f"{prefix}.avi1.avi"
    avi2 = session_dir / f"{prefix}.avi2.avi"

    sig_paths = {
        "HR": find_session_file(session_dir, prefix, "HR"),
        "BR": find_session_file(session_dir, prefix, "BR"),
        "peda": find_session_file(session_dir, prefix, "peda"),
        "pp": find_session_file(session_dir, prefix, "pp"),
    }

    print(f"Plotting {subject} | Session: {session_code}")
    t0 = float(start_s)
    t1 = t0 + float(duration_s)
    sample_times = t0 + np.linspace(0, duration_s, num=n_sample_frames, endpoint=False)

    signals = {}
    if sig_paths["HR"] is not None:
        try:
            hr = load_hr_br(sig_paths["HR"], low=40, high=120, smooth_n=smooth_n)
            signals["HR"] = clip_df_time(hr, t0, t1)
        except:
            pass
    if sig_paths["BR"] is not None:
        try:
            br = load_hr_br(sig_paths["BR"], low=4, high=40, smooth_n=smooth_n)
            signals["BR"] = clip_df_time(br, t0, t1)
        except:
            pass
    if sig_paths["peda"] is not None:
        try:
            ped = load_peda_raw(sig_paths["peda"], smooth_n=smooth_n)
            signals["peda"] = clip_df_time(ped, t0, t1)
        except:
            pass
    if sig_paths["pp"] is not None:
        try:
            pp = load_pp_raw(sig_paths["pp"], smooth_n=smooth_n)
            signals["pp"] = clip_df_time(pp, t0, t1)
        except:
            pass

    # Load Thermal
    raw_frames = None
    if use_raw_thermal:
        dat_path = raw_dir / f"{subject}-{session_code}.dat"
        inf_path = raw_dir / f"{subject}-{session_code}.inf"
        if dat_path.exists() and inf_path.exists():
            try:
                frames, ts, _, _ = load_thermal_dat(str(dat_path), str(inf_path), use_memmap=True)
                ts = normalize_timestamps_to_seconds(ts)
                if ts.size >= 2 and np.isfinite(ts).any():
                    ts2 = ts.copy()
                    ts2 = ts2[np.isfinite(ts2)]
                    if ts2.size > 0 and np.nanmin(ts2) > 1.0: ts = ts - np.nanmin(ts2)
                    idx_sort = np.argsort(ts)
                    ts_sorted = ts[idx_sort]
                    idx = np.searchsorted(ts_sorted, sample_times, side="left")
                    idx = np.clip(idx, 0, len(ts_sorted) - 1)
                    frame_idx = idx_sort[idx]
                else:
                    frame_idx = (sample_times * 7.5).astype(int)
                frame_idx = np.clip(frame_idx, 0, frames.shape[0] - 1)
                raw_frames = [normalize_for_display(frames[int(i)]) for i in frame_idx]
            except Exception as e:
                print(f"[WARN] Thermal read: {e}")

    # Load Visual
    avi1_frames = [None] * n_sample_frames
    avi2_frames = [None] * n_sample_frames
    if use_structured_videos:
        if avi1.exists():
            try:
                avi1_frames = read_video_frames_at_times(avi1, sample_times)
            except:
                pass
        if avi2.exists():
            try:
                avi2_frames = read_video_frames_at_times(avi2, sample_times)
            except:
                pass

    # -------------------------------------------
    # FANCY PLOTTING
    # -------------------------------------------
    ncols = n_sample_frames
    fig = plt.figure(figsize=(4.0 * ncols, 24))

    # Define Layout with top margin for title
    gs = gridspec.GridSpec(7, ncols,
                           height_ratios=[3, 3, 3, 1.2, 1.2, 1.2, 1.2],
                           hspace=0.5, wspace=0.05,
                           top=0.92, bottom=0.05, left=0.05, right=0.95)

    # 1. Raw Thermal
    for c in range(ncols):
        ax = fig.add_subplot(gs[0, c])
        ax.axis('off')
        if raw_frames is not None:
            ax.imshow(raw_frames[c], cmap="inferno")
            ax.set_title(f"Raw Thermal\nT={sample_times[c]:.1f}s", color="#B22222", fontweight='bold')
        else:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center")

    # 2. Visual Face
    for c in range(ncols):
        ax = fig.add_subplot(gs[1, c])
        ax.axis('off')
        fr = avi1_frames[c]
        if fr is not None:
            ax.imshow(fr)
            ax.set_title(f"Visual Face", color="#333333")
        else:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center")

    # 3. Perinasal ROI
    for c in range(ncols):
        ax = fig.add_subplot(gs[2, c])
        ax.axis('off')
        fr = avi2_frames[c]
        if fr is not None:
            ax.imshow(fr)
            ax.set_title(f"Perinasal ROI", color="#006400")
        else:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center")

    # 4. Signals
    sig_configs = [
        ("HR", "Heart Rate (BPM)", "#d62728"),  # Red
        ("BR", "Breathing Rate (BPM)", "#1f77b4"),  # Blue
        ("peda", "Palm EDA (kΩ)", "#9467bd"),  # Purple
        ("pp", "Perinasal EDA (°C²)", "#2ca02c"),  # Green
    ]

    for r_idx, (key, label, color) in enumerate(sig_configs):
        ax = fig.add_subplot(gs[3 + r_idx, :])
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(t0, t1)
        ax.set_ylabel(label, color=color, fontweight='bold')
        ax.set_facecolor('#fafafa')

        for spine in ax.spines.values(): spine.set_edgecolor('#cccccc')

        if key in signals and len(signals[key]) > 0:
            df = signals[key]
            if key == "pp":
                ax.plot(df["Time"], df["pp"], label="Raw PP", color=color, linewidth=1.5, alpha=0.6)
                if "pp_NR" in df.columns and np.isfinite(df["pp_NR"].to_numpy()).any():
                    ax.plot(df["Time"], df["pp_NR"], label="Noise Reduced PP", color='#32CD32', linewidth=2.5)
                ax.legend(loc="upper right", frameon=True, framealpha=0.9)
            else:
                ax.plot(df["Time"], df["Value"], label=label, color=color, linewidth=2.5)
                ax.legend(loc="upper right", frameon=True, framealpha=0.9)
        else:
            ax.text(0.5, 0.5, f"{key} Unavailable", transform=ax.transAxes, ha='center', color='gray')

        for t_mark in sample_times:
            ax.axvline(t_mark, color='black', linewidth=1.5, linestyle=':', alpha=0.5)

        if r_idx == 3:
            ax.set_xlabel("Session Time (seconds)", fontweight='bold')
        else:
            ax.set_xticklabels([])

    fig.suptitle(f"Multimodal Data Overview: Subject {subject} | {session_code}", fontweight='bold')

    if out_path is not None:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"[OK] Saved fancy plot to: {out_path}")

    plt.show()


if __name__ == "__main__":
    BASE_PATH = "/media/arritmic/HI-NRI-TINO-002/DATABASES/MULTIMODAL/Affective_States/SIMULATOR_STUDY_I"
    visualize_segment(
        base_path=BASE_PATH,
        subject="T003",
        session_code="ND",
        start_s=20.0,
        duration_s=240.0,
        n_sample_frames=5,
        use_raw_thermal=True,
        use_structured_videos=True,
        smooth_n=5,
        out_path="sim1_T003_ND_fancy_overview.png",
    )
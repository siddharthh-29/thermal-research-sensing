"""
mmslab_sim1_06_sync_gt_from_roi_csv.py

Synchronizes a pre-existing ROI temperature CSV with the contact ground-truth
signals (HR, BR, PP, PEDA) from StructuredStudyData.

Use this when ROI traces have already been extracted to CSV by an external tool
and only the GT synchronization step needs to be run.

For the full pipeline (extraction + sync in one pass), use:
  mmslab_sim1_06_extract_roi_sync_gt.py

Configuration:
  Edit SUBJECT, TASK, ROI_CSV_EXACT_NAME, and BASE_PATH below.
"""

import re
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Hardcoded configuration
# -----------------------------
BASE_PATH = "/media/arritmic/HI-NRI-TINO-002/DATABASES/MULTIMODAL/Affective_States/SIMULATOR_STUDY_I"

SUBJECT = "T003"
TASK = "ND"  # BL, PD, RD, ND, CD, ED, MD, FDN, FDL

OUT_FPS = 7.5

# Optional: limit duration for certain tasks (seconds)
# Use None to keep full overlap range.
TASK_DURATION_LIMIT_S = {
    "ND": 640.0,
    "CD": 640.0,
    "ED": 640.0,
    # "BL": 180.0,
    "PD": 180.0,
}

# If True: shift all time axes so that t=0 is the first valid time for each signal before syncing.
SHIFT_TO_ZERO = True

# If your ROI CSV naming is stable, set this.
# Otherwise leave as None and the script will search for it.
ROI_CSV_EXACT_NAME = "T003-004.roi_raw.csv"  # example: "T003-ND-roi_raw.csv"


# -----------------------------
# StructuredStudyData helpers
# -----------------------------
def norm_session_code(code: str) -> str:
    code = code.strip().upper()
    return "BL" if code == "B" else code


def find_structured_session_dir(structured_subject_dir: Path, session_code: str) -> Path:
    session_code = norm_session_code(session_code)
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


def find_session_file(session_dir: Path, prefix: str, ext: str) -> Path | None:
    target = f"{prefix}.{ext}".lower()
    for p in session_dir.iterdir():
        if p.is_file() and p.name.lower().startswith(target):
            return p
    return None


def find_roi_csv(session_dir: Path, subject: str, task: str) -> Path:
    subject = subject.upper()
    task = task.upper()

    if ROI_CSV_EXACT_NAME is not None:
        p = session_dir / ROI_CSV_EXACT_NAME
        if p.exists():
            return p

    # Search heuristics: any csv with subject+task and "roi" in its name
    candidates = []
    for p in session_dir.glob("*.csv"):
        name = p.name.lower()
        if subject.lower() in name and task.lower() in name and ("roi" in name or "signal" in name):
            candidates.append(p)

    # Fallback: any csv with subject+task
    if not candidates:
        for p in session_dir.glob("*.csv"):
            name = p.name.lower()
            if subject.lower() in name and task.lower() in name:
                candidates.append(p)

    if not candidates:
        raise FileNotFoundError(
            f"No ROI CSV found in {session_dir}. "
            f"Set ROI_CSV_EXACT_NAME or ensure a CSV exists with subject/task in the name."
        )

    # Prefer shortest name (usually the main file) and then alphabetical
    candidates = sorted(candidates, key=lambda x: (len(x.name), x.name))
    return candidates[0]


# -----------------------------
# OTACS Excel reader (custom extensions .HR/.BR/.peda/.pp)
# -----------------------------
def read_otacs_table(path: Path, start_row: int = 9) -> pd.DataFrame:
    """
    Reads OTACS measurement files that are Excel containers but have custom extensions.
    start_row=9 matches the authors R code (startRow=9).
    """
    b = path.read_bytes()

    OLE_SIG = b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1"  # .xls container
    ZIP_SIG = b"PK\x03\x04"  # .xlsx container

    bio = BytesIO(b)

    if b.startswith(OLE_SIG):
        # Needs: pip install xlrd==2.0.1
        df = pd.read_excel(bio, sheet_name=0, engine="xlrd", skiprows=start_row - 1)
    elif b.startswith(ZIP_SIG):
        df = pd.read_excel(bio, sheet_name=0, engine="openpyxl", skiprows=start_row - 1)
    else:
        # Rare fallback
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
    # if time is ms
    if np.nanmax(finite) > 1e4:
        return t / 1000.0
    return t


def _pick_cols_by_position_or_name(df: pd.DataFrame, n_expected: int):
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


def load_3col_signal(path: Path) -> pd.DataFrame:
    df = read_otacs_table(path, start_row=9)
    df = _numericize(df).dropna(axis=1, how="all")

    _, time_col, val_col, _ = _pick_cols_by_position_or_name(df, n_expected=3)
    if time_col is None or val_col is None:
        return pd.DataFrame(columns=["Time", "Value"])

    t = _normalize_time_seconds(df[time_col].to_numpy(dtype=np.float64))
    y = df[val_col].to_numpy(dtype=np.float64)

    out = pd.DataFrame({"Time": t, "Value": y})
    out = out.dropna(subset=["Time"]).sort_values("Time").drop_duplicates(subset=["Time"]).reset_index(drop=True)
    return out


def load_pp_signal(path: Path) -> pd.DataFrame:
    df = read_otacs_table(path, start_row=9)
    df = _numericize(df).dropna(axis=1, how="all")

    _, time_col, val_col, val2_col = _pick_cols_by_position_or_name(df, n_expected=4)
    if time_col is None or val_col is None:
        return pd.DataFrame(columns=["Time", "pp", "pp_NR"])

    t = _normalize_time_seconds(df[time_col].to_numpy(dtype=np.float64))
    pp = df[val_col].to_numpy(dtype=np.float64)

    out = pd.DataFrame({"Time": t, "pp": pp})

    if val2_col is not None:
        out["pp_NR"] = df[val2_col].to_numpy(dtype=np.float64)
    else:
        out["pp_NR"] = np.nan

    out = out.dropna(subset=["Time"]).sort_values("Time").drop_duplicates(subset=["Time"]).reset_index(drop=True)
    return out


# -----------------------------
# Sample rate estimation
# -----------------------------
def estimate_fs_from_time(t: np.ndarray) -> float | None:
    t = np.asarray(t, dtype=np.float64)
    t = t[np.isfinite(t)]
    if t.size < 3:
        return None
    t = np.sort(np.unique(t))
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size < 2:
        return None
    med = float(np.median(dt))
    if med <= 0:
        return None
    return 1.0 / med


# -----------------------------
# Resampling to a common grid
# -----------------------------
def shift_to_zero_df(df: pd.DataFrame, time_col: str = "Time") -> pd.DataFrame:
    if df is None or len(df) == 0 or time_col not in df.columns:
        return df
    out = df.copy()
    t = out[time_col].to_numpy(dtype=np.float64)
    finite = t[np.isfinite(t)]
    if finite.size == 0:
        return out
    out[time_col] = t - float(np.min(finite))
    return out


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

    # Remove duplicate times
    t_unique, unique_idx = np.unique(t, return_index=True)
    y_unique = y[unique_idx]

    return np.interp(t_grid, t_unique, y_unique, left=np.nan, right=np.nan)


def resample_df_to_grid(df: pd.DataFrame, t_grid: np.ndarray, time_col: str, out_prefix: str) -> dict:
    """
    Returns a dict of {colname: resampled_array}.
    """
    out = {}
    if df is None or len(df) == 0:
        return out

    t_in = df[time_col].to_numpy(dtype=np.float64)

    for c in df.columns:
        if c == time_col:
            continue
        y_in = df[c].to_numpy(dtype=np.float64)
        out[f"{out_prefix}{c}"] = interp_1d_to_grid(t_in, y_in, t_grid)

    return out


# -----------------------------
# Main
# -----------------------------
def main():
    base = Path(BASE_PATH)
    subject = SUBJECT.upper()
    task = norm_session_code(TASK)

    structured_subject_dir = base / "StructuredStudyData" / subject
    session_dir = find_structured_session_dir(structured_subject_dir, task)
    prefix = infer_prefix_in_session_dir(session_dir, subject)  # Txxx-zzz

    print(f"[INFO] subject={subject} task={task}")
    print(f"[INFO] session_dir={session_dir}")
    print(f"[INFO] prefix={prefix}")

    # Load ROI CSV
    roi_csv = find_roi_csv(session_dir, subject, task)
    roi_df = pd.read_csv(roi_csv)
    print(f"[INFO] ROI CSV: {roi_csv.name}  cols={list(roi_df.columns)}  n={len(roi_df)}")

    # Expect a Time column. If your ROI CSV uses a different name, add mapping here.
    if "Time" not in roi_df.columns:
        # common alternatives
        for alt in ["time", "t", "Time_s", "timestamp", "Timestamp"]:
            if alt in roi_df.columns:
                roi_df = roi_df.rename(columns={alt: "Time"})
                break
    if "Time" not in roi_df.columns:
        raise ValueError(f"ROI CSV does not contain a Time column. Columns: {list(roi_df.columns)}")

    roi_df["Time"] = pd.to_numeric(roi_df["Time"], errors="coerce")
    roi_df = roi_df.dropna(subset=["Time"]).sort_values("Time").drop_duplicates(subset=["Time"]).reset_index(drop=True)

    # Load GT signals
    hr_path = find_session_file(session_dir, prefix, "HR")
    br_path = find_session_file(session_dir, prefix, "BR")
    peda_path = find_session_file(session_dir, prefix, "peda")
    pp_path = find_session_file(session_dir, prefix, "pp")

    gt = {}

    if hr_path and hr_path.exists():
        gt["HR"] = load_3col_signal(hr_path).rename(columns={"Value": "HR"})
    if br_path and br_path.exists():
        gt["BR"] = load_3col_signal(br_path).rename(columns={"Value": "BR"})
    if peda_path and peda_path.exists():
        gt["peda"] = load_3col_signal(peda_path).rename(columns={"Value": "peda"})
    if pp_path and pp_path.exists():
        gt["pp"] = load_pp_signal(pp_path)

    # Optional shift-to-zero per signal
    if SHIFT_TO_ZERO:
        roi_df = shift_to_zero_df(roi_df, "Time")
        for k in list(gt.keys()):
            gt[k] = shift_to_zero_df(gt[k], "Time")

    # Print estimated sampling rates
    roi_fs = estimate_fs_from_time(roi_df["Time"].to_numpy())
    print(f"[FS] ROI (from Time): {roi_fs:.4f} Hz" if roi_fs else "[FS] ROI (from Time): n/a")

    for name, df in gt.items():
        fs = estimate_fs_from_time(df["Time"].to_numpy())
        print(f"[FS] {name}: {fs:.4f} Hz" if fs else f"[FS] {name}: n/a")

    # Determine common time interval (intersection)
    tmins = [float(np.nanmin(roi_df["Time"].to_numpy()))]
    tmaxs = [float(np.nanmax(roi_df["Time"].to_numpy()))]
    for df in gt.values():
        tmins.append(float(np.nanmin(df["Time"].to_numpy())))
        tmaxs.append(float(np.nanmax(df["Time"].to_numpy())))

    t0 = max(tmins)
    t1 = min(tmaxs)

    # Optional duration limit by task (useful to drop P5LDj range if present in ROI)
    limit = TASK_DURATION_LIMIT_S.get(task, None)
    if limit is not None:
        t1 = min(t1, t0 + float(limit))

    if not np.isfinite(t0) or not np.isfinite(t1) or (t1 <= t0):
        raise RuntimeError(f"Invalid overlap time range: t0={t0} t1={t1}")

    # Build uniform grid at 7.5 Hz
    dt = 1.0 / float(OUT_FPS)
    n = int(np.floor((t1 - t0) / dt)) + 1
    t_grid = t0 + np.arange(n, dtype=np.float64) * dt

    print(f"[SYNC] overlap [{t0:.3f}, {t1:.3f}] s  -> grid n={len(t_grid)} at {OUT_FPS} Hz")

    # Resample ROI
    sync_cols = {"Time": t_grid}

    # ROI columns: keep all non-Time columns
    for c in roi_df.columns:
        if c == "Time":
            continue
        sync_cols[f"roi_{c}"] = interp_1d_to_grid(
            roi_df["Time"].to_numpy(dtype=np.float64),
            pd.to_numeric(roi_df[c], errors="coerce").to_numpy(dtype=np.float64),
            t_grid,
        )

    # Resample GT
    if "HR" in gt:
        sync_cols["HR"] = interp_1d_to_grid(gt["HR"]["Time"].to_numpy(), gt["HR"]["HR"].to_numpy(), t_grid)
    if "BR" in gt:
        sync_cols["BR"] = interp_1d_to_grid(gt["BR"]["Time"].to_numpy(), gt["BR"]["BR"].to_numpy(), t_grid)
    if "peda" in gt:
        sync_cols["peda"] = interp_1d_to_grid(gt["peda"]["Time"].to_numpy(), gt["peda"]["peda"].to_numpy(), t_grid)
    if "pp" in gt:
        sync_cols["pp"] = interp_1d_to_grid(gt["pp"]["Time"].to_numpy(), gt["pp"]["pp"].to_numpy(), t_grid)
        if "pp_NR" in gt["pp"].columns:
            sync_cols["pp_NR"] = interp_1d_to_grid(gt["pp"]["Time"].to_numpy(), gt["pp"]["pp_NR"].to_numpy(), t_grid)

    sync_df = pd.DataFrame(sync_cols)

    # Save synchronized CSV
    out_csv = session_dir / f"{subject}-{task}-synced_{OUT_FPS:.1f}Hz.csv"
    sync_df.to_csv(out_csv, index=False)
    print(f"[OK] saved: {out_csv}")

    # -----------------------------
    # Visualization
    # -----------------------------
    # Keep plots readable: ROI lines in one axis, GT in others.
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(nrows=3, ncols=1, hspace=0.25)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.grid(True, alpha=0.3)
    ax0.set_title(f"{subject} {task} | ROI signals (resampled to {OUT_FPS} Hz)")
    ax0.set_ylabel("Raw thermal (a.u.)")

    roi_cols = [c for c in sync_df.columns if c.startswith("roi_")]
    for c in roi_cols:
        ax0.plot(sync_df["Time"], sync_df[c], label=c, linewidth=1.0)
    ax0.legend(loc="upper right", ncol=2, fontsize=8)

    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("HR and BR (GT, resampled)")
    ax1.set_ylabel("bpm")
    if "HR" in sync_df.columns:
        ax1.plot(sync_df["Time"], sync_df["HR"], label="HR", linewidth=1.0)
    if "BR" in sync_df.columns:
        ax1.plot(sync_df["Time"], sync_df["BR"], label="BR", linewidth=1.0)
    ax1.legend(loc="upper right")

    ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)
    ax2.grid(True, alpha=0.3)
    ax2.set_title("EDA signals (GT, resampled)")
    ax2.set_xlabel("Time (s)")
    if "peda" in sync_df.columns:
        ax2.plot(sync_df["Time"], sync_df["peda"], label="peda", linewidth=1.0)
    if "pp" in sync_df.columns:
        ax2.plot(sync_df["Time"], sync_df["pp"], label="pp", linewidth=1.0)
    if "pp_NR" in sync_df.columns:
        ax2.plot(sync_df["Time"], sync_df["pp_NR"], label="pp_NR", linewidth=1.0)
    ax2.legend(loc="upper right")

    plt.show()


if __name__ == "__main__":
    main()

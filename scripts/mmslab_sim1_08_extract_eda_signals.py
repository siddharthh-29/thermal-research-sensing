"""
mmslab_sim1_07_extract_eda_signals.py

Extracts EDA-like (electrodermal activity) signals from thermal ROI time series
and evaluates them against contact ground-truth signals (palm EDA and perinasal
perspiration index).

Physiological basis:
  Sympathetic activation → eccrine sweat gland activity → local skin cooling or
  warming (polarity depends on the competing vasoconstriction/evaporation balance).
  This manifests as a SLOW TREND (<0.1 Hz) in the ROI temperature signal; the
  perinasal region and cheeks are the most informative sites.

Pipeline:
  1. Load the synchronized 30 Hz CSV (ROI traces + GT PEDA + GT PP).
  2. Extract slow trend from each ROI using the configured methods.
  3. Z-score normalize all signals for comparison.
  4. Compute agreement metrics: PCC_abs, Spearman ρ, max cross-correlation,
     trend agreement, and optimal lag.
  5. Visualize extracted EDA trends vs ground-truth.

Configuration:
  Edit the CONFIGURATION block below (SUBJECT, TASK, BASE_PATH, EDA_ROIS,
  EDA_METHODS, etc.) before running.
"""

import re
from io import BytesIO
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import pearsonr, spearmanr
from scipy.signal import savgol_filter

# =============================================================================
# Configuration
# =============================================================================
BASE_PATH = Path("/media/arritmic/HI-NRI-TINO-002/DATABASES/MULTIMODAL/Affective_States/SIMULATOR_STUDY_I")
STRUCT_DIR = BASE_PATH / "StructuredStudyData"

SUBJECT = "T003"
TASK = "CD"  # BL, PD, ND, CD, ED

FS = 30.0  # sampling rate of the synchronized CSV

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

ROI_NAMES = ["nose", "eye_l", "eye_r", "cheek_l", "cheek_r", "forehead"]

# =============================================================================
# EDA extraction configuration
# =============================================================================
# Which ROIs to extract EDA-like signals from
EDA_ROIS = {
    "nose":       "nose",                      # perinasal → compare with PP
    "eyes_avg":   ["eye_l", "eye_r"],          # periorbital average → compare with PP
    "eye_l":      "eye_l",                     # left eye only
    "eye_r":      "eye_r",                     # right eye only
    "cheeks_avg": ["cheek_l", "cheek_r"],      # cheeks average → compare with PEDA
    "forehead":   "forehead",                  # forehead → compare with PEDA
}

# Which extraction methods to apply (all will be computed and compared)
# Each method extracts the slow trend differently
EDA_METHODS = [
    "lowpass_butterworth",    # Classic lowpass filter
    "lowpass_bessel",         # Bessel filter (maximally flat group delay — less ringing)
    "savgol",                 # Savitzky-Golay smoothing (polynomial local fit)
    "moving_average",         # Simple moving average
    "exponential_ma",         # Exponential moving average (causal)
    "median_filter",          # Median filter (robust to spikes)
    "envelope_hilbert",       # Hilbert envelope (amplitude modulation)
    "wavelet_approx",         # Wavelet approximation (lowest frequency band)
]

# Default method for main comparison plots
EDA_DEFAULT_METHOD = "lowpass_butterworth"

# Lowpass cutoff frequency (Hz) — defines what "slow trend" means
# EDA changes happen over 1–30+ seconds → < 0.05–0.1 Hz
EDA_LOWPASS_CUTOFF = 0.05  # Hz (changes slower than 20 seconds)
EDA_LOWPASS_ORDER = 3

# Moving average / median filter window (seconds)
EDA_SMOOTH_WINDOW_S = 30.0  # 30 second smoothing window

# Savitzky-Golay parameters
EDA_SG_WINDOW_S = 30.0  # window in seconds
EDA_SG_POLYORDER = 3

# Downsample output for plotting/metrics (EDA is very slow, 30 Hz is overkill)
EDA_OUTPUT_FS = 1.0  # 1 Hz is plenty for EDA

# Figures
SAVE_FIGS = True
FIG_OUT_DIR = BASE_PATH / "AnalysisOutputs" / "eda_plots" / SUBJECT
FIG_OUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# File discovery (reused from research signals script)
# =============================================================================
def find_structured_session_dir(structured_subject_dir: Path, session_code: str) -> Path:
    session_code = session_code.strip().upper()
    for p in structured_subject_dir.iterdir():
        if not p.is_dir():
            continue
        parts = p.name.split()
        if len(parts) >= 2 and parts[-1].upper() == session_code:
            return p
    raise FileNotFoundError(
        f"Could not find session folder ending with {session_code} in {structured_subject_dir}")


def infer_prefix_in_session_dir(session_dir: Path, subject: str) -> str:
    subject = subject.upper()
    pat = re.compile(rf"^{re.escape(subject)}-\d{{3}}\.", re.IGNORECASE)
    for p in session_dir.iterdir():
        if p.is_file() and pat.match(p.name):
            return p.name.split(".", 1)[0]
    raise FileNotFoundError(f"Could not infer Txxx-zzz prefix in: {session_dir}")


def find_sync_csv(session_dir: Path, subject: str, task: str, fs: float) -> Path:
    candidates = [
        session_dir / f"{subject}-{task}-roi_peda_pp_{fs:.0f}Hz.csv",
        session_dir / f"{subject}-{task}-roi_peda_pp_{fs:.1f}Hz.csv",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    pat = re.compile(
        rf"^{re.escape(subject)}-{re.escape(task)}-roi_peda_pp_.*Hz\.csv$", re.IGNORECASE)
    for p in session_dir.iterdir():
        if p.is_file() and pat.match(p.name):
            return p
    raise FileNotFoundError(
        f"Could not find synchronized ROI/PP/PEDA CSV for {subject}-{task} in {session_dir}")


# =============================================================================
# Signal utilities
# =============================================================================
def nan_interp_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    idx = np.arange(x.size)
    good = np.isfinite(x)
    if not np.any(good):
        return x
    y = x.copy()
    y[~good] = np.interp(idx[~good], idx[good], x[good])
    return y


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    m = np.nanmean(x)
    s = np.nanstd(x)
    if not np.isfinite(s) or s < 1e-12:
        return x - m if np.isfinite(m) else x * 0.0
    return (x - m) / s


def downsample_to_fs(t: np.ndarray, x: np.ndarray, target_fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Downsample by picking every N-th sample (after the signal is already lowpassed)."""
    current_fs = 1.0 / np.median(np.diff(t[np.isfinite(t)][:100]))
    step = max(1, int(round(current_fs / target_fs)))
    return t[::step], x[::step]


# =============================================================================
# EDA trend extraction methods
# =============================================================================

def extract_lowpass_butterworth(x: np.ndarray, fs: float,
                                cutoff: float = 0.05, order: int = 3) -> np.ndarray:
    """
    Butterworth lowpass filter.
    Standard approach: removes all fast components (pulse, respiration, noise),
    leaving only the slow sympathetic trend.

    Good for: clean separation of trend from oscillatory components.
    Caveat: can ring at sharp transitions (onset of sweating response).
    """
    x = nan_interp_1d(x)
    # Ensure cutoff is below Nyquist
    nyq = fs / 2.0
    if cutoff >= nyq:
        cutoff = nyq * 0.9
    sos = signal.butter(order, cutoff, btype='low', fs=fs, output='sos')
    return signal.sosfiltfilt(sos, x)


def extract_lowpass_bessel(x: np.ndarray, fs: float,
                           cutoff: float = 0.05, order: int = 3) -> np.ndarray:
    """
    Bessel lowpass filter.
    Maximally flat group delay → preserves waveform shape better than Butterworth.
    Less ringing on step-like EDA responses (sudden onset of perspiration).

    Preferred when temporal fidelity of the EDA waveform matters more
    than sharp frequency cutoff.
    """
    x = nan_interp_1d(x)
    nyq = fs / 2.0
    if cutoff >= nyq:
        cutoff = nyq * 0.9
    sos = signal.bessel(order, cutoff, btype='low', fs=fs, output='sos', norm='phase')
    return signal.sosfiltfilt(sos, x)


def extract_savgol(x: np.ndarray, fs: float,
                   window_s: float = 30.0, polyorder: int = 3) -> np.ndarray:
    """
    Savitzky-Golay smoothing filter.
    Fits a polynomial to each local window → extracts the smooth trend.

    Advantages: preserves sharp features (EDA peaks) better than lowpass filters.
    The polynomial order controls how much curvature is allowed in the trend.

    polyorder=2: very smooth (almost moving average)
    polyorder=3: allows moderate curvature (good default)
    polyorder=5: follows sharper features
    """
    x = nan_interp_1d(x)
    win = int(round(window_s * fs))
    if win % 2 == 0:
        win += 1
    win = max(win, polyorder + 2)
    return savgol_filter(x, window_length=win, polyorder=polyorder)


def extract_moving_average(x: np.ndarray, fs: float,
                           window_s: float = 30.0) -> np.ndarray:
    """
    Simple moving average (boxcar convolution).
    The simplest trend extractor. Equivalent to a sinc lowpass in frequency domain.

    window_s=30 means: each output sample is the average of the surrounding ±15 seconds.
    """
    x = nan_interp_1d(x)
    win = int(round(window_s * fs))
    if win < 3:
        win = 3
    kernel = np.ones(win) / win
    # Use 'same' mode and pad edges
    padded = np.pad(x, (win // 2, win // 2), mode='edge')
    return np.convolve(padded, kernel, mode='valid')[:len(x)]


def extract_exponential_ma(x: np.ndarray, fs: float,
                           window_s: float = 30.0) -> np.ndarray:
    """
    Exponential moving average (EMA).
    Causal filter: each output depends only on past values.
    Useful if you want a real-time-applicable trend estimator.

    alpha = 2 / (span + 1), where span ≈ window_s * fs.
    Smaller alpha = smoother.
    """
    x = nan_interp_1d(x)
    span = int(round(window_s * fs))
    alpha = 2.0 / (span + 1)
    out = np.zeros_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    # Apply backwards too for zero-phase (like filtfilt)
    out2 = np.zeros_like(out)
    out2[-1] = out[-1]
    for i in range(len(out) - 2, -1, -1):
        out2[i] = alpha * out[i] + (1 - alpha) * out2[i + 1]
    return 0.5 * (out + out2)


def extract_median_filter(x: np.ndarray, fs: float,
                          window_s: float = 30.0) -> np.ndarray:
    """
    Median filter followed by light smoothing.
    Extremely robust to spikes and outliers (e.g., face detection failures
    that cause sudden jumps in the ROI signal).

    The median filter removes impulsive noise while preserving step edges
    (important for EDA onset detection).
    """
    x = nan_interp_1d(x)
    win = int(round(window_s * fs))
    if win % 2 == 0:
        win += 1
    win = max(win, 3)
    med = signal.medfilt(x, kernel_size=win)
    # Light additional smoothing to remove median filter artifacts
    sg_win = max(int(5 * fs), 5)
    if sg_win % 2 == 0:
        sg_win += 1
    return savgol_filter(med, window_length=sg_win, polyorder=2)


def extract_envelope_hilbert(x: np.ndarray, fs: float,
                             cutoff: float = 0.05) -> np.ndarray:
    """
    Hilbert envelope: extracts the amplitude modulation of the signal.

    Steps:
      1. Bandpass to the physiological range (remove DC drift and high-freq noise)
      2. Compute analytic signal via Hilbert transform
      3. Take the envelope (amplitude) → this captures how the signal energy
         changes over time, which correlates with sympathetic activation

    Different from lowpass: this captures amplitude changes in the 0.1-3 Hz band,
    not just the DC trend. Can detect sympathetic activation that modulates
    pulse/respiration amplitude without changing the baseline temperature.
    """
    x = nan_interp_1d(x)
    # First bandpass to physiological range
    sos_bp = signal.butter(3, [0.05, 3.0], btype='bandpass', fs=fs, output='sos')
    x_bp = signal.sosfiltfilt(sos_bp, x)

    # Hilbert envelope
    analytic = signal.hilbert(x_bp)
    envelope = np.abs(analytic)

    # Lowpass the envelope to get the slow modulation
    sos_lp = signal.butter(3, cutoff, btype='low', fs=fs, output='sos')
    return signal.sosfiltfilt(sos_lp, envelope)


def extract_wavelet_approx(x: np.ndarray, fs: float, level: int = None) -> np.ndarray:
    """
    Wavelet decomposition: use the approximation coefficients at the deepest level.
    This is a multi-resolution approach that cleanly separates the slow trend.

    Uses PyWavelets if available, falls back to iterative lowpass halfband otherwise.

    At 30 Hz, level=9 gives frequency band 0–0.029 Hz (periods > 34s) — ideal for EDA.
    Level=8 gives 0–0.059 Hz (periods > 17s).
    """
    x = nan_interp_1d(x)

    try:
        import pywt
        if level is None:
            # Auto: pick level so approximation band covers 0 to ~0.05 Hz
            # At each level, bandwidth halves: level k → 0 to fs/(2^(k+1))
            # We want fs/(2^(k+1)) ≈ 0.05 → k = log2(fs/0.1) - 1
            level = max(1, int(np.log2(fs / 0.1)))
        level = min(level, pywt.dwt_max_level(len(x), 'db4'))

        coeffs = pywt.wavedec(x, 'db4', level=level)
        # Zero out all detail coefficients, keep only approximation
        for i in range(1, len(coeffs)):
            coeffs[i] = np.zeros_like(coeffs[i])
        return pywt.waverec(coeffs, 'db4')[:len(x)]

    except ImportError:
        print("  [WARN] pywt not installed, falling back to Butterworth lowpass")
        return extract_lowpass_butterworth(x, fs, cutoff=0.03, order=4)


# Dispatcher
_EDA_METHODS = {
    "lowpass_butterworth": extract_lowpass_butterworth,
    "lowpass_bessel":      extract_lowpass_bessel,
    "savgol":              extract_savgol,
    "moving_average":      extract_moving_average,
    "exponential_ma":      extract_exponential_ma,
    "median_filter":       extract_median_filter,
    "envelope_hilbert":    extract_envelope_hilbert,
    "wavelet_approx":      extract_wavelet_approx,
}


def extract_eda_trend(x: np.ndarray, fs: float, method: str = "lowpass_butterworth",
                      **kwargs) -> np.ndarray:
    """
    Extract EDA-like slow trend from a thermal ROI signal.

    Args:
        x: raw ROI signal (1D, at fs Hz)
        fs: sampling rate
        method: extraction method name
        **kwargs: passed to the specific method

    Returns:
        1D trend signal (same length as x)
    """
    fn = _EDA_METHODS.get(method)
    if fn is None:
        raise ValueError(f"Unknown EDA method: '{method}'. Available: {list(_EDA_METHODS.keys())}")

    # Set default kwargs per method
    defaults = {
        "lowpass_butterworth": {"cutoff": EDA_LOWPASS_CUTOFF, "order": EDA_LOWPASS_ORDER},
        "lowpass_bessel":      {"cutoff": EDA_LOWPASS_CUTOFF, "order": EDA_LOWPASS_ORDER},
        "savgol":              {"window_s": EDA_SG_WINDOW_S, "polyorder": EDA_SG_POLYORDER},
        "moving_average":      {"window_s": EDA_SMOOTH_WINDOW_S},
        "exponential_ma":      {"window_s": EDA_SMOOTH_WINDOW_S},
        "median_filter":       {"window_s": EDA_SMOOTH_WINDOW_S},
        "envelope_hilbert":    {"cutoff": EDA_LOWPASS_CUTOFF},
        "wavelet_approx":      {},
    }
    params = {**defaults.get(method, {}), **kwargs}

    return fn(x, fs, **params)


def build_roi_signal(roi_raw: Dict[str, np.ndarray], roi_spec) -> np.ndarray:
    """
    Build a single signal from an ROI spec.
    roi_spec can be:
      - a string: single ROI name
      - a list: average of multiple ROIs
    """
    if isinstance(roi_spec, str):
        return roi_raw[roi_spec].copy()
    elif isinstance(roi_spec, list):
        signals = [roi_raw[r] for r in roi_spec]
        return np.mean(signals, axis=0)
    else:
        raise ValueError(f"Invalid ROI spec: {roi_spec}")


# =============================================================================
# Similarity metrics for EDA comparison
# =============================================================================
def calc_eda_similarity(estimated: np.ndarray, reference: np.ndarray,
                        fs: float = 1.0) -> Dict[str, float]:
    """
    Compute similarity metrics between estimated thermal EDA and GT EDA.

    These are different from HR/BR error metrics because:
      - EDA signals have different units (thermal DN vs kΩ or °C²)
      - We care about correlation (shape similarity), not absolute error
      - Temporal alignment matters (lag between sympathetic response
        reaching skin surface vs sweat gland activation)

    Returns dict with:
      PCC:              Pearson correlation (linear similarity)
      Spearman:         Spearman rank correlation (monotonic similarity)
      PCC_abs:          PCC of absolute values (ignore sign flips)
      max_xcorr:        Maximum cross-correlation (best alignment)
      lag_at_max_xcorr: Time lag (seconds) at maximum cross-correlation
      RMSE_zscore:      RMSE after z-score normalization
      trend_agreement:  % of time both signals move in the same direction
    """
    est = np.asarray(estimated, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)

    valid = np.isfinite(est) & np.isfinite(ref)
    n_valid = int(np.sum(valid))

    empty = {
        "n_valid": 0, "PCC": np.nan, "Spearman": np.nan,
        "PCC_abs": np.nan, "max_xcorr": np.nan, "lag_at_max_s": np.nan,
        "RMSE_z": np.nan, "trend_agree_pct": np.nan,
    }
    if n_valid < 10:
        return empty

    e = est[valid]
    r = ref[valid]

    # --- Pearson correlation ---
    pcc, pcc_p = pearsonr(e, r)

    # --- Spearman rank correlation ---
    spear, spear_p = spearmanr(e, r)

    # --- PCC of absolute values (handles sign flip) ---
    # Thermal EDA may be inverted relative to GT (cooling = increased perspiration)
    pcc_abs = max(abs(pcc), abs(float(np.corrcoef(e, -r)[0, 1])))

    # --- Z-score RMSE ---
    ez = zscore(e)
    rz = zscore(r)
    rmse_z = float(np.sqrt(np.mean((ez - rz) ** 2)))
    # Also check inverted
    rmse_z_inv = float(np.sqrt(np.mean((-ez - rz) ** 2)))
    rmse_z = min(rmse_z, rmse_z_inv)

    # --- Cross-correlation (find best lag) ---
    max_lag_samples = min(int(120 * fs), n_valid // 3)  # search up to ±120s
    ez_centered = ez - np.mean(ez)
    rz_centered = rz - np.mean(rz)

    if max_lag_samples > 0 and n_valid > 20:
        xcorr = np.correlate(ez_centered, rz_centered, mode='full')
        # Normalize
        norm = np.sqrt(np.sum(ez_centered ** 2) * np.sum(rz_centered ** 2))
        if norm > 1e-12:
            xcorr /= norm

        mid = n_valid - 1
        lo = max(0, mid - max_lag_samples)
        hi = min(len(xcorr), mid + max_lag_samples + 1)

        xcorr_window = xcorr[lo:hi]
        best_idx = np.argmax(np.abs(xcorr_window))
        max_xc = float(xcorr_window[best_idx])
        lag_samples = best_idx - (mid - lo)
        lag_s = float(lag_samples / fs)
    else:
        max_xc = pcc
        lag_s = 0.0

    # --- Trend agreement (derivative sign) ---
    de = np.diff(e)
    dr = np.diff(r)
    # Both increasing or both decreasing
    agree = np.sum(np.sign(de) == np.sign(dr))
    trend_agree = 100.0 * agree / max(1, len(de))

    return {
        "n_valid": n_valid,
        "PCC": round(float(pcc), 4),
        "Spearman": round(float(spear), 4),
        "PCC_abs": round(float(pcc_abs), 4),
        "max_xcorr": round(float(max_xc), 4),
        "lag_at_max_s": round(lag_s, 1),
        "RMSE_z": round(rmse_z, 4),
        "trend_agree_pct": round(trend_agree, 1),
    }


def print_eda_metrics(label: str, metrics: Dict[str, float]):
    print(f"\n  === {label} ===")
    for k, v in metrics.items():
        print(f"    {k:18s}: {v}")


# =============================================================================
# Main
# =============================================================================
def main():
    subject = SUBJECT.upper()
    task = TASK.upper()

    # Segment
    if task in TASK_SEGMENTS:
        seg_t0, seg_dur = TASK_SEGMENTS[task]
        seg_t1 = seg_t0 + seg_dur
    else:
        seg_t0, seg_t1 = 0.0, None

    # --- Locate files ---
    structured_subject_dir = STRUCT_DIR / subject
    session_dir = find_structured_session_dir(structured_subject_dir, task)
    prefix = infer_prefix_in_session_dir(session_dir, subject)
    sync_csv = find_sync_csv(session_dir, subject, task, FS)

    # --- Load synchronized CSV ---
    df = pd.read_csv(sync_csv)
    t = df["Time"].to_numpy(dtype=np.float64)

    if seg_t1 is None:
        seg_t1 = float(np.nanmax(t))
    keep = (t >= seg_t0) & (t <= seg_t1) & np.isfinite(t)
    df = df.loc[keep].reset_index(drop=True)
    t = df["Time"].to_numpy(dtype=np.float64)

    # ROI signals
    roi_raw = {}
    for r in ROI_NAMES:
        col = f"roi_{r}"
        if col in df.columns:
            roi_raw[r] = df[col].to_numpy(dtype=np.float64)
        elif r in df.columns:
            roi_raw[r] = df[r].to_numpy(dtype=np.float64)
        else:
            roi_raw[r] = np.full_like(t, np.nan, dtype=np.float64)

    # GT signals
    peda = df["peda"].to_numpy(dtype=np.float64) if "peda" in df.columns else np.full_like(t, np.nan)
    pp = df["pp"].to_numpy(dtype=np.float64) if "pp" in df.columns else np.full_like(t, np.nan)
    pp_nr = df["pp_NR"].to_numpy(dtype=np.float64) if "pp_NR" in df.columns else np.full_like(t, np.nan)

    print(f"[INFO] subject={subject} task={task}")
    print(f"[INFO] sync_csv={sync_csv.name} | n={len(t)} | segment [{seg_t0:.1f}s, {seg_t1:.1f}s]")
    print(f"[INFO] EDA methods: {EDA_METHODS}")
    print(f"[INFO] EDA ROIs: {list(EDA_ROIS.keys())}")
    print(f"[INFO] Lowpass cutoff: {EDA_LOWPASS_CUTOFF} Hz | Smooth window: {EDA_SMOOTH_WINDOW_S}s")

    # =====================================================================
    # 1. Extract EDA trends from all ROIs × all methods
    # =====================================================================
    # Structure: eda_signals[roi_label][method_name] = 1D array at FS Hz
    eda_signals = {}
    for roi_label, roi_spec in EDA_ROIS.items():
        raw_sig = build_roi_signal(roi_raw, roi_spec)
        eda_signals[roi_label] = {}
        for method in EDA_METHODS:
            try:
                trend = extract_eda_trend(raw_sig, FS, method=method)
                eda_signals[roi_label][method] = trend
            except Exception as ex:
                print(f"  [WARN] {roi_label}/{method} failed: {ex}")
                eda_signals[roi_label][method] = np.full_like(raw_sig, np.nan)

    print(f"[INFO] Extracted {len(EDA_ROIS)} ROIs × {len(EDA_METHODS)} methods")

    # =====================================================================
    # 2. Downsample everything to EDA_OUTPUT_FS for metrics
    # =====================================================================
    t_ds, peda_ds = downsample_to_fs(t, peda, EDA_OUTPUT_FS)
    _, pp_ds = downsample_to_fs(t, pp, EDA_OUTPUT_FS)
    _, pp_nr_ds = downsample_to_fs(t, pp_nr, EDA_OUTPUT_FS)

    eda_ds = {}
    for roi_label in eda_signals:
        eda_ds[roi_label] = {}
        for method in eda_signals[roi_label]:
            _, ds = downsample_to_fs(t, eda_signals[roi_label][method], EDA_OUTPUT_FS)
            eda_ds[roi_label][method] = ds

    # =====================================================================
    # 3. Compute similarity metrics
    # =====================================================================
    # Compare each (ROI × method) against both PEDA and PP_NR
    results_pp = []
    results_peda = []

    for roi_label in eda_ds:
        for method in eda_ds[roi_label]:
            sig = eda_ds[roi_label][method]

            # vs PP_NR (perinasal perspiration — most relevant for nose and eyes)
            metrics_pp = calc_eda_similarity(sig, pp_nr_ds, fs=EDA_OUTPUT_FS)
            metrics_pp["roi"] = roi_label
            metrics_pp["method"] = method
            results_pp.append(metrics_pp)

            # vs PEDA (palm EDA)
            metrics_peda = calc_eda_similarity(sig, peda_ds, fs=EDA_OUTPUT_FS)
            metrics_peda["roi"] = roi_label
            metrics_peda["method"] = method
            results_peda.append(metrics_peda)

    # Build results tables
    df_pp = pd.DataFrame(results_pp)
    df_peda = pd.DataFrame(results_peda)

    # Sort by best PCC_abs
    df_pp_sorted = df_pp.sort_values("PCC_abs", ascending=False)
    df_peda_sorted = df_peda.sort_values("PCC_abs", ascending=False)

    print("\n" + "=" * 80)
    print("  SIMILARITY vs PP_NR (perinasal perspiration)")
    print("=" * 80)
    print(df_pp_sorted[["roi", "method", "PCC", "Spearman", "PCC_abs",
                         "max_xcorr", "lag_at_max_s", "RMSE_z", "trend_agree_pct"
                         ]].to_string(index=False))

    print("\n" + "=" * 80)
    print("  SIMILARITY vs PEDA (palm EDA)")
    print("=" * 80)
    print(df_peda_sorted[["roi", "method", "PCC", "Spearman", "PCC_abs",
                           "max_xcorr", "lag_at_max_s", "RMSE_z", "trend_agree_pct"
                           ]].to_string(index=False))

    # Find best combo for each GT
    best_pp = df_pp_sorted.iloc[0]
    best_peda = df_peda_sorted.iloc[0]
    print(f"\n[BEST vs PP_NR]  roi={best_pp['roi']}, method={best_pp['method']}, "
          f"PCC_abs={best_pp['PCC_abs']}, xcorr={best_pp['max_xcorr']}")
    print(f"[BEST vs PEDA]   roi={best_peda['roi']}, method={best_peda['method']}, "
          f"PCC_abs={best_peda['PCC_abs']}, xcorr={best_peda['max_xcorr']}")

    # =====================================================================
    # 4. Visualization
    # =====================================================================

    # --- Fig 1: Default method, all ROIs vs GT ---
    fig1, axes1 = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    # Panel 1: GT signals
    ax = axes1[0]
    ax.grid(True, alpha=0.3)
    ax.set_title(f"{subject}-{task} | Ground Truth EDA Signals", fontweight='bold')
    ax.plot(t, zscore(pp_nr), linewidth=1.0, label="PP_NR (perinasal, z-scored)", color='tab:red')
    ax.plot(t, zscore(peda), linewidth=1.0, label="PEDA (palm, z-scored)", color='tab:purple', alpha=0.8)
    ax.set_ylabel("Z-score")
    ax.legend(loc="upper right", fontsize=9)

    # Panel 2: Nose EDA vs PP_NR
    ax = axes1[1]
    ax.grid(True, alpha=0.3)
    nose_m = EDA_DEFAULT_METHOD
    nose_sig = eda_signals.get("nose", {}).get(nose_m, np.full_like(t, np.nan))
    nose_metrics = calc_eda_similarity(
        downsample_to_fs(t, nose_sig, EDA_OUTPUT_FS)[1],
        pp_nr_ds, fs=EDA_OUTPUT_FS)
    ax.plot(t, zscore(pp_nr), linewidth=1.0, color='tab:red', alpha=0.6, label="PP_NR (GT)")
    ax.plot(t, zscore(nose_sig), linewidth=1.0, color='tab:blue',
            label=f"Nose EDA [{nose_m}]")
    ax.set_title(f"Nose thermal EDA vs PP_NR | PCC={nose_metrics['PCC']}, "
                 f"Spearman={nose_metrics['Spearman']}, lag={nose_metrics['lag_at_max_s']}s")
    ax.set_ylabel("Z-score")
    ax.legend(loc="upper right", fontsize=9)

    # Panel 3: Eyes EDA vs PP_NR
    ax = axes1[2]
    ax.grid(True, alpha=0.3)
    eyes_sig = eda_signals.get("eyes_avg", {}).get(nose_m, np.full_like(t, np.nan))
    eyes_metrics = calc_eda_similarity(
        downsample_to_fs(t, eyes_sig, EDA_OUTPUT_FS)[1],
        pp_nr_ds, fs=EDA_OUTPUT_FS)
    ax.plot(t, zscore(pp_nr), linewidth=1.0, color='tab:red', alpha=0.6, label="PP_NR (GT)")
    ax.plot(t, zscore(eyes_sig), linewidth=1.0, color='tab:green',
            label=f"Eyes avg EDA [{nose_m}]")
    ax.set_title(f"Eyes thermal EDA vs PP_NR | PCC={eyes_metrics['PCC']}, "
                 f"Spearman={eyes_metrics['Spearman']}, lag={eyes_metrics['lag_at_max_s']}s")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Z-score")
    ax.legend(loc="upper right", fontsize=9)

    fig1.tight_layout()

    # --- Fig 2: All methods for nose ROI vs PP_NR ---
    n_methods = len(EDA_METHODS)
    fig2, axes2 = plt.subplots(n_methods, 1, figsize=(16, 3 * n_methods), sharex=True)
    if n_methods == 1:
        axes2 = [axes2]

    for i, method in enumerate(EDA_METHODS):
        ax = axes2[i]
        ax.grid(True, alpha=0.3)
        sig = eda_signals.get("nose", {}).get(method, np.full_like(t, np.nan))
        met = calc_eda_similarity(
            downsample_to_fs(t, sig, EDA_OUTPUT_FS)[1],
            pp_nr_ds, fs=EDA_OUTPUT_FS)
        ax.plot(t, zscore(pp_nr), linewidth=0.8, color='tab:red', alpha=0.5, label="PP_NR")
        ax.plot(t, zscore(sig), linewidth=0.8, color='tab:blue',
                label=f"nose / {method}")
        ax.set_title(f"{method} | PCC={met['PCC']}, Spear={met['Spearman']}, "
                     f"xcorr={met['max_xcorr']}, lag={met['lag_at_max_s']}s",
                     fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
        if i == n_methods - 1:
            ax.set_xlabel("Time (s)")

    fig2.suptitle(f"{subject}-{task} | Nose EDA: all methods vs PP_NR (z-scored)",
                  fontsize=13, fontweight='bold', y=1.001)
    fig2.tight_layout()

    # --- Fig 3: All ROIs with default method vs both GT ---
    roi_labels = list(EDA_ROIS.keys())
    fig3, axes3 = plt.subplots(len(roi_labels) + 1, 1,
                               figsize=(16, 3 * (len(roi_labels) + 1)), sharex=True)

    # First panel: both GT signals
    ax = axes3[0]
    ax.grid(True, alpha=0.3)
    ax.plot(t, zscore(pp_nr), linewidth=1.0, color='tab:red', label="PP_NR")
    ax.plot(t, zscore(peda), linewidth=1.0, color='tab:purple', alpha=0.8, label="PEDA")
    ax.set_title(f"{subject}-{task} | GT signals", fontweight='bold')
    ax.legend(loc="upper right", fontsize=9)

    for i, roi_label in enumerate(roi_labels):
        ax = axes3[i + 1]
        ax.grid(True, alpha=0.3)
        sig = eda_signals.get(roi_label, {}).get(EDA_DEFAULT_METHOD, np.full_like(t, np.nan))
        met_pp = calc_eda_similarity(
            downsample_to_fs(t, sig, EDA_OUTPUT_FS)[1],
            pp_nr_ds, fs=EDA_OUTPUT_FS)
        met_peda = calc_eda_similarity(
            downsample_to_fs(t, sig, EDA_OUTPUT_FS)[1],
            peda_ds, fs=EDA_OUTPUT_FS)

        ax.plot(t, zscore(pp_nr), linewidth=0.7, color='tab:red', alpha=0.4, label="PP_NR")
        ax.plot(t, zscore(peda), linewidth=0.7, color='tab:purple', alpha=0.4, label="PEDA")
        ax.plot(t, zscore(sig), linewidth=1.0, color='tab:blue',
                label=f"{roi_label} [{EDA_DEFAULT_METHOD}]")
        ax.set_title(f"{roi_label} | vs PP: PCC={met_pp['PCC']}, "
                     f"vs PEDA: PCC={met_peda['PCC']}",
                     fontsize=10)
        ax.legend(loc="upper right", fontsize=8)

    axes3[-1].set_xlabel("Time (s)")
    fig3.suptitle(f"{subject}-{task} | All ROIs [{EDA_DEFAULT_METHOD}] vs GT (z-scored)",
                  fontsize=13, fontweight='bold', y=1.001)
    fig3.tight_layout()

    # --- Fig 4: Heatmap of PCC_abs across ROIs × methods ---
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(16, 6))

    # vs PP_NR heatmap
    pivot_pp = df_pp.pivot_table(index="roi", columns="method", values="PCC_abs")
    pivot_pp = pivot_pp.reindex(columns=EDA_METHODS)  # consistent order
    im1 = ax4a.imshow(pivot_pp.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax4a.set_xticks(range(len(EDA_METHODS)))
    ax4a.set_xticklabels(EDA_METHODS, rotation=45, ha='right', fontsize=8)
    ax4a.set_yticks(range(len(pivot_pp.index)))
    ax4a.set_yticklabels(pivot_pp.index, fontsize=9)
    for i in range(pivot_pp.shape[0]):
        for j in range(pivot_pp.shape[1]):
            v = pivot_pp.values[i, j]
            if np.isfinite(v):
                ax4a.text(j, i, f"{v:.2f}", ha='center', va='center', fontsize=7,
                         color='black' if v > 0.3 else 'gray')
    ax4a.set_title("PCC_abs vs PP_NR", fontweight='bold')
    plt.colorbar(im1, ax=ax4a, shrink=0.8)

    # vs PEDA heatmap
    pivot_peda = df_peda.pivot_table(index="roi", columns="method", values="PCC_abs")
    pivot_peda = pivot_peda.reindex(columns=EDA_METHODS)
    im2 = ax4b.imshow(pivot_peda.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax4b.set_xticks(range(len(EDA_METHODS)))
    ax4b.set_xticklabels(EDA_METHODS, rotation=45, ha='right', fontsize=8)
    ax4b.set_yticks(range(len(pivot_peda.index)))
    ax4b.set_yticklabels(pivot_peda.index, fontsize=9)
    for i in range(pivot_peda.shape[0]):
        for j in range(pivot_peda.shape[1]):
            v = pivot_peda.values[i, j]
            if np.isfinite(v):
                ax4b.text(j, i, f"{v:.2f}", ha='center', va='center', fontsize=7,
                         color='black' if v > 0.3 else 'gray')
    ax4b.set_title("PCC_abs vs PEDA", fontweight='bold')
    plt.colorbar(im2, ax=ax4b, shrink=0.8)

    fig4.suptitle(f"{subject}-{task} | EDA Extraction: ROI × Method Comparison",
                  fontsize=13, fontweight='bold')
    fig4.tight_layout()

    # --- Fig 5: Cross-correlation functions for best ROIs ---
    fig5, (ax5a, ax5b) = plt.subplots(2, 1, figsize=(14, 8))

    for roi_label, color in [("nose", "tab:blue"), ("eyes_avg", "tab:green"),
                              ("forehead", "tab:orange")]:
        sig = eda_ds.get(roi_label, {}).get(EDA_DEFAULT_METHOD, np.array([]))
        if sig.size < 20:
            continue

        # Compute full xcorr
        ez = zscore(sig)
        rz = zscore(pp_nr_ds)
        valid = np.isfinite(ez) & np.isfinite(rz)
        if np.sum(valid) < 20:
            continue
        ez, rz = ez[valid], rz[valid]
        n = len(ez)

        xcorr = np.correlate(ez - ez.mean(), rz - rz.mean(), mode='full')
        norm = np.sqrt(np.sum((ez - ez.mean()) ** 2) * np.sum((rz - rz.mean()) ** 2))
        if norm > 1e-12:
            xcorr /= norm

        lags = np.arange(-(n - 1), n) / EDA_OUTPUT_FS
        ax5a.plot(lags, xcorr, linewidth=0.8, color=color, label=roi_label, alpha=0.8)

    ax5a.set_xlim(-120, 120)
    ax5a.axvline(0, color='k', linestyle='--', alpha=0.4)
    ax5a.set_title(f"{subject}-{task} | Cross-correlation with PP_NR [{EDA_DEFAULT_METHOD}]",
                   fontweight='bold')
    ax5a.set_ylabel("Normalized cross-correlation")
    ax5a.legend(fontsize=9)
    ax5a.grid(True, alpha=0.3)

    # Same for PEDA
    for roi_label, color in [("cheeks_avg", "tab:blue"), ("forehead", "tab:orange"),
                              ("nose", "tab:green")]:
        sig = eda_ds.get(roi_label, {}).get(EDA_DEFAULT_METHOD, np.array([]))
        if sig.size < 20:
            continue

        ez = zscore(sig)
        rz = zscore(peda_ds)
        valid = np.isfinite(ez) & np.isfinite(rz)
        if np.sum(valid) < 20:
            continue
        ez, rz = ez[valid], rz[valid]
        n = len(ez)

        xcorr = np.correlate(ez - ez.mean(), rz - rz.mean(), mode='full')
        norm = np.sqrt(np.sum((ez - ez.mean()) ** 2) * np.sum((rz - rz.mean()) ** 2))
        if norm > 1e-12:
            xcorr /= norm

        lags = np.arange(-(n - 1), n) / EDA_OUTPUT_FS
        ax5b.plot(lags, xcorr, linewidth=0.8, color=color, label=roi_label, alpha=0.8)

    ax5b.set_xlim(-120, 120)
    ax5b.axvline(0, color='k', linestyle='--', alpha=0.4)
    ax5b.set_title(f"Cross-correlation with PEDA [{EDA_DEFAULT_METHOD}]", fontweight='bold')
    ax5b.set_xlabel("Lag (seconds)")
    ax5b.set_ylabel("Normalized cross-correlation")
    ax5b.legend(fontsize=9)
    ax5b.grid(True, alpha=0.3)

    fig5.tight_layout()

    # --- Save ---
    if SAVE_FIGS:
        for fig_obj, suffix in [
            (fig1, "eda_nose_eyes_vs_gt"),
            (fig2, "eda_nose_all_methods"),
            (fig3, "eda_all_rois_vs_gt"),
            (fig4, "eda_heatmap"),
            (fig5, "eda_crosscorr"),
        ]:
            fpath = FIG_OUT_DIR / f"{subject}-{task}_{suffix}.png"
            fig_obj.savefig(fpath, dpi=160, bbox_inches="tight")
        print(f"\n[OK] saved {5} figures to: {FIG_OUT_DIR}")

        # Save metrics CSV
        csv_path = FIG_OUT_DIR / f"{subject}-{task}_eda_metrics_vs_pp.csv"
        df_pp_sorted.to_csv(csv_path, index=False)
        csv_path2 = FIG_OUT_DIR / f"{subject}-{task}_eda_metrics_vs_peda.csv"
        df_peda_sorted.to_csv(csv_path2, index=False)
        print(f"[OK] saved metrics: {csv_path.name}, {csv_path2.name}")

    plt.show()


if __name__ == "__main__":
    main()
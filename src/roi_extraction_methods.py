# roi_extraction_methods.py
#
# Alternative spatial aggregation methods for extracting 1D signals from
# thermal ROI patches. Drop-in replacements for roi_mean().
#
# The key insight: simple np.mean() treats all pixels equally, but for
# physiological signal extraction from thermal imaging:
#   - Edge pixels are contaminated by background (hair, air)
#   - Microbolometer noise creates outlier pixels
#   - The pulsatile signal is strongest in the center of blood-perfused regions
#   - Spatial gradients within the ROI carry information too
#
# Usage:
#   Replace roi_mean(frame_raw, rect) with roi_extract(frame_raw, rect, method="...")

import numpy as np
from typing import Tuple, Dict


# =============================================================================
# Configuration: choose which method to use per ROI
# =============================================================================
# Recommended defaults per ROI based on physiology:
#
#   forehead  → "gaussian_weighted" or "trimmed_mean"
#               (large flat skin area, good for pulse; edge contamination from hair)
#
#   nose      → "center_crop" or "median"
#               (small, curved, respiration signal from temperature fluctuation;
#                edges pick up cheek/lip background)
#
#   cheek_l/r → "gaussian_weighted"
#               (mid-size region, pulse + EDA; corners may include jawline)
#
#   eye_l/r   → "trimmed_mean"
#               (very small ROI near orbital bone, high-temperature contrast
#                from lacrimal caruncle; outlier-robust method preferred)
#
ROI_METHOD_MAP = {
    "nose":     "gaussian_weighted",
    "eye_l":    "trimmed_mean",
    "eye_r":    "trimmed_mean",
    "cheek_l":  "gaussian_weighted",
    "cheek_r":  "gaussian_weighted",
    "forehead": "gaussian_weighted",
    "face_bbox": "trimmed_mean",
}

# Fallback method if ROI name not in the map
DEFAULT_METHOD = "gaussian_weighted"


# =============================================================================
# Individual extraction methods
# =============================================================================

def roi_mean(patch: np.ndarray) -> float:
    """
    Baseline: simple arithmetic mean.
    Fast but sensitive to outliers and edge contamination.
    """
    if patch.size == 0:
        return float("nan")
    return float(np.mean(patch.astype(np.float64)))


def roi_median(patch: np.ndarray) -> float:
    """
    Median: robust to outliers (hot/cold pixel defects, specular reflections).
    Slightly noisier than mean for Gaussian-distributed data, but much more
    robust to the heavy-tailed noise typical of uncooled microbolometers.
    """
    if patch.size == 0:
        return float("nan")
    return float(np.median(patch.astype(np.float64)))


def roi_trimmed_mean(patch: np.ndarray, trim_pct: float = 0.10) -> float:
    """
    Trimmed mean: discard the lowest and highest trim_pct of pixel values,
    then average the rest. Combines the efficiency of mean with robustness
    of median.

    trim_pct=0.10 removes the 10% coldest and 10% hottest pixels.
    This handles:
      - Dead/hot pixels on the microbolometer
      - Edge pixels that see background (much colder than skin)
      - Hair pixels (different emissivity)
    """
    if patch.size == 0:
        return float("nan")
    vals = np.sort(patch.astype(np.float64).ravel())
    n = vals.size
    lo = int(np.floor(trim_pct * n))
    hi = n - lo
    if hi <= lo:
        return float(np.mean(vals))
    return float(np.mean(vals[lo:hi]))


def roi_gaussian_weighted(patch: np.ndarray, sigma_frac: float = 0.35) -> float:
    """
    2D Gaussian-weighted mean: center pixels contribute more, edges contribute less.

    This is the most important improvement over simple mean because:
      1. Physiological signal (blood perfusion, temperature) is strongest at
         the spatial center of each anatomical ROI
      2. ROI boundaries inevitably include some non-target tissue
      3. The Gaussian weighting provides a smooth falloff instead of a hard crop

    sigma_frac: Gaussian sigma as fraction of patch half-width.
                0.35 means ~95% of the weight falls within the central 70% of the ROI.
                Smaller = more concentrated on center. Larger = closer to uniform mean.
    """
    if patch.size == 0:
        return float("nan")

    h, w = patch.shape[:2]
    if h < 2 or w < 2:
        return float(np.mean(patch.astype(np.float64)))

    # Build 2D Gaussian kernel
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    sigma_y = sigma_frac * h
    sigma_x = sigma_frac * w

    y = np.arange(h, dtype=np.float64)
    x = np.arange(w, dtype=np.float64)
    yy, xx = np.meshgrid(y, x, indexing='ij')

    kernel = np.exp(-0.5 * (((yy - cy) / sigma_y) ** 2 + ((xx - cx) / sigma_x) ** 2))
    kernel /= kernel.sum()

    vals = patch.astype(np.float64)
    return float(np.sum(vals * kernel))


def roi_center_crop(patch: np.ndarray, crop_frac: float = 0.6) -> float:
    """
    Use only the central crop_frac of the ROI (e.g., 60% = discard outer 20% on each side).
    Then take the mean of the cropped center.

    Simpler than Gaussian weighting but effective at removing edge contamination.
    Good for small ROIs (nose, eyes) where the Gaussian kernel might be too few pixels.
    """
    if patch.size == 0:
        return float("nan")

    h, w = patch.shape[:2]
    margin_y = int((1.0 - crop_frac) / 2.0 * h)
    margin_x = int((1.0 - crop_frac) / 2.0 * w)

    y1 = max(0, margin_y)
    y2 = max(y1 + 1, h - margin_y)
    x1 = max(0, margin_x)
    x2 = max(x1 + 1, w - margin_x)

    center = patch[y1:y2, x1:x2]
    if center.size == 0:
        return float(np.mean(patch.astype(np.float64)))
    return float(np.mean(center.astype(np.float64)))


def roi_percentile(patch: np.ndarray, lo_pct: float = 25.0, hi_pct: float = 75.0) -> float:
    """
    IQR mean: average only pixels between the lo_pct and hi_pct percentiles.

    More adaptive than trimmed_mean for non-symmetric distributions (e.g., when
    part of the ROI sees skin and part sees background).

    Default (25th to 75th percentile) is the interquartile mean.
    """
    if patch.size == 0:
        return float("nan")
    vals = patch.astype(np.float64).ravel()
    lo = np.percentile(vals, lo_pct)
    hi = np.percentile(vals, hi_pct)
    mask = (vals >= lo) & (vals <= hi)
    if np.sum(mask) == 0:
        return float(np.mean(vals))
    return float(np.mean(vals[mask]))


def roi_hottest_fraction(patch: np.ndarray, frac: float = 0.3) -> float:
    """
    Mean of the hottest frac% of pixels.

    Rationale for perinasal EDA / respiration:
      - During exhalation, warm air heats the perinasal region
      - The warmest pixels correspond to the area directly in the airflow path
      - Averaging only these tracks the respiratory temperature modulation
        better than a full-ROI mean where cold surrounding pixels dilute the signal

    Also useful for pulse from forehead/cheeks: the warmest pixels are the
    most blood-perfused ones, which carry the strongest pulsatile signal.

    frac=0.3 means use the top 30% warmest pixels.
    """
    if patch.size == 0:
        return float("nan")
    vals = np.sort(patch.astype(np.float64).ravel())
    n = vals.size
    cutoff = max(1, int(np.ceil((1.0 - frac) * n)))
    return float(np.mean(vals[cutoff:]))


def roi_spatial_gradient(patch: np.ndarray) -> float:
    """
    Mean spatial gradient magnitude (Sobel-based).

    This does NOT extract a temperature value — instead it extracts the
    spatial texture/edge energy within the ROI. This is complementary to
    mean temperature and can be useful as:
      - A motion artifact indicator (sudden gradient changes = head movement)
      - An auxiliary signal for multi-channel decomposition (OMIT, ICA)
      - A quality metric: low gradient = uniform skin, high = mixed content

    Returns the mean gradient magnitude (always >= 0).
    """
    if patch.size == 0 or patch.shape[0] < 3 or patch.shape[1] < 3:
        return float("nan")

    vals = patch.astype(np.float64)

    # Sobel gradients
    gy = vals[2:, 1:-1] - vals[:-2, 1:-1]  # vertical
    gx = vals[1:-1, 2:] - vals[1:-1, :-2]  # horizontal
    mag = np.sqrt(gx ** 2 + gy ** 2)

    return float(np.mean(mag))


def roi_std(patch: np.ndarray) -> float:
    """
    Spatial standard deviation within the ROI.

    Useful as a quality/noise indicator:
      - High std = mixed content (skin + background) or noisy frame
      - Low std = clean, uniform skin patch
      - Can be used to weight or reject frames in downstream processing
    """
    if patch.size == 0:
        return float("nan")
    return float(np.std(patch.astype(np.float64)))


# =============================================================================
# Dispatcher: use this as the drop-in replacement for roi_mean()
# =============================================================================

_METHOD_DISPATCH = {
    "mean":              roi_mean,
    "median":            roi_median,
    "trimmed_mean":      roi_trimmed_mean,
    "gaussian_weighted": roi_gaussian_weighted,
    "center_crop":       roi_center_crop,
    "percentile":        roi_percentile,
    "hottest":           roi_hottest_fraction,
    "gradient":          roi_spatial_gradient,
    "std":               roi_std,
}


def roi_extract(frame_raw: np.ndarray, rect: Tuple[int, int, int, int],
                method: str = "gaussian_weighted") -> float:
    """
    Extract a scalar value from a rectangular ROI in a raw thermal frame.

    Drop-in replacement for roi_mean(frame_raw, rect).

    Args:
        frame_raw: 2D array (H, W), raw thermal values (uint16 or similar)
        rect: (x1, y1, x2, y2) bounding box
        method: one of the extraction methods (see _METHOD_DISPATCH)

    Returns:
        float scalar value
    """
    x1, y1, x2, y2 = rect
    patch = frame_raw[y1:y2, x1:x2]
    if patch.size == 0:
        return float("nan")

    fn = _METHOD_DISPATCH.get(method)
    if fn is None:
        raise ValueError(f"Unknown ROI extraction method: '{method}'. "
                         f"Available: {list(_METHOD_DISPATCH.keys())}")
    return fn(patch)


def roi_extract_multi(frame_raw: np.ndarray, rect: Tuple[int, int, int, int],
                      methods: list = None) -> Dict[str, float]:
    """
    Extract multiple features from a single ROI patch in one call.
    Useful for building a richer feature vector per ROI per frame.

    Default methods: ["gaussian_weighted", "std"] gives you the
    temperature signal + a quality indicator.

    Args:
        frame_raw: 2D raw thermal frame
        rect: (x1, y1, x2, y2)
        methods: list of method names

    Returns:
        dict mapping method name -> scalar value
    """
    if methods is None:
        methods = ["gaussian_weighted", "std"]

    x1, y1, x2, y2 = rect
    patch = frame_raw[y1:y2, x1:x2]

    out = {}
    for m in methods:
        if patch.size == 0:
            out[m] = float("nan")
        else:
            fn = _METHOD_DISPATCH.get(m)
            if fn is None:
                raise ValueError(f"Unknown method: '{m}'")
            out[m] = fn(patch)
    return out


def roi_extract_by_name(frame_raw: np.ndarray, rect: Tuple[int, int, int, int],
                        roi_name: str) -> float:
    """
    Extract using the recommended method for a specific ROI name.
    Uses ROI_METHOD_MAP configuration at the top of this file.

    Drop-in replacement:
        OLD: sig[name][k] = roi_mean(frame_raw, rois[name])
        NEW: sig[name][k] = roi_extract_by_name(frame_raw, rois[name], name)
    """
    method = ROI_METHOD_MAP.get(roi_name, DEFAULT_METHOD)
    return roi_extract(frame_raw, rect, method=method)
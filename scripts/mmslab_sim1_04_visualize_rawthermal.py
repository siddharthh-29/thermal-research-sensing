"""
mmslab_sim1_04_visualize_rawthermal.py

Loads a raw thermal recording (.dat + .inf) and renders a preview animation
saved as an MP4 file.

Pipeline step 4: Visualize raw thermal data before annotation.

Configuration:
  Edit DAT_PATH, INF_PATH, START_FRAME, END_FRAME, FPS, and OUT_PATH below.
"""

import os
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter


# =============================================================================
# Configuration
# =============================================================================
DAT_PATH = "/media/arritmic/T7/DATABASES/TEMPORAL/THERMALDD/T054/T054-BL.dat"
INF_PATH  = "/media/arritmic/T7/DATABASES/TEMPORAL/THERMALDD/T054/T054-BL.inf"

START_FRAME = 0
END_FRAME   = 100   # inclusive
FPS         = 7.5   # native SIM1 frame rate
OUT_PATH    = "T054_BL_000_100.mp4"


# =============================================================================
# .inf / .dat readers
# =============================================================================

def read_inf(inf_path: str):
    """Parse a .inf text header and return (n_frames, width, height, timestamps)."""
    with open(inf_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)

    if len(nums) < 3:
        raise ValueError(f"Could not find 3 numeric header values in: {inf_path}")

    n_frames = int(float(nums[0]))
    width    = int(float(nums[1]))
    height   = int(float(nums[2]))

    raw = np.array([float(x) for x in nums[3:]], dtype=np.float64)

    if raw.size == 0:
        timestamps = raw
    elif raw.size == n_frames:
        timestamps = raw
    elif raw.size % n_frames == 0:
        k = raw.size // n_frames
        timestamps = raw.reshape(n_frames, k)[:, -1]
    else:
        raise ValueError(
            f"Unexpected timestamp token count in {inf_path}. "
            f"n_frames={n_frames}, tokens_after_header={raw.size}"
        )

    return n_frames, width, height, timestamps


def guess_dtype_from_size(dat_path: str, n_frames: int, width: int, height: int) -> np.dtype:
    """Infer pixel dtype (uint16 / float32 / uint8) from file size."""
    expected_samples = n_frames * width * height
    size_bytes = os.path.getsize(dat_path)
    bps = size_bytes / expected_samples

    if abs(bps - 2.0) < 0.1:
        return np.uint16
    elif abs(bps - 4.0) < 0.1:
        return np.float32
    elif abs(bps - 1.0) < 0.1:
        return np.uint8
    else:
        raise ValueError(
            f"Unexpected bytes per sample: {bps:.4f}. File size={size_bytes} bytes."
        )


def load_thermal_dat(dat_path: str, inf_path: str, use_memmap: bool = True):
    """Load raw thermal frames from a .dat/.inf pair."""
    n_frames, width, height, ts = read_inf(inf_path)
    dtype  = guess_dtype_from_size(dat_path, n_frames, width, height)
    shape  = (n_frames, height, width)

    if use_memmap:
        frames = np.memmap(dat_path, dtype=dtype, mode="r", shape=shape)
    else:
        frames_1d = np.fromfile(dat_path, dtype=dtype)
        if frames_1d.size != n_frames * width * height:
            raise ValueError(
                f"Sample count mismatch. Expected {n_frames * width * height}, "
                f"got {frames_1d.size}"
            )
        frames = frames_1d.reshape(shape)

    return frames, ts, (width, height), dtype


# =============================================================================
# Main
# =============================================================================

def main():
    frames, ts, (w, h), dtype = load_thermal_dat(DAT_PATH, INF_PATH, use_memmap=True)
    print(f"Loaded  shape={frames.shape}  dtype={dtype}  timestamps={ts.shape}")

    # Single-frame preview
    fig0, ax0 = plt.subplots()
    ax0.imshow(frames[START_FRAME], cmap="inferno")
    ax0.set_title(
        f"Frame {START_FRAME}  t={ts[START_FRAME]:.3f}" if ts.size else f"Frame {START_FRAME}"
    )
    plt.colorbar(ax0.images[0], ax=ax0)
    plt.tight_layout()
    plt.show()

    # Animation
    fig, ax = plt.subplots()
    im    = ax.imshow(frames[START_FRAME], cmap="inferno")
    title = ax.set_title("")

    def update(k):
        im.set_data(frames[k])
        title.set_text(f"Frame {k}  t={ts[k]:.3f}" if ts.size else f"Frame {k}")
        return im, title

    frame_indices = range(START_FRAME, END_FRAME + 1)
    ani = FuncAnimation(fig, update, frames=frame_indices, interval=1000 / FPS, blit=False)

    writer = FFMpegWriter(fps=FPS, metadata={"artist": "matplotlib"}, bitrate=3000)
    ani.save(OUT_PATH, writer=writer, dpi=150)
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()

"""
mmslab_sim1_03_explore_dataset_structure.py

Prints the SIM1 dataset folder tree, annotating known file extensions with
their signal type (HR, BR, perinasal perspiration, palm EDA, etc.).

Pipeline step 3: Inspect the downloaded dataset before running annotation
or signal extraction.

Configuration:
  Edit BASE_PATH below.
"""

import os

# =============================================================================
# Configuration
# =============================================================================
BASE_PATH = "/media/arritmic/HI-NRI-TINO-002/DATABASES/MULTIMODAL/Affective_States/SIMULATOR_STUDY_I"

# Known file-extension annotations
_EXT_LABELS = {
    '.HR':   '(Heart Rate)',
    '.BR':   '(Breathing Rate)',
    '.pp':   '(Perinasal Perspiration Index)',
    '.peda': '(Palm EDA)',
    '.res':  '(Vehicle Performance)',
    '.stm':  '(Stimulus Markers)',
    '.avi2': '(Thermal ROI Video)',
}


# =============================================================================
# Explorer
# =============================================================================

def explore_sim1_dataset(base_path: str) -> None:
    """Print a tree of the three SIM1 sub-directories with annotated file types."""
    sub_dirs = ["RawThermalData", "StructuredStudyData", "R-FriendlyStudyData"]

    for main_folder in sub_dirs:
        folder_path = os.path.join(base_path, main_folder)
        if not os.path.exists(folder_path):
            print(f"Directory missing: {folder_path}")
            continue

        print(f"\n--- {main_folder} ---")

        subjects = sorted(d for d in os.listdir(folder_path) if d.startswith('T'))

        for subject in subjects:
            subject_path = os.path.join(folder_path, subject)
            print(f"\n[Subject: {subject}]")

            root_files = [
                f for f in os.listdir(subject_path)
                if os.path.isfile(os.path.join(subject_path, f))
            ]
            for rf in root_files:
                print(f"  ├─ {rf}")

            session_folders = sorted(
                d for d in os.listdir(subject_path)
                if os.path.isdir(os.path.join(subject_path, d))
            )
            for sf in session_folders:
                print(f"  ├─ Session: {sf}")
                session_path = os.path.join(subject_path, sf)
                for fname in sorted(os.listdir(session_path)):
                    _, ext = os.path.splitext(fname)
                    label = _EXT_LABELS.get(ext, '')
                    print(f"  │  └─ {fname} {label}")


if __name__ == "__main__":
    explore_sim1_dataset(BASE_PATH)

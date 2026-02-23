"""
mmslab_sim1_01_download_rawthermal.py

Downloads raw thermal recordings (.dat and .inf files) from the SIM1 dataset
hosted on OSF (https://osf.io/c42cn).

Pipeline step 1: Fetch raw thermal data before any processing.

Requirements: osfclient  (pip install osfclient)

Configuration:
  Edit SUBJECT_FILTER and TASK_FILTER below to select what to download.
  LOCAL_BASE_DIR must point to the intended storage location.
"""

import os
import sys
from osfclient.api import OSF

# =============================================================================
# Configuration
# =============================================================================
PROJECT_ID = 'c42cn'
LOCAL_BASE_DIR  = "/media/arritmic/HI-NRI-TINO-002/DATABASES/MULTIMODAL/Affective_States/SIMULATOR_STUDY_I/RawThermalData"

# Set to None to download all available subjects.
SUBJECT_FILTER = ["T014", "T016", "T017", "T018", "T023"]
TASK_FILTER    = ['PD', 'ND', 'CD', 'ED']


def download_raw_thermal_data():
    print(f"Connecting to OSF project {PROJECT_ID}...")
    osf     = OSF()
    project = osf.project(PROJECT_ID)
    store   = project.storage('osfstorage')

    print("Scanning remote files (may take a moment)...")

    count = 0
    for file in store.files:
        if "Raw Thermal Data" not in file.path:
            continue

        parts      = file.path.strip('/').split('/')
        if len(parts) < 2:
            continue

        subject_id = parts[1]
        filename   = parts[-1]

        if SUBJECT_FILTER is not None and subject_id not in SUBJECT_FILTER:
            continue

        # Filenames follow the pattern Txxx-TASK.dat / Txxx-TASK.inf
        if not any(f"-{task}." in filename.upper() for task in TASK_FILTER):
            continue

        relative_path = os.path.join(*parts[1:])
        local_path    = os.path.join(LOCAL_BASE_DIR, relative_path)
        local_dir     = os.path.dirname(local_path)

        if not os.path.exists(local_dir):
            os.makedirs(local_dir)

        if not os.path.exists(local_path):
            print(f"Downloading: {subject_id}/{filename} ...")
            try:
                with open(local_path, 'wb') as f:
                    file.write_to(f)
                count += 1
            except Exception as e:
                print(f"ERROR downloading {file.path}: {e}")
        else:
            print(f"Skipping (exists): {subject_id}/{filename}")

    print(f"\nDownload complete! {count} files downloaded to '{LOCAL_BASE_DIR}'.")


if __name__ == "__main__":
    download_raw_thermal_data()

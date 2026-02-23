"""
mmslab_sim1_02_download_structured_data.py

Downloads the StructuredStudyData folder from the SIM1 OSF repository
(https://osf.io/c42cn).  This contains the synchronized CSV files with
per-frame ROI temperature traces and ground-truth biosignals (HR, BR, PEDA, PP).

Pipeline step 2: Fetch structured data after downloading raw thermal files.

Requirements: osfclient  (pip install osfclient)

Configuration:
  Edit SUBJECTS_TO_SAMPLE to select which subjects to download.
"""

import os
from osfclient.api import OSF

# =============================================================================
# Configuration
# =============================================================================
PROJECT_ID     = 'c42cn'
LOCAL_BASE_DIR = "/media/arritmic/HI-NRI-TINO-002/DATABASES/MULTIMODAL/Affective_States/SIMULATOR_STUDY_I/StructuredStudyData"

ALWAYS_DOWNLOAD    = ['Dataset-Table-Index.xlsx']
SUBJECTS_TO_SAMPLE = ["T014", "T016", "T017", "T018", "T023"]


def download_structured_samples():
    print(f"Connecting to OSF project {PROJECT_ID}...")
    osf     = OSF()
    project = osf.project(PROJECT_ID)
    store   = project.storage('osfstorage')

    print(f"Scanning remote files... Index + subjects: {SUBJECTS_TO_SAMPLE}")

    count = 0
    for file in store.files:
        path_parts = file.path.strip('/').split('/')
        filename   = path_parts[-1]

        should_download = False

        if filename in ALWAYS_DOWNLOAD:
            should_download = True
        elif "Structured Study Data" in file.path:
            if any(subj in file.path for subj in SUBJECTS_TO_SAMPLE):
                should_download = True

        if should_download:
            local_path = os.path.join(LOCAL_BASE_DIR, *path_parts)
            local_dir  = os.path.dirname(local_path)

            if not os.path.exists(local_dir):
                os.makedirs(local_dir)

            if not os.path.exists(local_path):
                print(f"Downloading: {filename}")
                try:
                    with open(local_path, 'wb') as f:
                        file.write_to(f)
                    count += 1
                except Exception as e:
                    print(f"Error downloading {filename}: {e}")
            else:
                print(f"Skipping (exists): {filename}")

    print(f"\nSample download complete! {count} files saved to '{LOCAL_BASE_DIR}'.")


if __name__ == "__main__":
    download_structured_samples()
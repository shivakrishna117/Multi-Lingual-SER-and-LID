"""
download_guide_ML.py
====================
Week 1 — Dataset Download Helper & Folder Setup.

Run this FIRST. It:
  1. Creates the full project folder structure.
  2. Prints exact download links + placement instructions for each dataset.
  3. Installs all required Python packages.

Usage:
    python download_guide_ML.py
"""

import subprocess
import sys
from pathlib import Path

# ──────────────────────────────────────────────
# FOLDER STRUCTURE
# ──────────────────────────────────────────────
FOLDERS = [
    # Raw (put downloaded + unzipped datasets here)
    "data_ML/raw/telugu_ML",
    "data_ML/raw/hindi_ML",
    "data_ML/raw/tamil_ML",
    "data_ML/raw/english_ML",

    # Processed (auto-created by preprocess_ML.py)
    "data_ML/processed/telugu_ML",
    "data_ML/processed/hindi_ML",
    "data_ML/processed/tamil_ML",
    "data_ML/processed/english_ML",

    # Features (auto-created by extract_features_ML.py in Week 2)
    "data_ML/features",

    # Models (auto-created during training in Week 3)
    "models_ML/ser",
    "models_ML/lid",

    # Results
    "results_ML",

    # Streamlit app (Week 6)
    "app_ML",
]


def create_folders():
    print("\n  Creating project folder structure …")
    for f in FOLDERS:
        Path(f).mkdir(parents=True, exist_ok=True)
    print("  ✅ All folders ready.\n")


# ──────────────────────────────────────────────
# DATASET INSTRUCTIONS
# ──────────────────────────────────────────────
DATASETS = [
    {
        "name":    "RAVDESS (English)",
        "folder":  "data_ML/raw/english_ML/",
        "url":     "https://zenodo.org/records/1188976",
        "notes": (
            "Download 'Audio_Speech_Actors_01-24.zip'\n"
            "   Unzip → you get Actor_01/ … Actor_24/ folders.\n"
            "   Place ALL Actor_* folders directly inside data_ML/raw/english_ML/\n"
            "   The preprocessor will read RAVDESS emotion from filename (e.g. 03-01-03-…)."
        ),
        "emotion_structure": "filename-encoded",
    },
    {
        "name":    "IESC (Hindi)",
        "folder":  "data_ML/raw/hindi_ML/",
        "url":     "https://www.kaggle.com/datasets/ybsingh/indian-emotional-speech-corpora-iesc",
        "notes": (
            "Download via Kaggle CLI:\n"
            "   kaggle datasets download -d ybsingh/indian-emotional-speech-corpora-iesc\n"
            "   Unzip and place emotion sub-folders inside data_ML/raw/hindi_ML/\n"
            "   Expected: data_ML/raw/hindi_ML/angry/  neutral/  happy/  sad/  fearful/"
        ),
        "emotion_structure": "sub-folder per emotion",
    },
    {
        "name":    "EmoTa (Tamil)",
        "folder":  "data_ML/raw/tamil_ML/",
        "url":     "https://github.com/aaivu/EmoTa",
        "notes": (
            "Clone the repo or download ZIP from GitHub.\n"
            "   Place audio files in emotion sub-folders:\n"
            "   data_ML/raw/tamil_ML/angry/  neutrality/  happiness/  sadness/  fear/\n"
            "   The emotion map in preprocess_ML.py already handles 'neutrality'→'neutral'."
        ),
        "emotion_structure": "sub-folder per emotion",
    },
    {
        "name":    "IIIT-H TEMD or Kaggle Telugu (Telugu)",
        "folder":  "data_ML/raw/telugu_ML/",
        "url":     "https://www.kaggle.com/datasets/jettysowmith/telugu-emotion-speech",
        "notes": (
            "Option A (Kaggle):\n"
            "   kaggle datasets download -d jettysowmith/telugu-emotion-speech\n"
            "   Unzip → place emotion sub-folders in data_ML/raw/telugu_ML/\n\n"
            "Option B (IIIT-H TEMD):\n"
            "   http://speech.iiit.ac.in/svldownloads/IIIT-HTEMD/\n"
            "   Same folder layout expected."
        ),
        "emotion_structure": "sub-folder per emotion",
    },
]


def print_download_guide():
    print("=" * 65)
    print("  DATASET DOWNLOAD GUIDE")
    print("=" * 65)
    for i, ds in enumerate(DATASETS, 1):
        print(f"\n  [{i}] {ds['name']}")
        print(f"       URL    : {ds['url']}")
        print(f"       Folder : {ds['folder']}")
        print(f"       Layout : {ds['emotion_structure']}")
        print()
        for line in ds["notes"].split("\n"):
            print(f"       {line}")
    print("\n" + "=" * 65)


# ──────────────────────────────────────────────
# REQUIREMENTS
# ──────────────────────────────────────────────
PACKAGES = [
    "librosa>=0.10",
    "soundfile",
    "numpy",
    "pandas",
    "scikit-learn",
    "tqdm",
    "matplotlib",
    "seaborn",
]


def install_packages():
    print("\n  Installing Python dependencies …")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet"] + PACKAGES
    )
    print("  ✅ All packages installed.\n")


# ──────────────────────────────────────────────
# VERIFY SETUP
# ──────────────────────────────────────────────
def verify():
    print("  Verifying imports …")
    missing = []
    for pkg in ["librosa", "soundfile", "numpy", "pandas", "sklearn", "tqdm"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"  ⚠  Still missing: {missing}  — try installing manually.")
    else:
        print("  ✅ All imports OK.\n")


# ──────────────────────────────────────────────
# CHECKLIST
# ──────────────────────────────────────────────
def print_checklist():
    print("=" * 65)
    print("  WEEK 1 CHECKLIST")
    print("=" * 65)
    steps = [
        ("download_guide_ML.py",   "Folder setup + dependencies (this file)"),
        ("preprocess_ML.py",       "Standardise audio → 16 kHz mono WAV"),
        ("build_csv_ML.py",        "Build master_ML.csv"),
        ("split_dataset_ML.py",    "Train / Val / Test stratified split"),
        ("— Week 2 —",             "extract_features_ML.py"),
    ]
    for fname, desc in steps:
        print(f"  {'✅' if fname == 'download_guide_ML.py' else '⬜'}  {fname:<30s}  {desc}")
    print("=" * 65)


def main():
    print("=" * 65)
    print("  download_guide_ML.py  —  Project Bootstrap")
    print("=" * 65)

    create_folders()
    install_packages()
    verify()
    print_download_guide()
    print_checklist()

    print("\n  ➡  After downloading datasets, run:")
    print("      python preprocess_ML.py")


if __name__ == "__main__":
    main()
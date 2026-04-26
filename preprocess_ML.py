"""
preprocess_ML.py
================
Week 1 - Step 1: Audio Standardisation for all 4 datasets.

Run this from inside:  C:\\Users\\shiva\\Desktop\\multi-lang\\

Actual raw data locations (auto-detected):
  telugu_ML/telugu/<emotion>/<file>.wav
  hindi_ML/Indian Emotional Speech Corpora (IESC)/Speaker-X/<Emotion>/<file>.wav
  german_ML/wav/<file>.wav
  english_ML/Actor_XX/<file>.wav

Output goes to:
  data_ML/processed/<language>/<emotion>/<file>.wav
"""

import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
TARGET_SR  = 16_000
TOP_DB     = 30
MIN_SECS   = 0.3
MAX_SECS   = 10.0

PROC_ROOT  = Path("data_ML/processed")

KEEP_EMOTIONS = {"neutral", "happy", "sad", "angry", "fearful"}

# ──────────────────────────────────────────────
# RAW DATA PATHS
# Each entry: (language_tag, raw_root_path, detection_mode)
# detection_mode: "folder" | "ravdess" | "emodb"
# ──────────────────────────────────────────────
LANG_CONFIGS = [
    {
        "tag":   "telugu",
        "raw":   Path("telugu_ML/telugu"),   # <-- direct path, no data_ML/raw
        "mode":  "folder",
    },
    {
        "tag":   "hindi",
        "raw":   Path("hindi_ML/Indian Emotional Speech Corpora (IESC)"),
        "mode":  "folder",   # emotion = deepest folder (Speaker-X/Emotion/file)
    },
    {
        "tag":   "german",
        "raw":   Path("german_ML/wav"),
        "mode":  "emodb",    # emotion from filename character [5]
    },
    {
        "tag":   "english",
        "raw":   Path("english_ML"),
        "mode":  "ravdess",  # emotion from filename dash-segments
    },
]

# ──────────────────────────────────────────────
# EMOTION MAPS
# ──────────────────────────────────────────────

FOLDER_MAP = {
    # Standard
    "neutral":   "neutral",
    "happy":     "happy",    "happiness": "happy",
    "sad":       "sad",      "sadness":   "sad",
    "angry":     "angry",    "anger":     "angry",
    "fearful":   "fearful",  "fear":      "fearful",
    "disgust":   "disgust",
    "surprise":  "surprised","surprised": "surprised",
    # Telugu typos (actual folder names in your dataset)
    "nuetral":   "neutral",   # <-- your dataset has this typo
    "suprised":  "surprised", # <-- your dataset has this typo
    "suprise":   "surprised",
    "happyy":    "happy",
    "sorrow":    "sad",
}

RAVDESS_MAP = {
    "01": "neutral", "02": "neutral",
    "03": "happy",   "04": "sad",
    "05": "angry",   "06": "fearful",
    "07": "disgust", "08": "surprised",
}

# EmoDB filename stem: e.g. 08a01Wc -> stem[5] = 'W' -> angry
EMODB_MAP = {
    "W": "angry",   "L": "neutral",
    "E": "disgust", "A": "fearful",
    "F": "happy",   "T": "sad",
    "N": "neutral",
}

# ──────────────────────────────────────────────
# EMOTION EXTRACTORS
# ──────────────────────────────────────────────

def get_emotion(fpath: Path, mode: str):
    if mode == "ravdess":
        parts = fpath.stem.split("-")
        if len(parts) >= 3:
            return RAVDESS_MAP.get(parts[2])

    elif mode == "emodb":
        stem = fpath.stem      # e.g. '08a01Wc'
        if len(stem) >= 6:
            return EMODB_MAP.get(stem[5].upper())

    elif mode == "folder":
        # emotion = the immediate parent folder of the audio file
        # works for both:
        #   telugu_ML/telugu/angry/file.wav       -> parent = "angry"
        #   hindi_ML/IESC/Speaker-2/Neutral/file  -> parent = "Neutral"
        return FOLDER_MAP.get(fpath.parent.name.lower().strip())

    return None

# ──────────────────────────────────────────────
# AUDIO UTILS
# ──────────────────────────────────────────────

def load_and_clean(path: Path):
    try:
        y, sr = librosa.load(str(path), sr=TARGET_SR, mono=True)
        y, _  = librosa.effects.trim(y, top_db=TOP_DB)
        dur   = len(y) / TARGET_SR
        if dur < MIN_SECS or dur > MAX_SECS:
            return None
        return y, TARGET_SR
    except Exception as e:
        print(f"  [WARN] {path.name}: {e}")
        return None

def save_wav(y, sr, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), y, sr, subtype="PCM_16")

# ──────────────────────────────────────────────
# PER-LANGUAGE PROCESSOR
# ──────────────────────────────────────────────

def process_language(cfg: dict):
    tag      = cfg["tag"]
    raw_path = cfg["raw"]
    mode     = cfg["mode"]
    out_root = PROC_ROOT / tag

    print(f"\n{'─'*60}")
    print(f"  Language : {tag.upper()}")
    print(f"  Raw path : {raw_path}")
    print(f"  Exists   : {raw_path.exists()}")
    print(f"{'─'*60}")

    if not raw_path.exists():
        print(f"  [SKIP] Folder not found: {raw_path.resolve()}")
        return 0, 0

    audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    all_files  = [p for p in raw_path.rglob("*") if p.suffix.lower() in audio_exts]
    print(f"  Audio files found: {len(all_files)}")

    if not all_files:
        print(f"  [!] No audio files inside {raw_path.resolve()}")
        return 0, 0

    ok = skip = unrecog = 0
    unrecog_samples = set()

    for fpath in tqdm(all_files, desc=f"  {tag}", unit="file"):
        emotion = get_emotion(fpath, mode)

        if emotion is None:
            unrecog_samples.add(fpath.parent.name)
            unrecog += 1
            continue
        if emotion not in KEEP_EMOTIONS:
            skip += 1
            continue

        out_path = out_root / emotion / fpath.name
        if out_path.exists():
            ok += 1
            continue

        result = load_and_clean(fpath)
        if result is None:
            skip += 1
            continue

        save_wav(result[0], result[1], out_path)
        ok += 1

    print(f"  Saved: {ok}  |  Skipped: {skip}  |  Unrecognised: {unrecog}")
    if unrecog_samples:
        print(f"  [!] Unrecognised folder names: {unrecog_samples}")
        print(f"      Add these to FOLDER_MAP in preprocess_ML.py")
    return ok, skip

# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  preprocess_ML.py  --  Audio Standardisation")
    print(f"  Working directory: {Path('.').resolve()}")
    print("=" * 60)
    PROC_ROOT.mkdir(parents=True, exist_ok=True)

    total_ok = total_skip = 0
    for cfg in LANG_CONFIGS:
        ok, skip = process_language(cfg)
        total_ok   += ok
        total_skip += skip

    print("\n" + "=" * 60)
    print(f"  TOTAL -> Saved: {total_ok}  |  Skipped: {total_skip}")
    if total_ok > 0:
        print("  Next step --> python build_csv_ML.py")
    else:
        print("\n  [!] Still 0 files. Check 'Raw path' and 'Exists' lines above.")
        print("  Make sure you run this script FROM the multi-lang folder:")
        print("    cd C:\\Users\\shiva\\Desktop\\multi-lang")
        print("    python preprocess_ML.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
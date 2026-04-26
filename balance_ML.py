"""
balance_ML.py  v2
=================
Uses ALL processed audio — no undersampling.
Handles imbalance via class weights during training (not data removal).
This gives the model 3x more data than v1.
"""

import random
import pandas as pd
from pathlib import Path
from collections import defaultdict

PROC_ROOT = Path("data_ML/processed")
OUT_CSV   = Path("data_ML/balanced_master_ML.csv")
SEED      = 42
LANGUAGES = ["telugu", "hindi", "german", "english"]
KEEP_EMOTIONS = {"neutral", "happy", "sad", "angry", "fearful"}
random.seed(SEED)

def scan_processed():
    data = defaultdict(lambda: defaultdict(list))
    for lang in LANGUAGES:
        lang_path = PROC_ROOT / lang
        if not lang_path.exists():
            continue
        for emotion_dir in sorted(lang_path.iterdir()):
            if not emotion_dir.is_dir(): continue
            files = list(emotion_dir.glob("*.wav"))
            if files:
                data[lang][emotion_dir.name] = files
    return data

def find_common_emotions(data):
    sets = [set(data[l].keys()) for l in LANGUAGES if l in data]
    if not sets: return []
    common = sets[0]
    for s in sets[1:]: common &= s
    return sorted(common & KEEP_EMOTIONS)

def extract_speaker(fpath, lang):
    stem = fpath.stem
    if lang == "english":
        parts = stem.split("-")
        return f"en_{parts[6]}" if len(parts) == 7 else "en_unk"
    elif lang == "german":
        return f"de_{stem[:2]}" if len(stem) >= 2 and stem[:2].isdigit() else "de_unk"
    elif lang == "hindi":
        return f"hi_{stem.split('-')[1]}" if "-" in stem else "hi_unk"
    elif lang == "telugu":
        part = stem.split("_")[0] if "_" in stem else stem[:4]
        return f"te_{part}"
    return "unk"

def main():
    print("=" * 60)
    print("  balance_ML.py v2  --  Use ALL data (class weights strategy)")
    print("=" * 60)

    data   = scan_processed()
    common = find_common_emotions(data)
    print(f"\n  Common emotions: {common}")

    rows = []
    print("\n  ── All available data ────────────────────────────────────")
    for lang in LANGUAGES:
        if lang not in data: continue
        lang_total = 0
        for emo in common:
            if emo not in data[lang]: continue
            files = data[lang][emo]
            lang_total += len(files)
            for fpath in files:
                rows.append({
                    "file_path":  str(fpath.resolve()),
                    "emotion":    emo,
                    "language":   lang,
                    "speaker_id": extract_speaker(fpath, lang),
                })
        print(f"  {lang:10s}: {lang_total} files")

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print(f"\n  ── Language × Emotion pivot ──────────────────────────────")
    pivot = df.groupby(["language","emotion"])["file_path"].count().unstack(fill_value=0)
    print(pivot.to_string())

    print(f"\n  Total files  : {len(df)}  (was ~1280 before, now using all)")
    print(f"  Emotions     : {sorted(df['emotion'].unique())}")
    print(f"\n  CSV saved -> {OUT_CSV}")
    print("  Next --> python split_dataset_ML.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
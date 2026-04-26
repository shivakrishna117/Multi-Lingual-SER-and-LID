"""
extract_features_ML.py
======================
Week 2 - Feature Extraction with Augmentation.

Reads:    data_ML/train_ML.csv  |  val_ML.csv  |  test_ML.csv
Extracts: MFCC (40) + Delta + Delta-Delta + Mel-Spectrogram + Pitch + Energy
Augments: train set only (SpecAugment, speed, noise, pitch-shift)
Saves:    data_ML/features/train_ML.npz
          data_ML/features/val_ML.npz
          data_ML/features/test_ML.npz

Each .npz contains:
  X      -> (N, time_steps, n_features)   float32
  y      -> (N,)                           int   (emotion label)
  lang   -> (N,)                           int   (language label)
  files  -> (N,)                           str

Usage: python extract_features_ML.py
"""

import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from tqdm import tqdm

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
SR          = 16_000
DURATION    = 3.0           # fixed clip length (seconds) — pad/trim to this
N_MFCC      = 40
N_MELS      = 128
HOP_LENGTH  = 512
N_FFT       = 2048
MAX_FRAMES  = int(np.ceil(DURATION * SR / HOP_LENGTH))  # ~94 frames

FEAT_ROOT   = Path("data_ML/features")
SEED        = 42

SPLIT_FILES = {
    "train": Path("data_ML/train_ML.csv"),
    "val":   Path("data_ML/val_ML.csv"),
    "test":  Path("data_ML/test_ML.csv"),
}

np.random.seed(SEED)

# ──────────────────────────────────────────────
# LABEL ENCODERS  (built from full balanced CSV)
# ──────────────────────────────────────────────

def build_encoders(dfs: list) -> tuple[dict, dict]:
    all_emotions  = sorted(set(e for df in dfs for e in df["emotion"]))
    all_languages = sorted(set(l for df in dfs for l in df["language"]))
    emo_enc  = {e: i for i, e in enumerate(all_emotions)}
    lang_enc = {l: i for i, l in enumerate(all_languages)}
    print(f"  Emotion  classes : {emo_enc}")
    print(f"  Language classes : {lang_enc}")
    return emo_enc, lang_enc


# ──────────────────────────────────────────────
# AUDIO LOADING
# ──────────────────────────────────────────────

def load_fixed(path: str) -> np.ndarray:
    """Load audio, fix to exactly DURATION seconds."""
    try:
        y, _ = librosa.load(path, sr=SR, mono=True, duration=DURATION)
    except Exception as e:
        print(f"  [WARN] {Path(path).name}: {e}")
        return np.zeros(int(SR * DURATION))

    target_len = int(SR * DURATION)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y


# ──────────────────────────────────────────────
# AUGMENTATION (train only)
# ──────────────────────────────────────────────

def augment(y: np.ndarray) -> np.ndarray:
    """Randomly apply one of: noise, speed, pitch-shift."""
    choice = np.random.randint(0, 3)

    if choice == 0:
        # Additive Gaussian noise
        noise = np.random.normal(0, 0.005, len(y))
        y = y + noise

    elif choice == 1:
        # Speed perturbation (0.9 – 1.1x), then re-fix length
        rate  = np.random.uniform(0.9, 1.1)
        y     = librosa.effects.time_stretch(y, rate=rate)
        target = int(SR * DURATION)
        if len(y) < target:
            y = np.pad(y, (0, target - len(y)))
        else:
            y = y[:target]

    elif choice == 2:
        # Pitch shift (-2 to +2 semitones)
        steps = np.random.uniform(-2, 2)
        y     = librosa.effects.pitch_shift(y, sr=SR, n_steps=steps)

    return y.astype(np.float32)


# ──────────────────────────────────────────────
# FEATURE EXTRACTION
# ──────────────────────────────────────────────

def extract_features(y: np.ndarray) -> np.ndarray:
    """
    Returns feature matrix of shape (MAX_FRAMES, n_features).

    Features (per frame):
      MFCC          : 40
      Delta-MFCC    : 40
      Delta2-MFCC   : 40
      Mel-spectrogram: 128  (log scale)
      Pitch (f0)    :  1
      RMS energy    :  1
      ─────────────────
      Total         : 250
    """
    # MFCC + deltas
    mfcc   = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC,
                                   n_fft=N_FFT, hop_length=HOP_LENGTH)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    # Mel-spectrogram (log)
    mel    = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS,
                                             n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Pitch (fundamental frequency)
    f0, _, _  = librosa.pyin(y, fmin=librosa.note_to_hz("C2"),
                               fmax=librosa.note_to_hz("C7"),
                               sr=SR, hop_length=HOP_LENGTH)
    f0 = np.nan_to_num(f0, nan=0.0).reshape(1, -1)

    # RMS energy
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)

    # Stack all features: shape (n_features, frames)
    features = np.vstack([mfcc, delta, delta2, mel_db, f0, rms])

    # Transpose -> (frames, n_features)
    features = features.T

    # Pad or trim to MAX_FRAMES
    if features.shape[0] < MAX_FRAMES:
        pad = MAX_FRAMES - features.shape[0]
        features = np.pad(features, ((0, pad), (0, 0)))
    else:
        features = features[:MAX_FRAMES]

    return features.astype(np.float32)


# ──────────────────────────────────────────────
# SPEC-AUGMENT (applied in feature space)
# ──────────────────────────────────────────────

def spec_augment(feat: np.ndarray,
                 time_mask_max: int = 15,
                 freq_mask_max: int = 20) -> np.ndarray:
    """Time and frequency masking on the feature matrix."""
    feat = feat.copy()
    T, F = feat.shape

    # Time mask
    t = np.random.randint(0, time_mask_max)
    t0 = np.random.randint(0, max(1, T - t))
    feat[t0:t0+t, :] = 0

    # Frequency mask
    f = np.random.randint(0, freq_mask_max)
    f0 = np.random.randint(0, max(1, F - f))
    feat[:, f0:f0+f] = 0

    return feat


# ──────────────────────────────────────────────
# PROCESS ONE SPLIT
# ──────────────────────────────────────────────

def process_split(split: str, df: pd.DataFrame,
                  emo_enc: dict, lang_enc: dict,
                  augment_data: bool = False):
    X, y_emo, y_lang, file_list = [], [], [], []
    errors = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  {split}"):
        audio = load_fixed(row["file_path"])
        if audio is None or np.max(np.abs(audio)) < 1e-6:
            errors += 1
            continue

        # Original sample
        feat = extract_features(audio)
        if augment_data:
            feat = spec_augment(feat)

        X.append(feat)
        y_emo.append(emo_enc[row["emotion"]])
        y_lang.append(lang_enc[row["language"]])
        file_list.append(row["file_path"])

        # Augmented copy (train only — doubles train size)
        if augment_data:
            aug_audio = augment(audio)
            aug_feat  = extract_features(aug_audio)
            aug_feat  = spec_augment(aug_feat)
            X.append(aug_feat)
            y_emo.append(emo_enc[row["emotion"]])
            y_lang.append(lang_enc[row["language"]])
            file_list.append(row["file_path"] + "__aug")

    X      = np.array(X,     dtype=np.float32)
    y_emo  = np.array(y_emo, dtype=np.int32)
    y_lang = np.array(y_lang,dtype=np.int32)

    if errors:
        print(f"  [WARN] {errors} files could not be loaded.")

    return X, y_emo, y_lang, file_list


# ──────────────────────────────────────────────
# NORMALISE (z-score per feature dimension)
# ──────────────────────────────────────────────

def fit_normalise(X_train: np.ndarray):
    """Compute mean/std from train set."""
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std  = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    return mean, std


def apply_normalise(X: np.ndarray, mean, std) -> np.ndarray:
    return (X - mean) / std


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  extract_features_ML.py  --  Feature Extraction")
    print("=" * 60)

    # Load split CSVs
    dfs = {}
    for split, csv_path in SPLIT_FILES.items():
        if not csv_path.exists():
            print(f"  [!] {csv_path} not found. Run split_dataset_ML.py first.")
            return
        dfs[split] = pd.read_csv(csv_path)
        print(f"  {split:6s}: {len(dfs[split])} rows")

    # Build label encoders
    print()
    emo_enc, lang_enc = build_encoders(list(dfs.values()))

    # Save encoder mappings for use during inference
    enc_path = FEAT_ROOT / "encoders_ML.npz"
    FEAT_ROOT.mkdir(parents=True, exist_ok=True)
    np.savez(enc_path,
             emotions=list(emo_enc.keys()),
             languages=list(lang_enc.keys()))
    print(f"  Encoders saved -> {enc_path}")

    # Extract features
    print("\n  Extracting features (this may take a few minutes)...")
    results = {}
    for split, df in dfs.items():
        is_train = (split == "train")
        X, y_emo, y_lang, files = process_split(
            split, df, emo_enc, lang_enc, augment_data=is_train
        )
        results[split] = (X, y_emo, y_lang, files)
        print(f"  {split:6s}: X={X.shape}  y_emo={y_emo.shape}")

    # Normalise (fit on train, apply to all)
    print("\n  Normalising features...")
    X_train = results["train"][0]
    mean, std = fit_normalise(X_train)
    np.save(FEAT_ROOT / "norm_mean_ML.npy", mean)
    np.save(FEAT_ROOT / "norm_std_ML.npy",  std)
    print(f"  Norm stats saved -> {FEAT_ROOT}/norm_mean_ML.npy")

    # Save .npz files
    for split, (X, y_emo, y_lang, files) in results.items():
        X_norm = apply_normalise(X, mean, std)
        out    = FEAT_ROOT / f"{split}_ML.npz"
        np.savez_compressed(out,
                            X=X_norm,
                            y_emotion=y_emo,
                            y_language=y_lang,
                            files=np.array(files))
        print(f"  Saved {out}  (X={X_norm.shape})")

    print("\n  ── Feature summary ───────────────────────────────────")
    X_sample = results["train"][0]
    print(f"  Shape      : {X_sample.shape}")
    print(f"  Time steps : {X_sample.shape[1]}")
    print(f"  Features   : {X_sample.shape[2]}  "
          f"(MFCC40 + Delta40 + Delta240 + Mel128 + Pitch1 + RMS1)")
    print(f"  Emotions   : {list(emo_enc.keys())}")
    print(f"  Languages  : {list(lang_enc.keys())}")
    print(f"\n  Next step --> python train_ser_ML.py  (Week 3)")
    print("=" * 60)


if __name__ == "__main__":
    main()
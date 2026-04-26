"""
evaluate_ML.py
==============
Week 5 - Full Evaluation: per-language SER breakdown + combined report.

Outputs:
  results_ML/eval_per_language_ML.png   <- per-language confusion matrices
  results_ML/eval_summary_ML.txt        <- full summary table
  results_ML/eval_combined_ML.png       <- language x emotion heatmap

Usage: python evaluate_ML.py
"""

import os
import numpy as np
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

FEAT_ROOT   = Path("data_ML/features")
MODEL_DIR   = Path("models_ML")
RESULTS_DIR = Path("results_ML")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 7]


def load_split(name):
    d = np.load(FEAT_ROOT / f"{name}_ML.npz", allow_pickle=True)
    return (d["X"].astype(np.float32),
            d["y_emotion"].astype(np.int32),
            d["y_language"].astype(np.int32),
            d["files"])


def load_encoders():
    d = np.load(FEAT_ROOT / "encoders_ML.npz", allow_pickle=True)
    return list(d["emotions"]), list(d["languages"])


def load_ser_ensemble():
    models = []
    for seed in SEEDS:
        p = MODEL_DIR / "ser" / f"ser_seed{seed}_ML.h5"
        if p.exists():
            models.append(keras.models.load_model(str(p), compile=False))
    if not models:
        models = [keras.models.load_model(
            str(MODEL_DIR / "ser" / "best_ser_ML.h5"), compile=False)]
    print(f"  Loaded {len(models)} SER model(s)")
    return models


def load_lid():
    return keras.models.load_model(
        str(MODEL_DIR / "lid" / "best_lid_ML.h5"), compile=False)


def ensemble_predict(models, X):
    probs = np.mean([m.predict(X, verbose=0) for m in models], axis=0)
    return np.argmax(probs, axis=1), probs


def main():
    print("=" * 60)
    print("  evaluate_ML.py  --  Week 5 Full Evaluation")
    print("=" * 60)

    emotion_labels, lang_labels = load_encoders()

    print("\n  Loading test data...")
    X_test, y_emo, y_lang, files = load_split("test")
    print(f"  Test samples: {len(X_test)}")

    print("\n  Loading models...")
    ser_models = load_ser_ensemble()
    lid_model  = load_lid()

    # ── Overall SER ───────────────────────────────────────────────
    print("\n  Running SER ensemble prediction...")
    y_ser_pred, ser_probs = ensemble_predict(ser_models, X_test)
    ser_acc = np.mean(y_ser_pred == y_emo)
    report  = classification_report(y_emo, y_ser_pred,
                                    target_names=emotion_labels, output_dict=True)
    ser_uar = np.mean([report[e]["recall"] for e in emotion_labels if e in report])

    # ── Overall LID ───────────────────────────────────────────────
    print("  Running LID prediction...")
    lid_probs  = lid_model.predict(X_test, verbose=0)
    y_lid_pred = np.argmax(lid_probs, axis=1)
    lid_acc    = np.mean(y_lid_pred == y_lang)

    # ── Per-language SER breakdown ────────────────────────────────
    print("\n  ── Per-language SER breakdown ──────────────────────────")
    lang_results = {}
    for li, lang in enumerate(lang_labels):
        mask = (y_lang == li)
        if mask.sum() == 0:
            continue
        Xl   = X_test[mask]
        yl   = y_emo[mask]
        _, probs_l = ensemble_predict(ser_models, Xl)
        pred_l = np.argmax(probs_l, axis=1)
        acc_l  = np.mean(pred_l == yl)
        rep_l  = classification_report(yl, pred_l,
                                       target_names=emotion_labels,
                                       output_dict=True, zero_division=0)
        uar_l  = np.mean([rep_l[e]["recall"]
                          for e in emotion_labels if e in rep_l])
        lang_results[lang] = {"acc": acc_l, "uar": uar_l,
                               "y_true": yl, "y_pred": pred_l,
                               "n": int(mask.sum())}
        print(f"  {lang:10s}  n={mask.sum():3d}  "
              f"acc={acc_l*100:.1f}%  uar={uar_l*100:.1f}%")

    # ── Save summary txt ─────────────────────────────────────────
    summary_path = RESULTS_DIR / "eval_summary_ML.txt"
    with open(summary_path, "w") as f:
        f.write("=" * 55 + "\n")
        f.write("  MULTILINGUAL SER — EVALUATION SUMMARY\n")
        f.write("  Architecture: Multi-Head Attention + CRNN + BiGRU\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"  Overall SER Accuracy : {ser_acc*100:.1f}%\n")
        f.write(f"  Overall SER UAR      : {ser_uar*100:.1f}%\n")
        f.write(f"  LID Accuracy         : {lid_acc*100:.1f}%\n\n")
        f.write("  Per-Language SER:\n")
        for lang, r in lang_results.items():
            f.write(f"    {lang:10s}  acc={r['acc']*100:.1f}%  "
                    f"uar={r['uar']*100:.1f}%  (n={r['n']})\n")
        f.write("\n  Per-Emotion (overall):\n")
        f.write(classification_report(y_emo, y_ser_pred,
                                      target_names=emotion_labels))
    print(f"\n  Summary -> {summary_path}")

    # ── Plot: per-language confusion matrices ─────────────────────
    n_langs = len(lang_results)
    fig, axes = plt.subplots(1, n_langs, figsize=(5 * n_langs, 5))
    if n_langs == 1:
        axes = [axes]
    for ax, (lang, r) in zip(axes, lang_results.items()):
        cm = confusion_matrix(r["y_true"], r["y_pred"], normalize="true")
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=emotion_labels,
                    yticklabels=emotion_labels, ax=ax, vmin=0, vmax=1)
        ax.set_title(f"{lang.upper()}\nacc={r['acc']*100:.1f}%  uar={r['uar']*100:.1f}%")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.suptitle("Per-Language SER Confusion Matrices", fontsize=13, y=1.02)
    plt.tight_layout()
    ppath = RESULTS_DIR / "eval_per_language_ML.png"
    plt.savefig(ppath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Per-language plot -> {ppath}")

    # ── Plot: language x emotion accuracy heatmap ─────────────────
    emo_by_lang = np.zeros((n_langs, len(emotion_labels)))
    for li, (lang, r) in enumerate(lang_results.items()):
        rep = classification_report(r["y_true"], r["y_pred"],
                                    target_names=emotion_labels,
                                    output_dict=True, zero_division=0)
        for ei, emo in enumerate(emotion_labels):
            emo_by_lang[li, ei] = rep.get(emo, {}).get("recall", 0)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(emo_by_lang, annot=True, fmt=".2f", cmap="YlGnBu",
                xticklabels=emotion_labels,
                yticklabels=list(lang_results.keys()),
                vmin=0, vmax=1, ax=ax)
    ax.set_title("Recall per Language × Emotion")
    ax.set_xlabel("Emotion"); ax.set_ylabel("Language")
    plt.tight_layout()
    hpath = RESULTS_DIR / "eval_combined_ML.png"
    plt.savefig(hpath, dpi=150)
    plt.close()
    print(f"  Combined heatmap -> {hpath}")

    print("\n" + "=" * 60)
    print(f"  SER  Accuracy : {ser_acc*100:.1f}%   UAR : {ser_uar*100:.1f}%")
    print(f"  LID  Accuracy : {lid_acc*100:.1f}%")
    print("\n  Next --> python app_ML/app_ML.py   (Week 6 Streamlit)")
    print("=" * 60)


if __name__ == "__main__":
    main()
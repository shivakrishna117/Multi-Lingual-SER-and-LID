"""
train_lid_ML.py
===============
Week 3 - Train the Language Identification (LID) Model.

Lightweight CNN-only model — fast inference (<0.5s on CPU).
Input:  same .npz features as SER
Output: models_ML/lid/best_lid_ML.h5
        results_ML/lid_report_ML.txt

Languages: english | german | hindi | telugu  (4 classes)

Usage: python train_lid_ML.py
"""

import os
import numpy as np
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
FEAT_ROOT   = Path("data_ML/features")
MODEL_DIR   = Path("models_ML/lid")
RESULTS_DIR = Path("results_ML")

EPOCHS      = 60
BATCH_SIZE  = 32
LR          = 1e-3
PATIENCE    = 12
SEED        = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────

def load_split(name: str):
    path = FEAT_ROOT / f"{name}_ML.npz"
    d    = np.load(path, allow_pickle=True)
    X    = d["X"].astype(np.float32)
    y    = d["y_language"].astype(np.int32)   # <-- language labels
    return X, y


def load_encoders():
    d = np.load(FEAT_ROOT / "encoders_ML.npz", allow_pickle=True)
    return list(d["emotions"]), list(d["languages"])


# ──────────────────────────────────────────────
# MODEL: Lightweight CNN (faster than BiGRU)
# ──────────────────────────────────────────────

def build_lid_model(input_shape: tuple, n_classes: int) -> keras.Model:
    """
    Lightweight CNN for Language ID.
    Smaller than SER model → fast inference on CPU.
    Input:  (batch, 94, 250)
    Output: (batch, n_lang)
    """
    inp = keras.Input(shape=input_shape, name="input")

    # ── CNN block ──────────────────────────────────────────────────
    x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)

    # Global average pooling — keeps model small
    x = layers.GlobalAveragePooling1D()(x)

    # ── Dense head ─────────────────────────────────────────────────
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation="softmax", name="language")(x)

    model = keras.Model(inputs=inp, outputs=out, name="CNN_LID")
    return model


# ──────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────

def get_class_weights(y, n_classes):
    counts  = np.bincount(y, minlength=n_classes)
    total   = len(y)
    return {i: total / (n_classes * max(c, 1)) for i, c in enumerate(counts)}


def train(X_train, y_train, X_val, y_val, n_classes: int):
    model = build_lid_model(input_shape=X_train.shape[1:], n_classes=n_classes)
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    cb = [
        callbacks.EarlyStopping(
            monitor="val_accuracy", patience=PATIENCE,
            restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=6,
            min_lr=1e-6, verbose=1),
        callbacks.ModelCheckpoint(
            filepath=str(MODEL_DIR / "best_lid_ML.h5"),
            monitor="val_accuracy", save_best_only=True,
            save_format="h5",
            verbose=1),
    ]

    cw = get_class_weights(y_train, n_classes)
    print(f"\n  Class weights: {cw}")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=cw,
        callbacks=cb,
        verbose=1
    )
    return model, history


# ──────────────────────────────────────────────
# EVALUATION
# ──────────────────────────────────────────────

def evaluate(model, X_test, y_test, lang_labels: list):
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred      = np.argmax(y_pred_prob, axis=1)

    acc    = np.mean(y_pred == y_test)
    report = classification_report(y_test, y_pred,
                                   target_names=lang_labels,
                                   output_dict=True)
    uar = np.mean([report[l]["recall"] for l in lang_labels if l in report])

    print(f"\n  Test Accuracy : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  UAR           : {uar:.4f}  ({uar*100:.1f}%)")
    print("\n" + classification_report(y_test, y_pred, target_names=lang_labels))

    # Save report
    rpath = RESULTS_DIR / "lid_report_ML.txt"
    with open(rpath, "w") as f:
        f.write(f"LID Test Accuracy : {acc:.4f}\n")
        f.write(f"LID UAR           : {uar:.4f}\n\n")
        f.write(classification_report(y_test, y_pred, target_names=lang_labels))
    print(f"  Report saved -> {rpath}")
    return y_pred, acc, uar


def plot_confusion_matrix(y_test, y_pred, labels, title, save_path):
    cm = confusion_matrix(y_test, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Greens",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved -> {save_path}")


def plot_history(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history["accuracy"],     label="train acc")
    ax1.plot(history.history["val_accuracy"], label="val acc")
    ax1.set_title("LID Accuracy")
    ax1.legend(); ax1.set_xlabel("Epoch")
    ax2.plot(history.history["loss"],     label="train loss")
    ax2.plot(history.history["val_loss"], label="val loss")
    ax2.set_title("LID Loss")
    ax2.legend(); ax2.set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Training curves saved -> {save_path}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  train_lid_ML.py  --  CNN Language ID Training")
    print("=" * 60)

    gpus = tf.config.list_physical_devices("GPU")
    print(f"\n  GPUs available: {len(gpus)}")
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)

    print("\n  Loading features...")
    X_train, y_train = load_split("train")
    X_val,   y_val   = load_split("val")
    X_test,  y_test  = load_split("test")

    print(f"  train: {X_train.shape}  y={y_train.shape}")
    print(f"  val  : {X_val.shape}")
    print(f"  test : {X_test.shape}")

    emotion_labels, lang_labels = load_encoders()
    n_classes = len(lang_labels)
    print(f"\n  Languages ({n_classes}): {lang_labels}")

    print(f"\n  Training CNN LID model...")
    model, history = train(X_train, y_train, X_val, y_val, n_classes)

    # Save history
    np.savez(MODEL_DIR / "history_lid_ML.npz",
             **{k: np.array(v) for k, v in history.history.items()})

    print("\n  Evaluating on test set...")
    y_pred, acc, uar = evaluate(model, X_test, y_test, lang_labels)

    plot_history(history, RESULTS_DIR / "lid_training_curves_ML.png")
    plot_confusion_matrix(y_test, y_pred, lang_labels,
                          f"LID Confusion Matrix (Acc={acc:.2f})",
                          RESULTS_DIR / "lid_confusion_ML.png")

    print("\n" + "=" * 60)
    print(f"  DONE")
    print(f"  Model   -> {MODEL_DIR}/best_lid_ML.h5")
    print(f"  Test Accuracy : {acc*100:.1f}%")
    print(f"  UAR           : {uar*100:.1f}%")
    print("\n  Both models trained! Next --> python evaluate_ML.py  (Week 5)")
    print("=" * 60)


if __name__ == "__main__":
    main()
"""
train_ser_ML.py  v3
===================
Architecture: Multi-Head Attention + CRNN + BiGRU

Pipeline:
  Input (94, 250)
    → CRNN Block  (Conv1D + BatchNorm + MaxPool, repeated)
    → Multi-Head Self-Attention (Transformer-style, 8 heads)
    → BiGRU  (bidirectional, 2 layers)
    → Dense Head

Key improvements over v2:
  - Multi-Head Attention (8 heads) instead of single-head
  - CRNN: Conv layers feed directly into BiGRU (true CRNN)
  - Uses ALL data with class weights (not hard undersampling)
  - Focal loss gamma=2
  - Label smoothing on top of focal loss
  - Ensemble of 3 seeds

Usage: python train_ser_ML.py
"""

import os
import numpy as np
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers
import tensorflow.keras.backend as K
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
FEAT_ROOT   = Path("data_ML/features")
MODEL_DIR   = Path("models_ML/ser")
RESULTS_DIR = Path("results_ML")

EPOCHS      = 150
BATCH_SIZE  = 32
LR          = 3e-4
PATIENCE    = 25
NUM_HEADS   = 8      # Multi-head attention heads
KEY_DIM     = 32     # Dimension per attention head
SEEDS       = [42, 123, 7]

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────

def load_split(name):
    d = np.load(FEAT_ROOT / f"{name}_ML.npz", allow_pickle=True)
    return d["X"].astype(np.float32), d["y_emotion"].astype(np.int32)

def load_encoders():
    d = np.load(FEAT_ROOT / "encoders_ML.npz", allow_pickle=True)
    return list(d["emotions"]), list(d["languages"])


# ──────────────────────────────────────────────
# FOCAL LOSS
# ──────────────────────────────────────────────

def focal_loss(gamma=2.0, alpha=0.25, label_smoothing=0.1):
    def loss_fn(y_true, y_pred):
        # Apply label smoothing
        n_cls   = tf.cast(tf.shape(y_true)[-1], tf.float32)
        y_true  = y_true * (1 - label_smoothing) + label_smoothing / n_cls
        y_pred  = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        ce      = -y_true * K.log(y_pred)
        weight  = alpha * y_true * K.pow(1 - y_pred, gamma)
        return K.mean(K.sum(weight * ce, axis=-1))
    return loss_fn


# ──────────────────────────────────────────────
# MIXUP
# ──────────────────────────────────────────────

def mixup_batch(X, y_oh, alpha=0.4):
    n   = len(X)
    lam = np.random.beta(alpha, alpha, n).astype(np.float32)
    lam = np.maximum(lam, 1 - lam)
    idx = np.random.permutation(n)
    X_mix = lam[:,None,None] * X + (1-lam[:,None,None]) * X[idx]
    y_mix = lam[:,None]      * y_oh + (1-lam[:,None])   * y_oh[idx]
    return X_mix, y_mix


# ──────────────────────────────────────────────
# MULTI-HEAD ATTENTION + CRNN + BiGRU MODEL
# ──────────────────────────────────────────────

def crnn_block(x, filters, kernel_size, pool_size=2, dropout=0.25):
    """Convolutional block: Conv → BN → ReLU → Pool → Dropout"""
    x = layers.Conv1D(filters, kernel_size, padding="same",
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    if pool_size > 1:
        x = layers.MaxPooling1D(pool_size)(x)
    x = layers.Dropout(dropout)(x)
    return x


def transformer_block(x, num_heads, key_dim, ff_dim, dropout=0.2):
    """
    Transformer encoder block:
      Multi-Head Self-Attention → Add & Norm → FFN → Add & Norm
    """
    # Multi-Head Self-Attention
    attn_out = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=dropout
    )(x, x)                                   # self-attention: Q=K=V=x
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn_out)  # residual

    # Feed-Forward Network
    ffn_out = layers.Dense(ff_dim, activation="relu")(x)
    ffn_out = layers.Dropout(dropout)(ffn_out)
    ffn_out = layers.Dense(x.shape[-1])(ffn_out)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_out)   # residual

    return x


def build_model(input_shape, n_classes):
    """
    Full architecture:
      Input → CRNN blocks → Multi-Head Attention (8 heads) → BiGRU → Dense
    """
    inp = keras.Input(shape=input_shape)   # (94, 250)

    # ── CRNN Block 1 ──────────────────────────────────────────────
    x = crnn_block(inp, filters=64,  kernel_size=7, pool_size=2, dropout=0.2)
    # (47, 64)

    # ── CRNN Block 2 ──────────────────────────────────────────────
    x = crnn_block(x,   filters=128, kernel_size=5, pool_size=2, dropout=0.25)
    # (23, 128)

    # ── CRNN Block 3 ──────────────────────────────────────────────
    x = crnn_block(x,   filters=256, kernel_size=3, pool_size=1, dropout=0.25)
    # (23, 256)

    # ── Multi-Head Self-Attention (8 heads × 32 dim = 256) ────────
    # Positional encoding (simple learned)
    pos_emb = layers.Embedding(input_dim=x.shape[1], output_dim=x.shape[-1])
    positions = tf.range(start=0, limit=x.shape[1], delta=1)
    x = x + pos_emb(positions)

    x = transformer_block(x,
                          num_heads=NUM_HEADS,
                          key_dim=KEY_DIM,
                          ff_dim=512,
                          dropout=0.2)
    # Optional: stack a second transformer block
    x = transformer_block(x,
                          num_heads=NUM_HEADS,
                          key_dim=KEY_DIM,
                          ff_dim=512,
                          dropout=0.2)
    # (23, 256)

    # ── BiGRU ─────────────────────────────────────────────────────
    x = layers.Bidirectional(
            layers.GRU(128, return_sequences=True,
                       dropout=0.3, recurrent_dropout=0.1))(x)
    x = layers.Bidirectional(
            layers.GRU(64, return_sequences=False,
                       dropout=0.3))(x)
    # (256,)

    # ── Dense Head ────────────────────────────────────────────────
    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)

    model = keras.Model(inputs=inp, outputs=out,
                        name="MHA_CRNN_BiGRU")
    return model


# ──────────────────────────────────────────────
# COSINE ANNEALING LR
# ──────────────────────────────────────────────

class CosineAnnealingLR(callbacks.Callback):
    def __init__(self, lr_max, lr_min=1e-6, T_max=50):
        super().__init__()
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.T_max  = T_max

    def on_epoch_begin(self, epoch, logs=None):
        t  = epoch % self.T_max
        lr = self.lr_min + 0.5*(self.lr_max - self.lr_min)*(1 + np.cos(np.pi*t/self.T_max))
        K.set_value(self.model.optimizer.learning_rate, float(lr))


# ──────────────────────────────────────────────
# CLASS WEIGHTS (replaces hard undersampling)
# ──────────────────────────────────────────────

def get_class_weights(y, n_classes):
    counts = np.bincount(y, minlength=n_classes).astype(float)
    total  = float(len(y))
    # Inverse frequency weighting
    weights = total / (n_classes * np.maximum(counts, 1))
    return {i: w for i, w in enumerate(weights)}


# ──────────────────────────────────────────────
# TRAIN ONE SEED
# ──────────────────────────────────────────────

def train_single(X_train, y_train, X_val, y_val,
                 n_classes, seed, model_path):
    tf.random.set_seed(seed)
    np.random.seed(seed)

    model = build_model(X_train.shape[1:], n_classes)
    if seed == SEEDS[0]:
        model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(LR),
        loss=focal_loss(gamma=2.0, alpha=0.25, label_smoothing=0.1),
        metrics=["accuracy"]
    )

    y_oh    = keras.utils.to_categorical(y_train, n_classes)
    y_val_oh = keras.utils.to_categorical(y_val,  n_classes)
    cw      = get_class_weights(y_train, n_classes)
    print(f"  Class weights: { {k: round(v,2) for k,v in cw.items()} }")

    def data_gen():
        while True:
            idx = np.random.permutation(len(X_train))
            for s in range(0, len(X_train), BATCH_SIZE):
                b = idx[s:s+BATCH_SIZE]
                Xb, yb = mixup_batch(X_train[b], y_oh[b], alpha=0.4)
                yield Xb, yb

    steps = max(1, len(X_train) // BATCH_SIZE)

    cb = [
        callbacks.EarlyStopping(
            monitor="val_accuracy", patience=PATIENCE,
            restore_best_weights=True, verbose=0),
        callbacks.ModelCheckpoint(
            filepath=model_path, monitor="val_accuracy",
            save_best_only=True, save_format="h5", verbose=0),
        CosineAnnealingLR(lr_max=LR, lr_min=1e-6, T_max=50),
    ]

    history = model.fit(
        data_gen(),
        steps_per_epoch=steps,
        validation_data=(X_val, y_val_oh),
        epochs=EPOCHS,
        callbacks=cb,
        verbose=1
    )
    return model, history


# ──────────────────────────────────────────────
# ENSEMBLE
# ──────────────────────────────────────────────

def ensemble_predict(models, X):
    probs = np.mean([m.predict(X, verbose=0) for m in models], axis=0)
    return np.argmax(probs, axis=1)


# ──────────────────────────────────────────────
# EVALUATION
# ──────────────────────────────────────────────

def evaluate(y_pred, y_test, emotion_labels):
    acc    = np.mean(y_pred == y_test)
    report = classification_report(y_test, y_pred,
                                   target_names=emotion_labels, output_dict=True)
    uar    = np.mean([report[e]["recall"] for e in emotion_labels if e in report])

    print(f"\n  Test Accuracy : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  UAR           : {uar:.4f}  ({uar*100:.1f}%)")
    print("\n" + classification_report(y_test, y_pred, target_names=emotion_labels))

    with open(RESULTS_DIR / "ser_report_ML.txt", "w") as f:
        f.write(f"SER Test Accuracy : {acc:.4f}\n")
        f.write(f"SER UAR           : {uar:.4f}\n\n")
        f.write(classification_report(y_test, y_pred, target_names=emotion_labels))
    return acc, uar


def plot_confusion(y_test, y_pred, labels, uar, path):
    cm = confusion_matrix(y_test, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"MHA-CRNN-BiGRU Ensemble  (UAR={uar:.2f})")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"  Confusion matrix -> {path}")


def plot_histories(histories, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["royalblue", "tomato", "seagreen"]
    for i, h in enumerate(histories):
        axes[0].plot(h.history["val_accuracy"], color=colors[i],
                     label=f"seed {SEEDS[i]}", alpha=0.8)
        axes[1].plot(h.history["val_loss"], color=colors[i],
                     label=f"seed {SEEDS[i]}", alpha=0.8)
    for ax, title in zip(axes, ["Val Accuracy", "Val Loss"]):
        ax.set_title(title); ax.legend(); ax.set_xlabel("Epoch")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"  Training curves -> {path}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  train_ser_ML.py v3")
    print("  Architecture: Multi-Head Attention + CRNN + BiGRU")
    print("=" * 60)

    gpus = tf.config.list_physical_devices("GPU")
    print(f"\n  GPUs: {len(gpus)}")
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)

    print("\n  Loading features...")
    X_train, y_train = load_split("train")
    X_val,   y_val   = load_split("val")
    X_test,  y_test  = load_split("test")
    print(f"  train={X_train.shape}  val={X_val.shape}  test={X_test.shape}")

    emotion_labels, _ = load_encoders()
    n_classes = len(emotion_labels)
    print(f"  Emotions: {emotion_labels}")

    # Train 3 models
    models, histories = [], []
    for i, seed in enumerate(SEEDS):
        print(f"\n  {'='*55}")
        print(f"  Model {i+1}/{len(SEEDS)}  seed={seed}  Architecture: MHA-CRNN-BiGRU")
        print(f"  {'='*55}")
        mpath = str(MODEL_DIR / f"ser_seed{seed}_ML.h5")
        m, h  = train_single(X_train, y_train, X_val, y_val,
                              n_classes, seed, mpath)
        models.append(m); histories.append(h)
        vacc = np.mean(np.argmax(m.predict(X_val, verbose=0), axis=1) == y_val)
        print(f"  Seed {seed} best val accuracy: {vacc*100:.1f}%")

    # Per-seed test
    print("\n  ── Per-seed test accuracy ──────────────────────────────")
    for m, seed in zip(models, SEEDS):
        p   = np.argmax(m.predict(X_test, verbose=0), axis=1)
        acc = np.mean(p == y_test)
        print(f"  Seed {seed}: {acc*100:.1f}%")

    # Ensemble
    print("\n  ── Ensemble (soft voting) ──────────────────────────────")
    y_pred = ensemble_predict(models, X_test)
    acc, uar = evaluate(y_pred, y_test, emotion_labels)

    # Save best model
    val_accs = [np.mean(np.argmax(m.predict(X_val,verbose=0),axis=1)==y_val)
                for m in models]
    best = int(np.argmax(val_accs))
    models[best].save(str(MODEL_DIR / "best_ser_ML.h5"), save_format="h5")

    plot_histories(histories, RESULTS_DIR / "ser_curves_ML.png")
    plot_confusion(y_test, y_pred, emotion_labels, uar,
                   RESULTS_DIR / "ser_confusion_ML.png")

    print("\n" + "=" * 60)
    print(f"  Architecture  : Multi-Head Attention + CRNN + BiGRU")
    print(f"  Ensemble Acc  : {acc*100:.1f}%")
    print(f"  Ensemble UAR  : {uar*100:.1f}%")
    print(f"  Models saved  -> {MODEL_DIR}/")
    print("\n  NOTE: Re-run balance_ML.py + split + extract first")
    print("        to use ALL data (not just balanced subset)")
    print("\n  Next --> python train_lid_ML.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
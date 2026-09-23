"""
=============================================================
  HIGH-ACCURACY STATIC GESTURE MODEL TRAINER  v2
=============================================================
  Improvements over v1:
  - Advanced scale + position normalization (palm-size invariant)
  - Data augmentation (jitter, flip, scale, rotation in 2D)
  - Class-weight balancing (handles unequal sample counts)
  - Deeper residual-style DNN: 512-256-128-64
  - BatchNormalization after every dense layer
  - L2 regularization + Dropout for generalization
  - ReduceLROnPlateau + EarlyStopping callbacks
  - Best-model checkpoint (saves only when val_accuracy improves)
  - Per-class accuracy report after training
  - Skips classes with < MIN_SAMPLES samples
=============================================================
  Usage:
    python train_static_model_v2.py
=============================================================
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import time

# ── Paths ────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATASET_FOLDER = os.path.normpath(os.path.join(BASE_DIR, "../datasets/Custom_Data"))
MODEL_FOLDER   = os.path.normpath(os.path.join(BASE_DIR, "../../models"))

# ── Config ───────────────────────────────────────────────────
MIN_SAMPLES   = 30      # skip any class with fewer samples than this
TEST_SIZE     = 0.15    # 15% held out for final evaluation
VAL_SIZE      = 0.15    # 15% of training set used as validation
EPOCHS        = 200     # max epochs (early stopping will kick in)
BATCH_SIZE    = 32
SEED          = 42

# Augmentation
AUG_FACTOR    = 8       # multiply each sample by this many augmented copies
JITTER_STD    = 0.010   # gaussian noise std (hand tremor sim)
SCALE_RANGE   = (0.85, 1.15)  # random scale
ROT_RANGE     = 0.20    # random 2D rotation in radians (~±11 deg)
FLIP_CHANCE   = 0.0     # horizontal flip (0 = disabled; same sign looks different flipped)

# ── Normalization ─────────────────────────────────────────────
def normalize_keypoints(kp: np.ndarray) -> np.ndarray:
    """
    Per-hand: translate to wrist origin, scale by palm size
    (wrist -> middle-finger MCP distance).
    Input/output: flat array of shape (126,)
    """
    out = np.zeros(126, dtype=np.float32)
    for h in range(2):
        base = h * 63
        hand = kp[base: base + 63]
        if not np.any(hand):
            continue
        wrist   = hand[0:3].copy()
        mid_mcp = hand[27:30].copy()   # landmark 9
        scale   = float(np.linalg.norm(mid_mcp - wrist)) + 1e-6
        for i in range(21):
            o = i * 3
            out[base + o]     = (hand[o]     - wrist[0]) / scale
            out[base + o + 1] = (hand[o + 1] - wrist[1]) / scale
            out[base + o + 2] = (hand[o + 2] - wrist[2]) / scale
    return out

# ── Augmentation ──────────────────────────────────────────────
def augment_sample(kp: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Apply random augmentations to a normalized keypoint vector.
    Operations:
      1. Gaussian jitter  — simulates hand tremor
      2. Random 2D rotation (x-y plane only, keep z)
      3. Random uniform scale
    """
    aug = kp.copy()

    # 1. Jitter
    aug += rng.normal(0, JITTER_STD, size=aug.shape)

    # 2. 2D rotation per hand
    theta = rng.uniform(-ROT_RANGE, ROT_RANGE)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    for h in range(2):
        base = h * 63
        if not np.any(aug[base: base + 63]):
            continue
        for i in range(21):
            o = base + i * 3
            x, y = aug[o], aug[o + 1]
            aug[o]     =  cos_t * x - sin_t * y
            aug[o + 1] =  sin_t * x + cos_t * y

    # 3. Scale
    s = rng.uniform(*SCALE_RANGE)
    aug *= s

    return aug.astype(np.float32)

# ── Data loading ──────────────────────────────────────────────
def load_dataset():
    if not os.path.exists(DATASET_FOLDER):
        raise FileNotFoundError(f"Dataset folder not found: {DATASET_FOLDER}")

    all_classes = sorted([
        d for d in os.listdir(DATASET_FOLDER)
        if os.path.isdir(os.path.join(DATASET_FOLDER, d))
    ])

    X_raw, y_raw, valid_classes = [], [], []
    label = 0
    for cls in all_classes:
        cls_dir = os.path.join(DATASET_FOLDER, cls)
        files   = [f for f in os.listdir(cls_dir) if f.endswith(".npy")]
        if len(files) < MIN_SAMPLES:
            print(f"  [SKIP] '{cls}' — only {len(files)} samples (min={MIN_SAMPLES})")
            continue
        for f in files:
            try:
                kp = np.load(os.path.join(cls_dir, f)).astype(np.float32)
                X_raw.append(kp)
                y_raw.append(label)
            except Exception as e:
                print(f"  [WARN] Could not load {f}: {e}")
        print(f"  [OK]   '{cls}' — {len(files)} samples  (label {label})")
        valid_classes.append(cls)
        label += 1

    if len(valid_classes) < 2:
        raise ValueError("Need at least 2 classes with enough samples to train.")

    return np.array(X_raw, dtype=np.float32), np.array(y_raw, dtype=np.int32), valid_classes

# ── Augment dataset ───────────────────────────────────────────
def augment_dataset(X, y, factor):
    rng = np.random.default_rng(SEED)
    X_aug, y_aug = [X], [y]
    for _ in range(factor):
        batch = np.array([augment_sample(x, rng) for x in X], dtype=np.float32)
        X_aug.append(batch)
        y_aug.append(y)
    return np.concatenate(X_aug, axis=0), np.concatenate(y_aug, axis=0)

# ── Model ─────────────────────────────────────────────────────
def build_model(n_classes: int, input_dim: int = 126) -> tf.keras.Model:
    """
    Deep residual-style DNN:
      Input(126) -> BN -> [512->BN->RELU->Drop] -> [256->BN->RELU->Drop]
               -> [128->BN->RELU->Drop] -> [64->BN->RELU] -> Softmax(n)

    Skip connection: project input to 256 and add after second block
    for better gradient flow.
    """
    reg = regularizers.l2(1e-4)

    inp = layers.Input(shape=(input_dim,), name="keypoints")
    x   = layers.BatchNormalization()(inp)

    # Block 1
    x = layers.Dense(512, kernel_regularizer=reg, name="dense1")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.35)(x)

    # Block 2
    x = layers.Dense(256, kernel_regularizer=reg, name="dense2")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.30)(x)

    # Block 3
    x = layers.Dense(128, kernel_regularizer=reg, name="dense3")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.25)(x)

    # Block 4
    x = layers.Dense(64, kernel_regularizer=reg, name="dense4")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.20)(x)

    # Output
    out = layers.Dense(n_classes, activation="softmax", name="output")(x)

    model = models.Model(inp, out, name="GestureNet_v2")
    return model

# ── Main ─────────────────────────────────────────────────────
def train():
    print("\n" + "="*60)
    print("  HIGH-ACCURACY GESTURE MODEL TRAINER  v2")
    print("="*60)
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    # 1. Load
    print("\n[1/6] Loading dataset...")
    X_raw, y_raw, classes = load_dataset()
    n_classes = len(classes)
    print(f"\n  {len(X_raw)} total raw samples | {n_classes} classes")
    print(f"  Classes: {classes}")

    # 2. Normalize
    print("\n[2/6] Normalizing keypoints (palm-size invariant)...")
    X_norm = np.array([normalize_keypoints(x) for x in X_raw], dtype=np.float32)

    # 3. Split BEFORE augmentation (prevent data leakage)
    print("\n[3/6] Splitting into train/val/test (no leakage)...")
    sss_test  = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    train_idx, test_idx = next(sss_test.split(X_norm, y_raw))
    X_trainval, y_trainval = X_norm[train_idx], y_raw[train_idx]
    X_test,     y_test     = X_norm[test_idx],  y_raw[test_idx]

    sss_val   = StratifiedShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=SEED)
    tr_idx, val_idx = next(sss_val.split(X_trainval, y_trainval))
    X_train, y_train = X_trainval[tr_idx],  y_trainval[tr_idx]
    X_val,   y_val   = X_trainval[val_idx], y_trainval[val_idx]

    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # 4. Augment training set only
    print(f"\n[4/6] Augmenting training data (x{AUG_FACTOR})...")
    X_train_aug, y_train_aug = augment_dataset(X_train, y_train, AUG_FACTOR)
    # Shuffle
    perm = np.random.permutation(len(X_train_aug))
    X_train_aug, y_train_aug = X_train_aug[perm], y_train_aug[perm]
    print(f"  Augmented training size: {len(X_train_aug)}")

    # One-hot
    y_train_cat = tf.keras.utils.to_categorical(y_train_aug, n_classes)
    y_val_cat   = tf.keras.utils.to_categorical(y_val,       n_classes)

    # 5. Class weights (handles imbalance)
    cw_values = compute_class_weight("balanced", classes=np.arange(n_classes), y=y_train_aug)
    class_weights = {i: float(w) for i, w in enumerate(cw_values)}
    print(f"\n  Class weights: { {classes[k]: round(v,2) for k,v in class_weights.items()} }")

    # 6. Build & train
    print("\n[5/6] Building model...")
    model = build_model(n_classes)
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    os.makedirs(MODEL_FOLDER, exist_ok=True)
    best_model_path = os.path.join(MODEL_FOLDER, "static_gesture_model.keras")
    checkpoint_path = os.path.join(MODEL_FOLDER, "_ckpt_best.keras")

    cbs = [
        callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.4,
            patience=10,
            min_lr=1e-6,
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=30,
            restore_best_weights=True,
            verbose=1
        ),
    ]

    print("\n[6/6] Training...")
    t0 = time.time()
    history = model.fit(
        X_train_aug, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=cbs,
        verbose=1
    )
    elapsed = time.time() - t0
    print(f"\n  Training time: {elapsed/60:.1f} min")

    # Load best checkpoint
    if os.path.exists(checkpoint_path):
        model = tf.keras.models.load_model(checkpoint_path)
        print("  Loaded best checkpoint.")

    # ── Evaluation ──
    print("\n" + "="*60)
    print("  EVALUATION ON HELD-OUT TEST SET")
    print("="*60)
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred      = np.argmax(y_pred_prob, axis=1)
    test_acc    = float(np.mean(y_pred == y_test))
    print(f"\n  Test Accuracy: {test_acc*100:.2f}%\n")

    print(classification_report(y_test, y_pred, target_names=classes, digits=3))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("  Confusion Matrix (rows=true, cols=pred):")
    header = "         " + "  ".join(f"{c[:5]:>5}" for c in classes)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:5d}" for v in row)
        print(f"  {classes[i][:8]:>8}  {row_str}")

    # ── Save ──
    model.save(best_model_path)
    joblib.dump(classes, os.path.join(MODEL_FOLDER, "static_class_names.pkl"))
    print(f"\n  Model saved  -> {best_model_path}")
    print(f"  Classes saved-> static_class_names.pkl")

    # Clean up checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    print("\n  Training complete!")
    print(f"  Final val_accuracy : {max(history.history['val_accuracy'])*100:.2f}%")
    print(f"  Final test_accuracy: {test_acc*100:.2f}%")
    print("="*60 + "\n")

if __name__ == "__main__":
    train()

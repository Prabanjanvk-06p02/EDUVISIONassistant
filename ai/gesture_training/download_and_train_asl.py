"""
=============================================================
  DOWNLOAD + PROCESS + TRAIN — ASL A-Z DATASET PIPELINE
=============================================================
  This script downloads the best publicly available
  MediaPipe-processed ASL alphabet dataset from Kaggle,
  converts it into our keypoint format, and trains
  a high-accuracy model that can recognize all 26 letters
  (A-Z) — enough to fingerspell ANY English word.

  WHAT IT DOES:
    1. Downloads "grassknoted/asl-alphabet" from Kaggle
       (87,000 images, 200x200px, 29 classes)
    2. Runs MediaPipe on every image to extract
       21-landmark hand keypoints (126 features)
    3. Saves keypoints to datasets/Custom_Data/<letter>/
    4. Trains a high-accuracy DNN (v2 pipeline)

  PREREQUISITES:
    Step 1: pip install kaggle
    Step 2: Go to https://www.kaggle.com/settings/account
            -> Create New Token -> downloads kaggle.json
    Step 3: Place kaggle.json at:
            C:\\Users\\<YourName>\\.kaggle\\kaggle.json

  USAGE:
    python download_and_train_asl.py

  OUTPUT:
    models/static_gesture_model.keras  — 26-class A-Z model
    models/static_class_names.pkl      — ['A','B','C',...'Z']
=============================================================
"""

import os
import sys
import cv2
import mediapipe as mp
import numpy as np
import json
import zipfile
import shutil
import time
import joblib
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models, regularizers, callbacks

# ── Paths ────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATASET_FOLDER  = os.path.normpath(os.path.join(BASE_DIR, "../../ai/datasets/Custom_Data"))
MODEL_FOLDER    = os.path.normpath(os.path.join(BASE_DIR, "../../models"))
TEMP_FOLDER     = os.path.normpath(os.path.join(BASE_DIR, "../../temp_asl_download"))

# ── Config ───────────────────────────────────────────────────
KAGGLE_DATASET    = "grassknoted/asl-alphabet"   # 87k images, 200x200, 29 classes
MAX_IMAGES_PER_CLASS = 800     # use up to 800 images per letter for speed
                                # (all 3000 takes ~30 min to process)
NUM_WORKERS       = 4           # parallel image processing threads
MIN_SAMPLES       = 100         # skip classes with fewer valid keypoints
SEED              = 42

# Training config (same as v2)
EPOCHS      = 200
BATCH_SIZE  = 64
AUG_FACTOR  = 6
TEST_SIZE   = 0.15
VAL_SIZE    = 0.15

# ── MediaPipe (shared; NOT thread-safe — one per worker) ─────
def _make_hands():
    return mp.solutions.hands.Hands(
        static_image_mode=True,   # process individual images
        max_num_hands=1,          # ASL letters use 1 hand
        min_detection_confidence=0.50,
        min_tracking_confidence=0.50
    )

mp_draw = mp.solutions.drawing_utils

# ── Keypoint helpers ─────────────────────────────────────────
def normalize_keypoints(kp: np.ndarray) -> np.ndarray:
    out = np.zeros(126, dtype=np.float32)
    for base in (0, 63):
        hand = kp[base: base + 63]
        if not np.any(hand):
            continue
        wrist   = hand[0:3]
        mid_mcp = hand[27:30]
        scale   = float(np.linalg.norm(mid_mcp - wrist)) + 1e-6
        reshaped = hand.reshape(21, 3)
        out[base: base + 63] = ((reshaped - wrist) / scale).ravel()
    return out

def extract_keypoints_from_image(img_path: str, hands_det) -> np.ndarray | None:
    """Extract 126-feature keypoint vector from an image file."""
    img = cv2.imread(img_path)
    if img is None:
        return None

    # Resize for faster MediaPipe processing
    img = cv2.resize(img, (224, 224))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = hands_det.process(rgb)

    if not results.multi_hand_landmarks:
        return None

    kp = np.zeros(126, dtype=np.float32)
    for hand_idx, hand_lm in enumerate(results.multi_hand_landmarks):
        if hand_idx >= 2:
            break
        offset = hand_idx * 63
        for i, lm in enumerate(hand_lm.landmark):
            kp[offset + i*3]     = lm.x
            kp[offset + i*3 + 1] = lm.y
            kp[offset + i*3 + 2] = lm.z

    return kp

# ── STEP 1: Download ─────────────────────────────────────────
def download_dataset():
    print("\n" + "="*60)
    print("  STEP 1: Downloading ASL Alphabet dataset from Kaggle")
    print("="*60)

    try:
        import kaggle
        kaggle.api.authenticate()
        print(f"  [OK] Kaggle authenticated")
    except Exception as e:
        print(f"\n  [ERROR] Kaggle authentication failed: {e}")
        print("""
  To fix this:
  1. Go to: https://www.kaggle.com/settings
  2. Scroll to 'API' section -> Click 'Create New Token'
  3. This downloads 'kaggle.json'
  4. Move it to: C:\\Users\\<YourName>\\.kaggle\\kaggle.json
  5. Run this script again.
        """)
        sys.exit(1)

    os.makedirs(TEMP_FOLDER, exist_ok=True)

    print(f"  Downloading '{KAGGLE_DATASET}'...")
    print(f"  (This is ~1.2 GB — may take a few minutes)")

    try:
        kaggle.api.dataset_download_files(
            KAGGLE_DATASET,
            path=TEMP_FOLDER,
            unzip=True,
            quiet=False
        )
        print(f"  [OK] Download complete -> {TEMP_FOLDER}")
    except Exception as e:
        print(f"  [ERROR] Download failed: {e}")
        sys.exit(1)

    # Find the asl_alphabet_train folder
    train_dir = None
    for root, dirs, files in os.walk(TEMP_FOLDER):
        if "asl_alphabet_train" in dirs:
            train_dir = os.path.join(root, "asl_alphabet_train")
            break
        for d in dirs:
            sub = os.path.join(root, d)
            if os.path.isdir(sub):
                sub_dirs = os.listdir(sub)
                if len(sub_dirs) > 20:   # likely the class folders
                    train_dir = sub
                    break

    if train_dir is None:
        # Try finding directly
        for item in os.listdir(TEMP_FOLDER):
            full = os.path.join(TEMP_FOLDER, item)
            if os.path.isdir(full) and len(os.listdir(full)) > 20:
                train_dir = full
                break

    if train_dir is None:
        print(f"  [ERROR] Could not find training directory in {TEMP_FOLDER}")
        print(f"  Contents: {os.listdir(TEMP_FOLDER)}")
        sys.exit(1)

    print(f"  Training images found at: {train_dir}")
    return train_dir

# ── STEP 2: Extract keypoints from images ────────────────────
def process_class(class_name: str, img_paths: list[str]) -> tuple[str, int, int]:
    """Process all images for one class. Returns (class_name, saved, skipped)."""
    out_dir = os.path.join(DATASET_FOLDER, class_name)
    os.makedirs(out_dir, exist_ok=True)

    # Get existing frame count
    existing = len([f for f in os.listdir(out_dir) if f.endswith(".npy")])
    frame_id = existing

    # Create a per-thread MediaPipe instance
    hands_det = _make_hands()

    saved  = 0
    failed = 0

    for img_path in img_paths:
        kp = extract_keypoints_from_image(img_path, hands_det)
        if kp is not None:
            normalized = normalize_keypoints(kp)
            save_path  = os.path.join(out_dir, f"{frame_id}.npy")
            np.save(save_path, normalized)   # save normalized directly
            frame_id += 1
            saved    += 1
        else:
            failed += 1

    hands_det.close()
    return class_name, saved, failed

def extract_keypoints(train_dir: str):
    print("\n" + "="*60)
    print("  STEP 2: Extracting MediaPipe keypoints from images")
    print("="*60)

    os.makedirs(DATASET_FOLDER, exist_ok=True)

    # Get class folders
    class_dirs = sorted([
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    ])

    # Filter: only A-Z (skip 'del', 'nothing', 'space' etc.)
    letter_classes = [c for c in class_dirs if len(c) == 1 and c.isalpha()]
    print(f"  Found {len(letter_classes)} letter classes: {letter_classes}")

    total_saved  = 0
    total_failed = 0
    t0 = time.time()

    for idx, cls in enumerate(letter_classes):
        cls_dir   = os.path.join(train_dir, cls)
        all_imgs  = [
            os.path.join(cls_dir, f)
            for f in os.listdir(cls_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        # Shuffle and limit
        rng = np.random.default_rng(SEED + idx)
        rng.shuffle(all_imgs)
        imgs_to_use = all_imgs[:MAX_IMAGES_PER_CLASS]

        elapsed = time.time() - t0
        print(f"  [{idx+1:2d}/{len(letter_classes)}] {cls:2s} | "
              f"{len(imgs_to_use)} images | "
              f"elapsed {elapsed:.0f}s", end=" ... ", flush=True)

        _, saved, failed = process_class(cls.upper(), imgs_to_use)
        total_saved  += saved
        total_failed += failed
        pct = saved / max(1, saved + failed) * 100
        print(f"saved={saved} ({pct:.0f}%), skipped={failed}")

    elapsed = time.time() - t0
    print(f"\n  [OK] Keypoint extraction complete!")
    print(f"  Total saved : {total_saved}")
    print(f"  Total failed: {total_failed} (no hand detected)")
    print(f"  Time taken  : {elapsed/60:.1f} min")
    print(f"  Output dir  : {DATASET_FOLDER}")

# ── STEP 3: Augmentation ─────────────────────────────────────
def augment_sample(kp, rng):
    aug = kp.copy()
    aug += rng.normal(0, 0.010, size=aug.shape)
    theta   = rng.uniform(-0.20, 0.20)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    for base in (0, 63):
        if not np.any(aug[base: base+63]):
            continue
        for i in range(21):
            o      = base + i * 3
            x, y   = aug[o], aug[o+1]
            aug[o]   =  cos_t * x - sin_t * y
            aug[o+1] =  sin_t * x + cos_t * y
    aug *= rng.uniform(0.85, 1.15)
    return aug.astype(np.float32)

# ── STEP 4: Train ─────────────────────────────────────────────
def build_model(n_classes: int) -> tf.keras.Model:
    reg = regularizers.l2(1e-4)
    inp = layers.Input(shape=(126,))
    x   = layers.BatchNormalization()(inp)
    x   = layers.Dense(512, kernel_regularizer=reg)(x)
    x   = layers.BatchNormalization()(x); x = layers.Activation("relu")(x); x = layers.Dropout(0.35)(x)
    x   = layers.Dense(256, kernel_regularizer=reg)(x)
    x   = layers.BatchNormalization()(x); x = layers.Activation("relu")(x); x = layers.Dropout(0.30)(x)
    x   = layers.Dense(128, kernel_regularizer=reg)(x)
    x   = layers.BatchNormalization()(x); x = layers.Activation("relu")(x); x = layers.Dropout(0.25)(x)
    x   = layers.Dense(64,  kernel_regularizer=reg)(x)
    x   = layers.BatchNormalization()(x); x = layers.Activation("relu")(x); x = layers.Dropout(0.20)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return models.Model(inp, out, name="ASL_GestureNet_AZ")

def train_model():
    print("\n" + "="*60)
    print("  STEP 3: Training high-accuracy A-Z gesture model")
    print("="*60)

    if not os.path.exists(DATASET_FOLDER):
        print(f"  [ERROR] Dataset folder not found: {DATASET_FOLDER}")
        return

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
            print(f"  [SKIP] '{cls}' — {len(files)} samples (min={MIN_SAMPLES})")
            continue
        for f in files:
            try:
                kp = np.load(os.path.join(cls_dir, f)).astype(np.float32)
                X_raw.append(kp)
                y_raw.append(label)
            except Exception:
                pass
        print(f"  [OK]   '{cls}' — {len(files)} samples (label {label})")
        valid_classes.append(cls)
        label += 1

    if len(valid_classes) < 2:
        print("  [ERROR] Not enough classes. Run Step 2 first.")
        return

    X = np.array(X_raw, dtype=np.float32)
    y = np.array(y_raw,  dtype=np.int32)
    n_classes = len(valid_classes)

    print(f"\n  {len(X)} samples | {n_classes} classes: {valid_classes}")

    # Split (before augmentation to avoid leakage)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    tr_idx, te_idx = next(sss.split(X, y))
    X_tv, y_tv = X[tr_idx], y[tr_idx]
    X_te, y_te = X[te_idx], y[te_idx]

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=SEED)
    tr2_idx, va_idx = next(sss2.split(X_tv, y_tv))
    X_tr, y_tr = X_tv[tr2_idx], y_tv[tr2_idx]
    X_va, y_va = X_tv[va_idx],  y_tv[va_idx]

    print(f"  Train: {len(X_tr)} | Val: {len(X_va)} | Test: {len(X_te)}")

    # Augment
    print(f"  Augmenting train set x{AUG_FACTOR}...")
    rng = np.random.default_rng(SEED)
    X_aug_list, y_aug_list = [X_tr], [y_tr]
    for _ in range(AUG_FACTOR):
        X_aug_list.append(np.array([augment_sample(x, rng) for x in X_tr]))
        y_aug_list.append(y_tr)
    X_aug = np.concatenate(X_aug_list); y_aug = np.concatenate(y_aug_list)
    perm  = np.random.permutation(len(X_aug))
    X_aug, y_aug = X_aug[perm], y_aug[perm]
    print(f"  Augmented size: {len(X_aug)}")

    y_aug_cat = tf.keras.utils.to_categorical(y_aug, n_classes)
    y_va_cat  = tf.keras.utils.to_categorical(y_va,  n_classes)

    # Class weights
    cw_vals = compute_class_weight("balanced", classes=np.arange(n_classes), y=y_aug)
    cw = {i: float(w) for i, w in enumerate(cw_vals)}

    # Build & compile
    model = build_model(n_classes)
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    os.makedirs(MODEL_FOLDER, exist_ok=True)
    ckpt_path = os.path.join(MODEL_FOLDER, "_ckpt_az.keras")

    cbs = [
        callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy",
                                  save_best_only=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.4,
                                    patience=12, min_lr=1e-6, verbose=1),
        callbacks.EarlyStopping(monitor="val_accuracy", patience=35,
                                restore_best_weights=True, verbose=1),
    ]

    print("\n  Training...")
    t0 = time.time()
    history = model.fit(
        X_aug, y_aug_cat,
        validation_data=(X_va, y_va_cat),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        class_weight=cw, callbacks=cbs, verbose=1
    )
    elapsed = time.time() - t0
    print(f"\n  Training time: {elapsed/60:.1f} min")

    if os.path.exists(ckpt_path):
        model = tf.keras.models.load_model(ckpt_path)

    # Evaluate
    y_pred  = np.argmax(model.predict(X_te, verbose=0), axis=1)
    test_acc = float(np.mean(y_pred == y_te))
    print(f"\n  Test Accuracy: {test_acc*100:.2f}%")

    from sklearn.metrics import classification_report
    print(classification_report(y_te, y_pred, target_names=valid_classes, digits=3))

    # Save
    best_path = os.path.join(MODEL_FOLDER, "static_gesture_model.keras")
    model.save(best_path)
    joblib.dump(valid_classes, os.path.join(MODEL_FOLDER, "static_class_names.pkl"))
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    print(f"\n  Model saved  -> {best_path}")
    print(f"  Classes      -> {valid_classes}")
    print(f"  Val accuracy -> {max(history.history['val_accuracy'])*100:.2f}%")
    print(f"  Test accuracy-> {test_acc*100:.2f}%")
    print("="*60 + "\n")

# ── Main ─────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  ASL A-Z DATASET DOWNLOAD + TRAIN PIPELINE")
    print("="*60)
    print(f"  Target: 26-class model (A-Z fingerspelling)")
    print(f"  Any English word can be spelled with A-Z!")
    print(f"  Max images per class: {MAX_IMAGES_PER_CLASS}")
    print()

    # Check what we already have
    existing_classes = []
    if os.path.exists(DATASET_FOLDER):
        existing_classes = [
            d for d in os.listdir(DATASET_FOLDER)
            if os.path.isdir(os.path.join(DATASET_FOLDER, d))
        ]

    already_has_az = len([c for c in existing_classes if len(c)==1 and c.isalpha()]) >= 20

    if already_has_az:
        print(f"  Found existing A-Z dataset ({len(existing_classes)} classes).")
        choice = input("  Re-download and re-process? (y/n, default=n): ").strip().lower()
        if choice != 'y':
            print("  Skipping download — using existing dataset.")
            train_model()
            return

    print("  Mode:")
    print("    1. Download from Kaggle (BEST quality, needs Kaggle account)")
    print("    2. Train on existing data only")
    mode = input("  Choose (1/2): ").strip()

    if mode == "1":
        train_dir = download_dataset()
        extract_keypoints(train_dir)
        print("\n  Cleaning up temporary download files...")
        try:
            shutil.rmtree(TEMP_FOLDER)
            print("  [OK] Temp files removed.")
        except Exception:
            pass

    train_model()

if __name__ == "__main__":
    main()

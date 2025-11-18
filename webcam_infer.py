#!/usr/bin/env python3
"""
webcam_infer.py (MediaPipe hand-crop edition)
- Optionally uses MediaPipe Hands to detect hands and crop to the hand-region (with padding).
- Temporal smoothing for both softmax predictions and bounding-box to reduce jitter.
- Auto-detects model input size and backbone preprocessing (EffNet/MobileNet).
- Shows top-3 predictions on screen and FPS.
"""

import os
import sys
import time
import json
import numpy as np
import cv2

from collections import deque

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mn_preprocess

# ===== USER CONFIG =====
MODEL_PATH = "asl_model.h5"
CLASS_JSON = "class_indices.json"
META_JSON = "model_meta.json"

USE_HAND_CROP = True          # set False to disable MediaPipe cropping
HAND_PAD = 1.25              # >1.0 expands bbox (1.25 = 25% padding)
HAND_SMOOTH_N = 6            # temporal smoothing length for bbox
SMOOTH_N = 8                 # temporal smoothing for predictions
ARCH_HINT = "efficientnetb0"  # fallback arch hint
# =======================

# Try to import MediaPipe if desired
mp_hands = None
if USE_HAND_CROP:
    try:
        import mediapipe as mp
        mp_hands = mp.solutions.hands
        print("[INFO] mediapipe imported. Hand-crop enabled.")
    except Exception as e:
        print("[WARN] mediapipe not available; falling back to full-frame inference. Install with `pip install mediapipe`.")
        mp_hands = None
        USE_HAND_CROP = False

# --- Console single-key helper (simple) ---
class ConsoleKeyReader:
    def __init__(self):
        self.is_windows = (os.name == 'nt')
        self.fd = None
        self.old_settings = None
        if not self.is_windows:
            try:
                import termios, tty, sys, select
                self.termios = termios
                self.tty = tty
                self.fd = sys.stdin.fileno()
                self.old_settings = termios.tcgetattr(self.fd)
                tty.setcbreak(self.fd)
            except Exception:
                self.fd = None
                self.old_settings = None

    def get_key(self):
        if self.is_windows:
            try:
                import msvcrt
                if msvcrt.kbhit():
                    return msvcrt.getwch()
            except Exception:
                return None
            return None
        else:
            if self.fd is None:
                return None
            import select
            dr, _, _ = select.select([sys.stdin], [], [], 0)
            if dr:
                return sys.stdin.read(1)
            return None

    def restore(self):
        if not self.is_windows and self.fd is not None and self.old_settings is not None:
            try:
                self.termios.tcsetattr(self.fd, self.termios.TCSADRAIN, self.old_settings)
            except Exception:
                pass

# --- Load model ---
print("[INFO] Loading model:", MODEL_PATH)
model = load_model(MODEL_PATH, compile=False)
print("[INFO] Model loaded.")

# ---------- Use metadata if present ----------
meta = None
if os.path.exists(META_JSON):
    try:
        with open(META_JSON, "r") as f:
            meta = json.load(f)
            print("[INFO] Loaded model_meta.json:", meta)
    except Exception as e:
        print("[WARN] Could not read model_meta.json:", e)

# --- Determine input size expected by the model ---
input_shape = model.input_shape  # often (None, H, W, C)
print("[INFO] model.input_shape:", input_shape)
if meta and "target_size" in meta:
    H, W = int(meta["target_size"][0]), int(meta["target_size"][1])
    print(f"[INFO] Using target size from model_meta.json: H={H}, W={W}")
else:
    if isinstance(input_shape, tuple) and len(input_shape) == 4:
        _, H, W, C = input_shape
        H = int(H) if H is not None else 224
        W = int(W) if W is not None else 224
    else:
        H = W = 224
    print(f"[INFO] Derived target size from model.input_shape: H={H}, W={W}")

IMG_SIZE = (int(W), int(H))  # cv2.resize expects (width, height)
EXPECTED_CHANNELS = model.input_shape[-1] if model.input_shape[-1] is not None else 3
print(f"[INFO] Using resize -> {IMG_SIZE}, expected channels = {EXPECTED_CHANNELS}")

# --- Auto-detect preprocess fn ---
layer_names = " ".join([l.name.lower() for l in model.layers[:200]])
if meta and "model_arch" in meta:
    arch_hint = str(meta["model_arch"]).lower()
    if arch_hint.startswith("mobilenet"):
        preproc = mn_preprocess
    else:
        preproc = eff_preprocess
    print("[INFO] Using model_arch hint ->", arch_hint)
else:
    if "efficientnet" in layer_names:
        preproc = eff_preprocess
        print("[INFO] Detected EfficientNet backbone; using EfficientNet preprocess.")
    elif "mobilenet" in layer_names or "mobilenetv2" in layer_names:
        preproc = mn_preprocess
        print("[INFO] Detected MobileNet backbone; using MobileNetV2 preprocess.")
    else:
        if ARCH_HINT.lower().startswith("mobilenet"):
            preproc = mn_preprocess
        else:
            preproc = eff_preprocess
        print("[WARN] Could not auto-detect backbone; using ARCH_HINT:", ARCH_HINT)

# --- Load class mapping robustly ---
class_labels = {}
if os.path.exists(CLASS_JSON):
    try:
        with open(CLASS_JSON, "r") as f:
            class_indices = json.load(f)
            cleaned = {}
            for k, v in class_indices.items():
                try:
                    idx = int(v)
                except Exception:
                    try:
                        idx = int(float(v))
                    except Exception:
                        idx = v
                cleaned[str(k)] = int(idx)
            class_labels = {v: k for k, v in cleaned.items()}
            print("[INFO] Loaded class indices:", cleaned)
    except Exception as e:
        print("[WARN] Failed to load class_indices.json:", e)
else:
    print("[WARN] class_indices.json not found; labels will be numeric indices.")

# --- Setup MediaPipe Hands (if enabled) ---
mp_hands_instance = None
if USE_HAND_CROP and mp_hands is not None:
    # use default detection/confidence values; change min_detection_confidence if needed
    mp_hands_instance = mp_hands.Hands(static_image_mode=False,
                                        max_num_hands=2,
                                        min_detection_confidence=0.5,
                                        min_tracking_confidence=0.5)
    print("[INFO] MediaPipe Hands initialized.")

# temporal smoothing buffers
pred_buffer = deque(maxlen=SMOOTH_N)
bbox_buffer = deque(maxlen=HAND_SMOOTH_N)  # stores normalized bbox (x_min,y_min,x_max,y_max)

# --- Open webcam ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Could not open webcam. Exiting.")
    sys.exit(1)
print("[INFO] Webcam opened.")
key_reader = ConsoleKeyReader()

def expand_and_clip_bbox(xmin, ymin, xmax, ymax, pad, img_w, img_h):
    """Expand normalized bbox by pad factor around center, clip to image bounds, return integer pixel coords."""
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    w = (xmax - xmin) * pad
    h = (ymax - ymin) * pad
    # make it square by taking max
    s = max(w, h)
    xmin2 = cx - s/2.0
    ymin2 = cy - s/2.0
    xmax2 = cx + s/2.0
    ymax2 = cy + s/2.0
    # clip
    xmin2 = max(0.0, xmin2); ymin2 = max(0.0, ymin2)
    xmax2 = min(1.0, xmax2); ymax2 = min(1.0, ymax2)
    # convert to pixel coords
    x1 = int(round(xmin2 * img_w)); y1 = int(round(ymin2 * img_h))
    x2 = int(round(xmax2 * img_w)); y2 = int(round(ymax2 * img_h))
    # ensure non-zero
    if x2 <= x1: x2 = min(img_w, x1 + 1)
    if y2 <= y1: y2 = min(img_h, y1 + 1)
    return x1, y1, x2, y2

def get_smooth_bbox_from_buffer(buffer, img_w, img_h):
    """Average normalized boxes in buffer and return pixel bbox."""
    if not buffer:
        return None
    arr = np.stack(list(buffer), axis=0)  # (N,4)
    mean = np.mean(arr, axis=0)
    return expand_and_clip_bbox(mean[0], mean[1], mean[2], mean[3], 1.0, img_w, img_h)

try:
    print("[INFO] Starting webcam loop. Press 'q' in image window or console to quit.")
    last_time = time.time()
    frame_count = 0
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to read frame. Exiting loop.")
            break
        frame_count += 1
        img_h, img_w = frame.shape[:2]

        # MediaPipe expects RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- Hand detection & crop (if enabled) ---
        crop_img = None
        used_bbox_for_draw = None  # for visualization
        if USE_HAND_CROP and mp_hands_instance is not None:
            # run hands detection on a resized copy for speed (MediaPipe handles scaling internally),
            # but we pass full RGB frame anyway (MP works fine).
            results = mp_hands_instance.process(img_rgb)
            if results and results.multi_hand_landmarks:
                # compute bounding box that contains all hands
                xmins, ymins, xmaxs, ymaxs = [], [], [], []
                for hand_landmarks in results.multi_hand_landmarks:
                    xs = [lm.x for lm in hand_landmarks.landmark]
                    ys = [lm.y for lm in hand_landmarks.landmark]
                    xmins.append(min(xs)); ymins.append(min(ys))
                    xmaxs.append(max(xs)); ymaxs.append(max(ys))
                xmin = max(0.0, min(xmins)); ymin = max(0.0, min(ymins))
                xmax = min(1.0, max(xmaxs)); ymax = min(1.0, max(ymaxs))
                # add to temporal buffer (normalized coords)
                bbox_buffer.append((xmin, ymin, xmax, ymax))
                # get smoothed bbox
                smoothed_pixel_bbox = get_smooth_bbox_from_buffer(bbox_buffer, img_w, img_h)
                if smoothed_pixel_bbox is not None:
                    x1, y1, x2, y2 = smoothed_pixel_bbox
                    # expand by HAND_PAD around center (we already kept square), re-clip
                    # compute normalized mean bbox to re-expand
                    nx1 = x1 / img_w; ny1 = y1 / img_h; nx2 = x2 / img_w; ny2 = y2 / img_h
                    x1, y1, x2, y2 = expand_and_clip_bbox(nx1, ny1, nx2, ny2, HAND_PAD, img_w, img_h)
                    used_bbox_for_draw = (x1, y1, x2, y2)
                    # crop and use that for inference
                    crop_img = frame[y1:y2, x1:x2]
            else:
                # If no hands detected, optionally clear or keep recent bbox; here we'll just not crop.
                # Do not clear buffer to allow brief occlusions to still use previous bbox.
                crop_img = None

        # If we have a cropped hand image, use it; else use full frame
        if crop_img is not None and crop_img.size > 0:
            src = crop_img
        else:
            src = frame

        # Prepare src for model: convert BGR->RGB (if src is BGR), but our preprocess expects RGB already.
        if src.shape[2] == 3:
            img_proc = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        else:
            img_proc = src

        # Resize to model input (cv2.resize expects (width,height))
        img_resized = cv2.resize(img_proc, IMG_SIZE, interpolation=cv2.INTER_AREA)
        img_resized = img_resized.astype(np.float32)
        img_pre = preproc(np.expand_dims(img_resized, axis=0))

        # Predict
        try:
            preds = model.predict(img_pre, verbose=0)[0]
            if np.any(np.isnan(preds)):
                raise ValueError("Model returned NaN predictions")
        except Exception as ex:
            print("[ERROR] Prediction failed:", ex)
            cv2.imshow("ASL Recognition (error)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            k = key_reader.get_key()
            if k is not None and (str(k).lower() == 'q'):
                break
            continue

        # smoothing over predictions
        pred_buffer.append(preds)
        avg_preds = np.mean(np.stack(pred_buffer, axis=0), axis=0)

        top3_idx = avg_preds.argsort()[-3:][::-1]
        top3 = [(int(i), float(avg_preds[i])) for i in top3_idx]

        main_label = class_labels.get(top3[0][0], str(top3[0][0]))
        main_conf = top3[0][1]
        label_text = f"{main_label} ({main_conf:.2f})"
        cv2.putText(frame, label_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3, cv2.LINE_AA)

        # draw next top-2
        for i, (cid, conf) in enumerate(top3[1:], start=1):
            txt = f"{class_labels.get(cid, cid)} {conf:.2f}"
            cv2.putText(frame, txt, (30, 50 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2, cv2.LINE_AA)

        # draw hand bbox if we used it (for debugging / UX)
        if used_bbox_for_draw is not None:
            x1, y1, x2, y2 = used_bbox_for_draw
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
            cv2.putText(frame, "HandCrop", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)

        # estimate FPS every 15 frames
        if frame_count % 15 == 0:
            now = time.time()
            fps = 15.0 / (now - last_time) if (now - last_time) > 0 else fps
            last_time = now
        cv2.putText(frame, f"FPS:{fps:.1f}", (30, frame.shape[0]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        cv2.imshow("ASL Recognition", frame)

        # GUI key
        k = cv2.waitKey(1)
        if k != -1:
            if (k & 0xFF) == ord('q'):
                print("[INFO] 'q' pressed in image window. Exiting.")
                break
            if k == 27:  # ESC
                print("[INFO] ESC pressed. Exiting.")
                break

        # console key
        ck = key_reader.get_key()
        if ck is not None:
            if str(ck).lower() == 'q':
                print("[INFO] 'q' pressed in console. Exiting.")
                break

        # tiny sleep
        time.sleep(0.005)

finally:
    if mp_hands is not None and mp_hands_instance is not None:
        try:
            mp_hands_instance.close()
        except Exception:
            pass
    cap.release()
    cv2.destroyAllWindows()
    key_reader.restore()
    print("[INFO] Exiting.")

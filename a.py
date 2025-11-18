#!/usr/bin/env python3
"""
a.py - Training script with auto-resume, robust ImageNet weight loading,
proper preprocessing, batch_size and model_arch options, CSV logging, plotting,
and optional fine-tuning (unfreeze last N layers).

Usage examples:
  python a.py --train --dataset_dir ./dataset --model_path asl_model.h5 --img_size 224 --epochs 10 --batch_size 16 --model_arch efficientnetb0
  python a.py --train --dataset_dir ./dataset --model_path asl_model.h5 --img_size 160 --epochs 10 --batch_size 32 --model_arch mobilenetv2
  # train then fine-tune in same run:
  python a.py --train --dataset_dir ./dataset --model_path asl_model.h5 --img_size 224 --epochs 8 --fine_tune_epochs 6 --unfreeze_layers 60 --fine_tune_lr 1e-5
"""
import os
import argparse
import shutil
import traceback
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, Input
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

import pandas as pd
import matplotlib.pyplot as plt

# Optional imports for preprocessors & backbones
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input as eff_preprocess
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mn_preprocess


def build_model(num_classes, img_size=224, arch='efficientnetb0', verbose=True):
    """
    Robust builder:
      - supports 'efficientnetb0' and 'mobilenetv2' backbones
      - tries to load imagenet weights once, on failure removes keras cache and retries
      - falls back to weights=None if still failing
      - returns compiled model and a flag whether imagenet was used
    """
    arch = arch.lower()
    inputs = Input(shape=(img_size, img_size, 3))
    def try_create_base(weights_choice):
        if arch.startswith('efficientnet'):
            return EfficientNetB0(include_top=False, weights=weights_choice, input_tensor=inputs)
        elif arch.startswith('mobilenet'):
            return MobileNetV2(include_top=False, weights=weights_choice, input_tensor=inputs)
        else:
            # default to efficientnetb0
            return EfficientNetB0(include_top=False, weights=weights_choice, input_tensor=inputs)

    imagenet_loaded = False
    try:
        base = try_create_base("imagenet")
        imagenet_loaded = True
        if verbose:
            print("[INFO] Loaded ImageNet weights successfully on first attempt.")
    except Exception as e:
        print("Warning: failed to load imagenet weights on first attempt:", e)
        # try to remove cached keras models and retry
        try:
            user_keras_models = os.path.join(os.path.expanduser("~"), ".keras", "models")
            if os.path.exists(user_keras_models):
                print("Removing cached keras models at:", user_keras_models)
                shutil.rmtree(user_keras_models, ignore_errors=True)
        except Exception as e2:
            print("While cleaning cache encountered:", e2)

        # retry once
        try:
            print("Retrying to load imagenet weights after cache cleanup...")
            base = try_create_base("imagenet")
            imagenet_loaded = True
            if verbose:
                print("[INFO] Loaded ImageNet weights successfully on second attempt.")
        except Exception as e3:
            print("Second attempt to load imagenet weights failed:", e3)
            print("Traceback:")
            traceback.print_exc()
            print("[WARN] Falling back to weights=None (training from scratch).")
            base = try_create_base(None)
            imagenet_loaded = False

    # Freeze base initially for transfer learning
    try:
        base.trainable = False
    except Exception:
        pass

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=optimizers.Adam(1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model, imagenet_loaded


def plot_training(log_file="training_log.csv"):
    if not os.path.exists(log_file):
        print(f"[WARNING] Log file {log_file} not found. Skipping plots.")
        return

    df = pd.read_csv(log_file)
    # CSVLogger typically writes epoch, accuracy, loss, val_accuracy, val_loss etc.
    if 'epoch' not in df.columns:
        # the CSVLogger may have no epoch column; create one
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'epoch'}, inplace=True)

    print("[INFO] Training log loaded. Showing head:")
    print(df.head())

    # Plot accuracy
    plt.figure(figsize=(8, 5))
    if 'accuracy' in df.columns and 'val_accuracy' in df.columns:
        plt.plot(df['epoch'], df['accuracy'], label='Train Accuracy')
        plt.plot(df['epoch'], df['val_accuracy'], label='Val Accuracy')
        plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Accuracy'); plt.legend(); plt.grid(True)
        plt.savefig('accuracy_plot.png'); plt.show()
    else:
        print("[WARN] accuracy columns not found in CSV log.")

    # Plot loss
    plt.figure(figsize=(8, 5))
    if 'loss' in df.columns and 'val_loss' in df.columns:
        plt.plot(df['epoch'], df['loss'], label='Train Loss')
        plt.plot(df['epoch'], df['val_loss'], label='Val Loss')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss'); plt.legend(); plt.grid(True)
        plt.savefig('loss_plot.png'); plt.show()
    else:
        print("[WARN] loss columns not found in CSV log.")

    print("[INFO] Plots saved (accuracy_plot.png, loss_plot.png)")


def _detect_preproc_from_model(model):
    """
    Inspect top ~100 layer names to detect common backbone keywords and pick the appropriate preprocess_input.
    """
    names = " ".join([l.name.lower() for l in model.layers[:120]])
    if "efficientnet" in names:
        return eff_preprocess, "efficientnetb0"
    if "mobilenet" in names or "mobilenetv2" in names:
        return mn_preprocess, "mobilenetv2"
    # fallback
    return eff_preprocess, None


def main(args):
    checkpoint_path = args.checkpoint_path or "asl_checkpoint.h5"
    csv_log_path = args.csv_log or "training_log.csv"

    # If checkpoint exists, try loading model first so we know expected input size.
    model = None
    imagenet_loaded = False
    initial_epoch = 0
    resumed = False

    if os.path.exists(checkpoint_path):
        print(f"[INFO] Resuming from checkpoint: {checkpoint_path}")
        try:
            model = load_model(checkpoint_path)
            resumed = True
            # try to infer initial epoch from optimizer iterations/steps (best-effort)
            try:
                initial_epoch = int(model.optimizer.iterations.numpy() // 1)
            except Exception:
                initial_epoch = 0
            print(f"[INFO] Loaded checkpoint model. initial_epoch hint = {initial_epoch}")
        except Exception as e:
            print("[WARN] Failed to load checkpoint file, will rebuild: ", e)
            model = None
            resumed = False

    # Decide preprocessing function and target image size for generators:
    if resumed and model is not None:
        # detect expected input size from loaded model
        input_shape = model.input_shape  # often (None, H, W, C)
        print(f"[INFO] Checkpoint model input_shape: {input_shape}")
        if isinstance(input_shape, tuple) and len(input_shape) == 4:
            _, H, W, C = input_shape
            if H is None or W is None:
                print("[WARN] model has dynamic H/W; falling back to args.img_size")
                target_size = (args.img_size, args.img_size)
            else:
                # ImageDataGenerator.flow_from_directory expects (height, width)
                target_size = (int(H), int(W))
        else:
            target_size = (args.img_size, args.img_size)

        # detect appropriate preprocess
        preproc_fn, detected_arch = _detect_preproc_from_model(model)
        if detected_arch:
            print(f"[INFO] Detected backbone '{detected_arch}' from checkpoint; using matching preprocess.")
        else:
            print("[WARN] Could not auto-detect backbone from checkpoint; using EfficientNet preprocess by default.")
            preproc_fn = eff_preprocess

    else:
        # not resuming from checkpoint: use CLI arch and img_size
        target_size = (args.img_size, args.img_size)
        arch = args.model_arch.lower()
        if arch.startswith('efficientnet'):
            preproc_fn = eff_preprocess
        elif arch.startswith('mobilenet'):
            preproc_fn = mn_preprocess
        else:
            preproc_fn = eff_preprocess

    print(f"[INFO] Using target_size={target_size} for ImageDataGenerator.")

    # Create ImageDataGenerator & generators now that we know target_size + preproc
    # conservative augmentations for hand-sign images (avoid horizontal_flip unless you _intend_ mirrored signs)
    datagen = ImageDataGenerator(
        preprocessing_function=preproc_fn,
        validation_split=0.2,
        rotation_range=8,
        width_shift_range=0.08,
        height_shift_range=0.08,
        zoom_range=0.12,
        shear_range=0.03,
        brightness_range=(0.85, 1.15),
        fill_mode='nearest'
    )

    train_gen = datagen.flow_from_directory(
        args.dataset_dir,
        target_size=target_size,
        batch_size=args.batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        color_mode='rgb'
    )
    val_gen = datagen.flow_from_directory(
        args.dataset_dir,
        target_size=target_size,
        batch_size=args.batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        color_mode='rgb'
    )

    # Save class indices mapping and small metadata (write after generator creation)
    class_indices = train_gen.class_indices  # dict: class_name -> index
    with open("class_indices.json", "w") as f:
        json.dump(class_indices, f)
    meta = {
        "target_size": [int(target_size[0]), int(target_size[1])],  # [height, width]
        "model_arch": args.model_arch,
        "class_indices": class_indices,
        "imagenet_loaded": bool(imagenet_loaded)
    }
    with open("model_meta.json", "w") as f:
        json.dump(meta, f)
    print("[INFO] Saved class_indices.json and model_meta.json")

    num_classes = len(train_gen.class_indices)
    print(f"[INFO] Found {train_gen.n} training images, {val_gen.n} validation images, {num_classes} classes.")
    print("[INFO] class indices:", train_gen.class_indices)

    # If resumed model was loaded, check that its input size and output size match:
    if resumed and model is not None:
        # check input spatial size
        model_input_shape = model.input_shape
        if isinstance(model_input_shape, tuple) and len(model_input_shape) == 4:
            _, mH, mW, mC = model_input_shape
            if mH is not None and mW is not None:
                if (mH, mW) != tuple(target_size):
                    print(f"[WARN] Checkpoint expects {(mH, mW)} but generators using {tuple(target_size)}. "
                          f"Generators were set from the checkpoint input (preferred).")
        # check output classes
        try:
            model_output_units = model.output_shape[-1]
            if model_output_units != num_classes:
                print(f"[WARN] Checkpoint model has {model_output_units} output units but dataset has {num_classes} classes.")
                print("[WARN] This mismatch may cause incorrect training or require rebuilding the model.")
        except Exception:
            pass
    else:
        # build model from scratch using num_classes and args.img_size/arch
        print("[INFO] Building model from scratch (or trying to load ImageNet weights)...")
        model, imagenet_loaded = build_model(num_classes, img_size=args.img_size, arch=args.model_arch)
        initial_epoch = 0

    print("[INFO] Model summary (top):")
    model.summary(line_length=120)

    if imagenet_loaded:
        print("[INFO] Using ImageNet pretrained weights for the backbone.")
    else:
        print("[INFO] Backbone ImageNet weights NOT used (or unknown for resumed model).")

    # callbacks
    checkpoint_cb = ModelCheckpoint(checkpoint_path, save_best_only=False, save_weights_only=False, verbose=1)
    csv_logger = CSVLogger(csv_log_path, append=True)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False, verbose=1)

    callbacks_list = [checkpoint_cb, csv_logger, reduce_lr]

    # TRAINING: run initial training only if initial_epoch < args.epochs
    if initial_epoch < args.epochs:
        print(f"[INFO] Starting training from epoch {initial_epoch} to {args.epochs} ...")
        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=args.epochs,
            initial_epoch=initial_epoch,
            callbacks=callbacks_list
        )
    else:
        print("[INFO] initial_epoch >= args.epochs, skipping initial training phase.")

    # ---------- optional fine-tuning ----------
    if args.fine_tune_epochs and args.fine_tune_epochs > 0:
        # Determine start epoch for fine-tune
        ft_start = max(args.epochs, initial_epoch)
        ft_end = ft_start + args.fine_tune_epochs
        print(f"[INFO] Starting fine-tuning from epoch {ft_start} to {ft_end}: unfreeze last {args.unfreeze_layers} layers, lr={args.fine_tune_lr}")

        # set all layers to non-trainable, then unfreeze last N layers
        for layer in model.layers:
            layer.trainable = False

        n = args.unfreeze_layers
        if n < 0:
            n = abs(n)
        n = min(n, len(model.layers))
        if n <= 0:
            print("[WARN] unfreeze_layers <= 0; skipping unfreeze and fine-tune.")
        else:
            print(f"[INFO] Unfreezing last {n} layers (out of {len(model.layers)})")
            for layer in model.layers[-n:]:
                layer.trainable = True

            # recompile with lower LR for fine-tuning
            from tensorflow.keras.optimizers import Adam
            model.compile(optimizer=Adam(args.fine_tune_lr),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

            # Reuse the same callbacks (checkpoint & csv logger) with append=True
            model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=ft_end,
                initial_epoch=ft_start,
                callbacks=callbacks_list + [early_stop]
            )

            print("[INFO] Fine-tuning complete.")

    # final save and plotting
    model.save(args.model_path)
    print(f"[INFO] Final model saved to {args.model_path}")
    print(f"[INFO] Training log saved to {csv_log_path}")

    # plot
    plot_training(csv_log_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", action="store_true")
    p.add_argument("--dataset_dir", type=str, required=True)
    p.add_argument("--model_path", type=str, default="asl_model.h5")
    p.add_argument("--checkpoint_path", type=str, default="asl_checkpoint.h5")
    p.add_argument("--csv_log", type=str, default="training_log.csv")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=10, help="Initial training epochs (head training / baseline).")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--model_arch", type=str, default="efficientnetb0", help="efficientnetb0 (default) or mobilenetv2")
    # Fine-tune options
    p.add_argument("--fine_tune_epochs", type=int, default=0, help="If >0, run fine-tuning after initial training")
    p.add_argument("--unfreeze_layers", type=int, default=50, help="Number of final layers to unfreeze for fine-tuning (use negative to mean 'from index')")
    p.add_argument("--fine_tune_lr", type=float, default=1e-5, help="Learning rate for fine-tuning")
    args = p.parse_args()

    if args.train:
        main(args)

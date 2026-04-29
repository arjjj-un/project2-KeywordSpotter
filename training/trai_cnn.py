#!/usr/bin/env python3
# coding: utf-8

import glob
import os
import pathlib
import random
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.lite.experimental.microfrontend.python.ops import (
    audio_microfrontend_op as frontend_op,
)

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

FSAMP = 16000
WAVE_MS = 1000
WAVE_SAMPS = int(WAVE_MS * FSAMP / 1000)
WINDOW_SIZE_MS = 30
WINDOW_STEP_MS = 20
NUM_FILTERS = 40
I16MIN = -(2**15)
I16MAX = (2**15) - 1

BATCH_SIZE = 32
EPOCHS = 30
AUG_FACTOR = 3
UNKNOWN_PER_FOLDER = 350

SILENCE_STR = "_silence"
UNKNOWN_STR = "_unknown"
KEYWORD_ALEXA = "alexa"
KEYWORD_YES = "yes"
LABEL_LIST = [SILENCE_STR, UNKNOWN_STR, KEYWORD_ALEXA, KEYWORD_YES]
NUM_LABELS = len(LABEL_LIST)

HOME = pathlib.Path.home()
DATA_ROOT = (HOME / "ml-iot" / "hw4" / "data" / "mini_speech_commands_extracted").resolve()
ALEXA_DIR = DATA_ROOT / "alexa"
YES_DIR = DATA_ROOT / "yes"
MINI_SPEECH = DATA_ROOT / "mini_speech_commands"


def load_wav(path: str) -> np.ndarray:
    raw, _ = tf.audio.decode_wav(tf.io.read_file(path), desired_channels=1)
    wav = tf.squeeze(raw, axis=-1).numpy().astype(np.float32)
    if len(wav) >= WAVE_SAMPS:
        return wav[:WAVE_SAMPS]
    return np.pad(wav, (0, WAVE_SAMPS - len(wav)))


def wav_to_spec(wav: np.ndarray) -> np.ndarray:
    wav_i16 = (0.5 * wav * (I16MAX - I16MIN)).astype(np.int16)
    spec = frontend_op.audio_microfrontend(
        tf.constant(wav_i16),
        sample_rate=FSAMP,
        num_channels=NUM_FILTERS,
        window_size=WINDOW_SIZE_MS,
        window_step=WINDOW_STEP_MS,
    )
    return tf.expand_dims(spec, axis=-1).numpy().astype(np.float32)  # (20, 32, 1)


def time_shift(wav: np.ndarray, max_ms: float = 100.0) -> np.ndarray:
    max_s = int(max_ms * FSAMP / 1000)
    shift = np.random.randint(-max_s, max_s + 1)
    return np.roll(wav, shift).astype(np.float32)


def add_noise(wav: np.ndarray, snr_db_range=(10, 30)) -> np.ndarray:
    snr_db = np.random.uniform(*snr_db_range)
    sig_pow = np.mean(wav**2) + 1e-10
    noise_pow = sig_pow / (10 ** (snr_db / 10))
    noise = np.random.randn(len(wav)) * np.sqrt(noise_pow)
    return (wav + noise).astype(np.float32)


def augment(wav: np.ndarray) -> np.ndarray:
    if np.random.rand() < 0.5:
        wav = time_shift(wav)
    if np.random.rand() < 0.5:
        wav = add_noise(wav)
    return wav


def _clean_wav_list(folder: pathlib.Path):
    return [
        f for f in glob.glob(str(folder / "*.wav"))
        if not os.path.basename(f).startswith("._")
    ]


def collect_files():
    alexa = _clean_wav_list(ALEXA_DIR)
    yes = _clean_wav_list(YES_DIR)
    if not alexa:
        raise RuntimeError(f"No alexa wav files under {ALEXA_DIR}")
    if not yes:
        raise RuntimeError(f"No yes wav files under {YES_DIR}")
    random.shuffle(alexa)
    random.shuffle(yes)

    folders = sorted(
        d for d in MINI_SPEECH.iterdir()
        if d.is_dir()
        and d.name.lower() not in (KEYWORD_ALEXA, KEYWORD_YES)
        and not d.name.startswith("_")
    )
    if not folders:
        raise RuntimeError(f"No unknown folders under {MINI_SPEECH}")

    per_folder = UNKNOWN_PER_FOLDER
    unknown = []
    for d in folders:
        wavs = _clean_wav_list(d)
        random.shuffle(wavs)
        unknown.extend(wavs[:min(per_folder, len(wavs))])
    random.shuffle(unknown)
    return alexa, yes, unknown


def make_silence(n: int):
    out = np.zeros((n, WAVE_SAMPS), dtype=np.float32)
    for i in range(n):
        out[i] = (0.01 * np.random.randn(WAVE_SAMPS)).astype(np.float32)
    return out


def build_dataset():
    alexa_files, yes_files, unknown_files = collect_files()
    specs, labels = [], []

    for f in alexa_files:
        wav = load_wav(f)
        specs.append(wav_to_spec(wav))
        labels.append(LABEL_LIST.index(KEYWORD_ALEXA))
        for _ in range(AUG_FACTOR):
            specs.append(wav_to_spec(augment(wav.copy())))
            labels.append(LABEL_LIST.index(KEYWORD_ALEXA))

    for f in yes_files:
        wav = load_wav(f)
        specs.append(wav_to_spec(wav))
        labels.append(LABEL_LIST.index(KEYWORD_YES))
        for _ in range(AUG_FACTOR):
            specs.append(wav_to_spec(augment(wav.copy())))
            labels.append(LABEL_LIST.index(KEYWORD_YES))

    for f in unknown_files:
        specs.append(wav_to_spec(load_wav(f)))
        labels.append(LABEL_LIST.index(UNKNOWN_STR))

    n_silence = max(
        100,
        int(0.2 * ((len(alexa_files) + len(yes_files)) * (1 + AUG_FACTOR) + len(unknown_files))),
    )
    for wav in make_silence(n_silence):
        specs.append(wav_to_spec(wav))
        labels.append(LABEL_LIST.index(SILENCE_STR))

    X = np.stack(specs, axis=0).astype(np.float32)
    y = np.array(labels, dtype=np.int32)

    print("\nDataset total :", len(y), "samples")
    for idx, label in enumerate(LABEL_LIST):
        print(f"  {label:10s}: {int(np.sum(y == idx))}")

    return X, y


def build_model(input_shape):
    inp = layers.Input(shape=input_shape, name="spectrogram")
    x = inp
    for filters in [24, 32, 48]:
        x = layers.DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, (1, 1), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D((2, 2), padding="same")(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(NUM_LABELS)(x)
    m = models.Model(inp, out, name="alexa_yes_dscnn_tflm")
    m.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return m


def representative_dataset(features: np.ndarray):
    n = min(200, len(features))
    for i in range(n):
        yield [features[i : i + 1].astype(np.float32)]


def write_c_header(blob: bytes, path: pathlib.Path, var_name: str = "kws_model"):
    hex_bytes = ", ".join(f"0x{b:02x}" for b in blob)
    text = (
        "#pragma once\n#include <stdint.h>\n\n"
        f"alignas(16) const unsigned char {var_name}[] = {{\n  {hex_bytes}\n}};\n"
        f"const unsigned int {var_name}_len = {len(blob)};\n"
    )
    path.write_text(text, encoding="utf-8")


def save_training_curves(history, out_path: pathlib.Path):
    hist = history.history
    epochs = range(1, len(hist["loss"]) + 1)

    fig, axes = plt.subplots(2, 1, figsize=(9, 8))

    axes[0].plot(epochs, hist["loss"], label="train_loss", color="#1f77b4")
    axes[0].plot(epochs, hist["val_loss"], label="val_loss", color="#ff7f0e", linestyle="--")
    axes[0].set_title("Alexa+Yes KWS Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, hist["accuracy"], label="train_acc", color="#1f77b4")
    axes[1].plot(epochs, hist["val_accuracy"], label="val_acc", color="#ff7f0e", linestyle="--")
    axes[1].set_title("Alexa+Yes KWS Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim([0, 1])
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    print("[INFO] Building dataset...")
    X, y = build_dataset()
    print(f"[INFO] X={X.shape}, y={y.shape}")

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.15, random_state=SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.40, random_state=SEED, stratify=y_tmp
    )

    model = build_model(X_train.shape[1:])
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"[INFO] Test acc={test_acc:.4f}, loss={test_loss:.4f}")

    keras_path = pathlib.Path("kws_alexa_yes_tflm.keras")
    model.save(keras_path, overwrite=True)
    print(f"[OK] Saved {keras_path}")

    curves_path = pathlib.Path("training_curves_alexa_yes_tflm.png")
    save_training_curves(history, curves_path)
    print(f"[OK] Saved {curves_path}")

    repset = X_train[: min(300, len(X_train))].astype(np.float32)
    np.save("repset_features.npy", repset)
    print(f"[OK] Saved repset_features.npy: {repset.shape}")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(repset)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_blob = converter.convert()

    tflite_path = pathlib.Path("kws_alexa_yes_tflm_int8.tflite")
    tflite_path.write_bytes(tflite_blob)
    print(f"[OK] Saved {tflite_path} ({len(tflite_blob)} bytes)")

    header_path = pathlib.Path("kws_alexa_yes_tflm_model_data.h")
    write_c_header(tflite_blob, header_path, "kws_model")
    print(f"[OK] Saved {header_path}")

    print("\nDone. Use kws_alexa_yes_tflm_model_data.h in ESP-IDF app.")


if __name__ == "__main__":
    main()

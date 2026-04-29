import os, glob, random, time, pathlib
from datetime import datetime as dt

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.lite.experimental.microfrontend.python.ops import (
    audio_microfrontend_op as frontend_op,
)

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("[WARN] librosa not found – pitch-shift augmentation disabled.\n"
          "       Install with: pip install librosa")

print("TF version:", tf.__version__)

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

FSAMP          = 16000
WAVE_MS        = 1000
WAVE_SAMPS     = int(WAVE_MS * FSAMP / 1000)   # 16 000
WINDOW_SIZE_MS = 64
WINDOW_STEP_MS = 48
NUM_FILTERS    = 32
I16MIN         = -(2**15)
I16MAX         =  (2**15) - 1
BATCH_SIZE = 32
EPOCHS     = 40
AUG_FACTOR = 4        
UNKNOWN_PER_FOLDER = 250

# ── Labels ────────────────────────────────────────────────────────────────────
SILENCE_STR = "_silence"
UNKNOWN_STR = "_unknown"
KEYWORD_ALEXA = "alexa"
KEYWORD_YES  = "yes"
LABEL_LIST  = [SILENCE_STR, UNKNOWN_STR, KEYWORD_ALEXA, KEYWORD_YES]   # idx: 0, 1, 2, 3
NUM_LABELS  = len(LABEL_LIST)
print("Labels:", LABEL_LIST)

# ── Data directories ──────────────────────────────────────────────────────────
HOME = pathlib.Path.home()
DATA_ROOT = (HOME / "ml-iot" / "hw4" / "data" / "mini_speech_commands_extracted").resolve()
ALEXA_DIR = DATA_ROOT / "alexa"
YES_DIR = DATA_ROOT / "yes"
MINI_SPEECH = DATA_ROOT / "mini_speech_commands"

if not MINI_SPEECH.exists():
    print("mini_speech_commands not found locally – downloading …")
    tf.keras.utils.get_file(
        "mini_speech_commands.zip",
        origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
        extract=True,
        cache_dir=str(HOME / "ml-iot/hw4/data"),
        cache_subdir="mini_speech_commands_extracted",
    )

def _valid_wavs(folder: pathlib.Path) -> list:
    return [
        f for f in glob.glob(str(folder / "*.wav"))
        if not os.path.basename(f).startswith("._")
    ]


def load_alexa_files() -> list:
    files = _valid_wavs(ALEXA_DIR)
    print(f"Alexa raw samples   : {len(files)}")
    print(f"Alexa after aug     : {len(files) * (1 + AUG_FACTOR)}")
    return files

def load_yes_files() -> list:
    files = _valid_wavs(YES_DIR)
    print(f"Yes raw samples     : {len(files)}")
    print(f"Yes after aug       : {len(files) * (1 + AUG_FACTOR)}")
    return files


def load_unknown_files(target_total: int) -> list:
    """
    Pull up to UNKNOWN_PER_FOLDER wavs from each non-keyword folder in
    mini_speech_commands, for balanced unknown coverage.
    """
    folders = sorted([
        d for d in MINI_SPEECH.iterdir()
        if d.is_dir()
        and d.name.lower() not in (KEYWORD_ALEXA, KEYWORD_YES)
        and not d.name.startswith("_")      # exclude _background_noise_ etc.
    ])
    if not folders:
        raise RuntimeError(f"No subfolders found under {MINI_SPEECH}")

    per_folder = UNKNOWN_PER_FOLDER
    selected = []
    for folder in folders:
        wavs = _valid_wavs(folder)
        random.shuffle(wavs)
        selected.extend(wavs[:min(per_folder, len(wavs))])

    random.shuffle(selected)
    selected = selected[:min(len(selected), target_total)]
    print(f"Unknown samples    : {len(selected)}  "
          f"(capped at {per_folder} each from {len(folders)} folders: "
          f"{[d.name for d in folders]})")
    return selected

# ── Augmentation 
def time_shift(wav: np.ndarray, max_ms: float = 100.0) -> np.ndarray:
    max_s = int(max_ms * FSAMP / 1000)
    shift = np.random.randint(-max_s, max_s + 1)
    return np.roll(wav, shift).astype(np.float32)


def add_noise(wav: np.ndarray, snr_db_range=(10, 30)) -> np.ndarray:
    snr_db    = np.random.uniform(*snr_db_range)
    sig_pow   = np.mean(wav ** 2) + 1e-10
    noise_pow = sig_pow / (10 ** (snr_db / 10))
    noise     = np.random.randn(len(wav)) * np.sqrt(noise_pow)
    return (wav + noise).astype(np.float32)


def pitch_shift(wav: np.ndarray, semitone_range=(-2, 2)) -> np.ndarray:
    if not HAS_LIBROSA:
        return wav
    n_steps = np.random.uniform(*semitone_range)
    return librosa.effects.pitch_shift(
        wav.astype(np.float32), sr=FSAMP, n_steps=n_steps
    ).astype(np.float32)


def augment(wav: np.ndarray) -> np.ndarray:
    if np.random.rand() < 0.5:
        wav = time_shift(wav)
    if np.random.rand() < 0.5:
        wav = add_noise(wav)
    if HAS_LIBROSA and np.random.rand() < 0.4:
        wav = pitch_shift(wav)
    return wav

# ── Audio helpers 
def load_wav(path: str) -> np.ndarray:
    """Load WAV → float32 mono, padded/trimmed to exactly WAVE_SAMPS."""
    raw, _ = tf.audio.decode_wav(tf.io.read_file(path), desired_channels=1)
    wav = tf.squeeze(raw, axis=-1).numpy().astype(np.float32)
    if len(wav) >= WAVE_SAMPS:
        return wav[:WAVE_SAMPS]
    return np.pad(wav, (0, WAVE_SAMPS - len(wav)))


def wav_to_spec(wav: np.ndarray) -> np.ndarray:
    """float32 waveform → log-mel spectrogram (T, NUM_FILTERS, 1)."""
    wav_i16 = (0.5 * wav * (I16MAX - I16MIN)).astype(np.int16)
    spec = frontend_op.audio_microfrontend(
        tf.constant(wav_i16),
        sample_rate=FSAMP,
        num_channels=NUM_FILTERS,
        window_size=WINDOW_SIZE_MS,
        window_step=WINDOW_STEP_MS,
    )
    return tf.expand_dims(spec, axis=-1).numpy().astype(np.float32)


def make_silence_waves(n: int) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    rms = rng.uniform(0.005, 0.15, size=n)
    out = np.zeros((n, WAVE_SAMPS), dtype=np.float32)
    for i in range(n):
        out[i] = (rms[i] * rng.standard_normal(WAVE_SAMPS)).astype(np.float32)
    return out

# ── Build dataset ─────────────────────────────────────────────────────────────
def build_dataset(alexa_files: list, yes_files: list, unknown_files: list) -> tuple:

    specs, labels = [], []

    # ── alexa (original + augmented copies) ───────────────────────────────────
    print(f"\nLoading {len(alexa_files)} alexa files …")
    for fpath in alexa_files:
        wav = load_wav(fpath)
        specs.append(wav_to_spec(wav))
        labels.append(LABEL_LIST.index(KEYWORD_ALEXA))
        for _ in range(AUG_FACTOR):
            specs.append(wav_to_spec(augment(wav.copy())))
            labels.append(LABEL_LIST.index(KEYWORD_ALEXA))

    # ── yes (original + augmented copies) ─────────────────────────────────────
    print(f"Loading {len(yes_files)} yes files …")
    for fpath in yes_files:
        wav = load_wav(fpath)
        specs.append(wav_to_spec(wav))
        labels.append(LABEL_LIST.index(KEYWORD_YES))
        for _ in range(AUG_FACTOR):
            specs.append(wav_to_spec(augment(wav.copy())))
            labels.append(LABEL_LIST.index(KEYWORD_YES))

    # ── unknown ───────────────────────────────────────────────────────────────
    print(f"Loading {len(unknown_files)} unknown files …")
    for fpath in unknown_files:
        try:
            wav = load_wav(fpath)
        except Exception:
            continue
        specs.append(wav_to_spec(wav))
        labels.append(LABEL_LIST.index(UNKNOWN_STR))

    # ── silence (synthetic noise) ─────────────────────────────────────────────
    n_alexa_aug = len(alexa_files) * (1 + AUG_FACTOR)
    n_yes_aug = len(yes_files) * (1 + AUG_FACTOR)
    n_silence   = max(int(0.20 * (n_alexa_aug + n_yes_aug + len(unknown_files))), 100)
    print(f"Generating {n_silence} silence clips …")
    for wav in make_silence_waves(n_silence):
        specs.append(wav_to_spec(wav))
        labels.append(LABEL_LIST.index(SILENCE_STR))

    X = np.stack(specs, axis=0)           # (N, T, F, 1)
    y = np.array(labels, dtype=np.int32)

    print(f"\nDataset total : {len(y)} samples")
    print(f"  _silence    : {np.sum(y == 0)}")
    print(f"  _unknown    : {np.sum(y == 1)}")
    print(f"  alexa       : {np.sum(y == 2)}")
    print(f"  yes         : {np.sum(y == 3)}")
    return X, y

alexa_files = load_alexa_files()
yes_files     = load_yes_files()
alexa_post_aug = len(alexa_files) * (1 + AUG_FACTOR)
yes_post_aug  = len(yes_files) * (1 + AUG_FACTOR)
unknown_target = max(alexa_post_aug, yes_post_aug)
unknown_files  = load_unknown_files(target_total=unknown_target)

print(f"\nClass sample counts (raw):")
print(f"  alexa   : {len(alexa_files)}")
print(f"  yes     : {len(yes_files)}")
print(f"  unknown : {len(unknown_files)}")

print("\n─── Building dataset (takes ~1–2 min) ──────────────────────────────────")
t0 = time.time()
X, y = build_dataset(alexa_files, yes_files, unknown_files)
print(f"Dataset built in {time.time() - t0:.1f}s")

# ── Train / val / test split ──────────────────────────────────────────────────
# 85% train  |  9% val  |  6% test  (stratified — each split stays balanced)
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y, test_size=0.15, random_state=SEED, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.40, random_state=SEED, stratify=y_tmp
)
print(f"\nSplit — train:{len(y_train)}  val:{len(y_val)}  test:{len(y_test)}")

INPUT_SHAPE = X_train.shape[1:]   # (T, NUM_FILTERS, 1)
print("Input shape:", INPUT_SHAPE)

# ── tf.data pipelines ─────────────────────────────────────────────────────────
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .shuffle(len(y_train), seed=SEED)
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTOTUNE)
)
val_ds = (
    tf.data.Dataset.from_tensor_slices((X_val, y_val))
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTOTUNE)
)
test_ds = (
    tf.data.Dataset.from_tensor_slices((X_test, y_test))
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

def build_model(input_shape):
    inp = layers.Input(shape=input_shape, name="spectrogram")

    # block 1 — pool freq ÷2
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((1, 2))(x)

    # block 2 — pool freq ÷2
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((1, 2))(x)

    # block 3
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    # block 4
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    # reshape: (B, T, F_reduced, 128) → (B, T, F_reduced × 128)
    t_dim = tf.keras.backend.int_shape(x)[1]
    f_dim = tf.keras.backend.int_shape(x)[2]
    c_dim = tf.keras.backend.int_shape(x)[3]
    x = layers.Reshape((t_dim, f_dim * c_dim), name="cnn_to_gru")(x)

    # BiGRU stack
    x = layers.Bidirectional(
        layers.GRU(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.1),
        name="bigru_1",
    )(x)
    x = layers.Bidirectional(
        layers.GRU(32, return_sequences=False, dropout=0.2),
        name="bigru_2",
    )(x)

    # classification head
    x   = layers.Dense(64, activation="relu")(x)
    x   = layers.Dropout(0.3)(x)
    out = layers.Dense(NUM_LABELS, name="logits")(x)

    m = models.Model(inp, out, name="cnn_gru_kws")
    m.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return m

model = build_model(INPUT_SHAPE)
model.summary()

callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=12, restore_best_weights=True, verbose=1
    ),
]

# ── Train ─────────────────────────────────────────────────────────────────────
print("\n─── Training ───────────────────────────────────────────────────────────")
t0 = time.time()
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)
print(f"Training finished in {time.time() - t0:.1f}s")

# ── Save Keras model ──────────────────────────────────────────────────────────
KERAS_PATH = "kws_alexa_yes.keras"
model.save(KERAS_PATH, overwrite=True)
print(f"Saved Keras model → {KERAS_PATH}")

# ── Training curves  ← saved immediately so they're never lost ───────────────
m = history.history
fig, axes = plt.subplots(2, 1, figsize=(9, 8))

axes[0].semilogy(history.epoch, m["loss"],     label="train", color="#2196F3")
axes[0].semilogy(history.epoch, m["val_loss"], label="val",   color="#FF5722", linestyle="--")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss (log scale)")
axes[0].set_title("Loss - Alexa/Yes KWS (CNN+GRU)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.epoch, m["accuracy"],     label="train", color="#2196F3")
axes[1].plot(history.epoch, m["val_accuracy"], label="val",   color="#FF5722", linestyle="--")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Accuracy - Alexa/Yes KWS (CNN+GRU)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim([0, 1])

plt.tight_layout()
plt.savefig("training_curves_alexa_yes.png", dpi=150)
plt.close()
print("Saved training_curves_alexa_yes.png")

# ── TFLite conversion ─────────────────────────────────────────────────────────
print("\nConverting to TFLite …")
TFLITE_PATH  = "kws_alexa_yes.tflite"
tflite_model = None
tflite_kb    = 0.0
try:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter._experimental_lower_tensor_list_ops = False
    tflite_model = converter.convert()
    with open(TFLITE_PATH, "wb") as f:
        f.write(tflite_model)
    tflite_kb = len(tflite_model) / 1024
    print(f"Saved TFLite model → {TFLITE_PATH}  ({tflite_kb:.1f} KB) ✓")
except Exception as e:
    print(f"[ERROR] TFLite conversion failed: {e}")
    print("  Keras weights are intact – re-run conversion separately.")

# ── Evaluate on held-out test set ─────────────────────────────────────────────
test_loss, test_acc = model.evaluate(test_ds, verbose=2)
print(f"\nTest accuracy : {test_acc:.1%}   Test loss : {test_loss:.4f}")

y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = y_test

# ── Confusion matrix ──────────────────────────────────────────────────────────
conf_mtx = tf.math.confusion_matrix(y_true, y_pred).numpy()
plt.figure(figsize=(6, 5))
sns.heatmap(conf_mtx, xticklabels=LABEL_LIST, yticklabels=LABEL_LIST,
            annot=True, fmt="g", cmap="Blues")
plt.gca().invert_yaxis()
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Alexa/Yes KWS (CNN+GRU)")
plt.tight_layout()
plt.savefig("confusion_matrix_alexa_yes.png", dpi=150)
plt.close()
print("Saved confusion_matrix_alexa_yes.png")

print("\nPer-class metrics:")
for i, lbl in enumerate(LABEL_LIST):
    tp  = conf_mtx[i, i]
    tpr = tp / (np.sum(conf_mtx[i, :]) + 1e-10)
    fpr = (np.sum(conf_mtx[:, i]) - tp) / (
           np.sum(conf_mtx) - np.sum(conf_mtx[i, :]) + 1e-10)
    print(f"  {lbl:12s}  TPR={tpr:.3f}  FPR={fpr:.3f}")


info_path = "kws_alexa_yes_info.txt"
with open(info_path, "w") as f:
    f.write(f"model             = CNN+GRU\n")
    f.write(f"keywords          = [{KEYWORD_ALEXA}, {KEYWORD_YES}]\n")
    f.write(f"label_list        = {LABEL_LIST}\n")
    f.write(f"fsamp             = {FSAMP}\n")
    f.write(f"wave_length_samps = {WAVE_SAMPS}\n")
    f.write(f"window_size_ms    = {WINDOW_SIZE_MS}\n")
    f.write(f"window_step_ms    = {WINDOW_STEP_MS}\n")
    f.write(f"num_filters       = {NUM_FILTERS}\n")
    f.write(f"input_shape       = {INPUT_SHAPE}\n")
    f.write(f"aug_factor        = {AUG_FACTOR}\n")
    f.write(f"alexa_raw         = {len(alexa_files)}\n")
    f.write(f"alexa_post_aug    = {alexa_post_aug}\n")
    f.write(f"yes_raw           = {len(yes_files)}\n")
    f.write(f"yes_post_aug      = {yes_post_aug}\n")
    f.write(f"unknown_total     = {len(unknown_files)}\n")
    f.write(f"epochs_run        = {len(history.epoch)}\n")
    f.write(f"test_accuracy     = {test_acc:.4f}\n")
    f.write(f"test_loss         = {test_loss:.4f}\n")
    f.write(f"keras_model       = {KERAS_PATH}\n")
    f.write(f"tflite_model      = {TFLITE_PATH if tflite_model else 'FAILED'}\n")
    f.write(f"tflite_size_kb    = {tflite_kb:.1f}\n")
    aug_str = "time_shift, additive_noise"
    if HAS_LIBROSA:
        aug_str += ", pitch_shift"
    f.write(f"augmentations     = {aug_str}\n")

print(f"\n{'─'*50}")
print("Done. Artifacts written:")
print(f"  {KERAS_PATH}")
if tflite_model:
    print(f"  {TFLITE_PATH}")
print(f"  training_curves_alexa_yes.png")
print(f"  confusion_matrix_alexa_yes.png")
print(f"  {info_path}")
print(f"{'─'*50}")
"""Emotion recognition from speech using MFCC-style features.

The default command creates a small, deterministic dummy dataset and trains a
NumPy softmax classifier.  TensorFlow CNN/BiLSTM builders are available when
TensorFlow is installed, and real WAV files can be supplied when librosa is
installed.
"""

from __future__ import annotations

import argparse
import csv
import json
import wave
from pathlib import Path
import numpy as np

RANDOM_STATE = 42
EMOTIONS = ("angry", "happy", "sad", "neutral")


def generate_dummy_dataset(
    output_path: str | Path,
    samples_per_emotion: int = 40,
    seed: int = RANDOM_STATE,
) -> Path:
    """Create a reproducible feature dataset for an offline demonstration."""
    if samples_per_emotion < 2:
        raise ValueError("samples_per_emotion must be at least 2")
    rng = np.random.default_rng(seed)
    profiles = {
        "angry": (0.85, 0.80, 0.75),
        "happy": (0.70, 0.65, 0.80),
        "sad": (0.25, 0.25, 0.30),
        "neutral": (0.50, 0.45, 0.50),
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "emotion", "energy", "zcr", "spectral_centroid"])
        sample_id = 0
        for emotion in EMOTIONS:
            energy, zcr, centroid = profiles[emotion]
            for _ in range(samples_per_emotion):
                values = np.clip(
                    rng.normal((energy, zcr, centroid), 0.06), 0.0, 1.0
                )
                writer.writerow([sample_id, emotion, *values.round(6)])
                sample_id += 1
    return path


def load_feature_dataset(path: str | Path) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Load the dummy CSV format and return features, integer labels, classes."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Dataset is empty: {path}")
    required = {"emotion", "energy", "zcr", "spectral_centroid"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Dataset is missing columns: {', '.join(sorted(missing))}")
    classes = tuple(sorted({row["emotion"] for row in rows}))
    class_to_index = {name: index for index, name in enumerate(classes)}
    features = np.asarray(
        [[float(row[name]) for name in ("energy", "zcr", "spectral_centroid")] for row in rows],
        dtype=np.float64,
    )
    labels = np.asarray([class_to_index[row["emotion"]] for row in rows], dtype=np.int64)
    return features, labels, classes


def mfcc_from_wav(path: str | Path, sample_rate: int = 16000, n_mfcc: int = 13) -> np.ndarray:
    """Extract a compact MFCC-style vector from a PCM WAV without third-party code.

    For production datasets, librosa's mel filter bank is recommended.  This
    lightweight implementation keeps the demo runnable in a clean environment.
    """
    with wave.open(str(path), "rb") as audio:
        channels, width, source_rate, frames = (
            audio.getnchannels(),
            audio.getsampwidth(),
            audio.getframerate(),
            audio.readframes(audio.getnframes()),
        )
    if width != 2:
        raise ValueError("Only 16-bit PCM WAV files are supported")
    signal = np.frombuffer(frames, dtype="<i2").astype(np.float64) / 32768.0
    if channels > 1:
        signal = signal.reshape(-1, channels).mean(axis=1)
    if source_rate != sample_rate:
        indices = np.linspace(0, len(signal) - 1, max(1, int(len(signal) * sample_rate / source_rate)))
        signal = np.interp(indices, np.arange(len(signal)), signal)
    if len(signal) < 512:
        signal = np.pad(signal, (0, 512 - len(signal)))
    frame_count = max(1, 1 + (len(signal) - 512) // 256)
    frames_2d = np.stack([signal[i * 256 : i * 256 + 512] for i in range(frame_count)])
    spectrum = np.abs(np.fft.rfft(frames_2d * np.hanning(512), axis=1)) + 1e-8
    log_spectrum = np.log(spectrum)
    # DCT-II over the first frequency bins is a useful MFCC-style descriptor.
    basis = np.cos(np.pi * np.arange(n_mfcc)[:, None] * (2 * np.arange(log_spectrum.shape[1]) + 1)
                   / (2 * log_spectrum.shape[1]))
    return (log_spectrum @ basis.T).mean(axis=0)


class SoftmaxClassifier:
    """Small NumPy baseline used when TensorFlow is unavailable."""

    def __init__(self, learning_rate: float = 0.15, epochs: int = 250):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights: np.ndarray | None = None
        self.bias: np.ndarray | None = None

    def fit(self, features: np.ndarray, labels: np.ndarray, class_count: int) -> list[float]:
        if len(features) != len(labels):
            raise ValueError("features and labels must contain the same number of samples")
        mean, scale = features.mean(axis=0), features.std(axis=0)
        scale[scale == 0] = 1.0
        normalized = (features - mean) / scale
        self._mean, self._scale = mean, scale
        self.weights = np.zeros((features.shape[1], class_count))
        self.bias = np.zeros(class_count)
        one_hot = np.eye(class_count)[labels]
        losses = []
        for _ in range(self.epochs):
            logits = normalized @ self.weights + self.bias
            probabilities = _softmax(logits)
            losses.append(float(-np.mean(np.sum(one_hot * np.log(probabilities + 1e-12), axis=1))))
            gradient = (probabilities - one_hot) / len(labels)
            self.weights -= self.learning_rate * normalized.T @ gradient
            self.bias -= self.learning_rate * gradient.sum(axis=0)
        return losses

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.weights is None or self.bias is None:
            raise RuntimeError("Call fit before predict")
        normalized = (features - self._mean) / self._scale
        return np.argmax(_softmax(normalized @ self.weights + self.bias), axis=1)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def build_tensorflow_models(input_shape: tuple[int, ...], class_count: int):
    """Return CNN and BiLSTM models, importing TensorFlow only on request."""
    try:
        import tensorflow as tf  # pyright: ignore[reportMissingModuleSource]
    except ImportError as exc:
        raise RuntimeError("Install TensorFlow to build CNN/BiLSTM models") from exc
    cnn = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(class_count, activation="softmax"),
        ],
        name="mfcc_cnn",
    )
    cnn.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    lstm = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_shape[1], input_shape[0])),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(class_count, activation="softmax"),
        ],
        name="mfcc_bilstm",
    )
    lstm.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return cnn, lstm


def run_demo(dataset_path: Path, test_fraction: float = 0.2) -> dict:
    """Generate, train, evaluate, and persist a complete offline demo."""
    generate_dummy_dataset(dataset_path)
    features, labels, classes = load_feature_dataset(dataset_path)
    rng = np.random.default_rng(RANDOM_STATE)
    indices = rng.permutation(len(labels))
    split = int(len(labels) * (1 - test_fraction))
    train_idx, test_idx = indices[:split], indices[split:]
    model = SoftmaxClassifier()
    losses = model.fit(features[train_idx], labels[train_idx], len(classes))
    predictions = model.predict(features[test_idx])
    accuracy = float(np.mean(predictions == labels[test_idx]))
    metrics = {"classes": classes, "samples": len(labels), "test_accuracy": accuracy,
               "final_loss": losses[-1], "predictions": predictions.tolist()}
    dataset_path.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/dummy_emotions.csv"))
    args = parser.parse_args()
    metrics = run_demo(args.dataset)
    print(f"Dataset: {args.dataset} ({metrics['samples']} samples)")
    print(f"Classes: {', '.join(metrics['classes'])}")
    print(f"NumPy baseline test accuracy: {metrics['test_accuracy']:.3f}")
    print(f"Metrics: {args.dataset.with_suffix('.metrics.json')}")


if __name__ == "__main__":
    main()

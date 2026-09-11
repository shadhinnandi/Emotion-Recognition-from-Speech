# Speech Emotion Recognition

An end to end machine learning reference project for classifying emotions from
speech audio. The pipeline covers feature extraction, reproducible dataset
creation, model training, evaluation, and export of machine readable metrics.


## Overview

The project uses acoustic features associated with speech emotion, including
energy, zero crossing rate, spectral characteristics, and MFCC style
descriptors. It includes:

1. An offline NumPy baseline that runs without a deep learning framework.
2. A lightweight 16 bit PCM WAV feature extractor.
3. Optional TensorFlow CNN and bidirectional LSTM model builders.
4. Deterministic dummy data for repeatable local testing and CI smoke tests.
5. JSON output for downstream evaluation, dashboards, or API integration.

The demo recognizes four classes:

```text
angry · happy · neutral · sad
```

The label set can be extended for datasets such as RAVDESS, TESS, and EMODB.

## Setup and quick start

### Requirements

1. Python 3.10 or newer for the baseline
2. `pip`

The baseline does not require TensorFlow, librosa, scikit learn, or GPU
hardware.

### macOS/Linux

Run these commands from the project root:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install pip
python -m pip install -r requirements.txt
python3 emotion_recognition.py
```

On Windows PowerShell, activate the environment with:

```powershell
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python emotion_recognition.py
```

If `python3 emotion_recognition.py` reports `No module named 'numpy'`, the
virtual environment is either not activated or its dependencies were not
installed. Activate it and run:

```bash
source venv/bin/activate
python -m pip install -r requirements.txt
```

### Verify the installation

```bash
python -m py_compile emotion_recognition.py
python emotion_recognition.py
```

The command creates the following files:

```text
data/dummy_emotions.csv
data/dummy_emotions.metrics.json
```

Example output:

```text
Dataset: data/dummy_emotions.csv (160 samples)
Classes: angry, happy, neutral, sad
NumPy baseline test accuracy: 0.969
Metrics: data/dummy_emotions.metrics.json
```

The generated data uses a fixed random seed (`42`), so the pipeline is
reproducible across runs.

## Pipeline

```text
Speech audio
    ↓
WAV loading and signal normalization
    ↓
MFCC / acoustic feature extraction
    ↓
Train, validation, and test split
    ↓
NumPy baseline or TensorFlow CNN / BiLSTM
    ↓
Predictions and evaluation metrics
    ↓
JSON metrics export
```

### Feature extraction

`mfcc_from_wav()` provides a dependency light MFCC style descriptor for
16 bit PCM WAV files. For production experiments, the recommended approach is
to use librosa to extract:

1. MFCCs
2. First and second order MFCC deltas
3. Chroma features
4. Spectral contrast
5. Spectral centroid
6. Zero crossing rate
7. RMS energy

Audio features should be padded or truncated to a consistent number of time
frames before being passed to a neural network.

### Models

The project provides:

| Model | Purpose | Expected input |
| NumPy softmax classifier | Fast, dependency light baseline | Fixed length feature vectors |
| CNN | Learns local time frequency patterns in MFCC maps | `(samples, mfcc_bins, time_frames, 1)` |
| Bidirectional LSTM | Learns temporal dependencies across frames | `(samples, time_frames, mfcc_bins)` |

Build the optional TensorFlow models from Python:

```python
from emotion_recognition import build_tensorflow_models

cnn, bilstm = build_tensorflow_models(
    input_shape=(40, 174, 1),
    class_count=8,
)
```

Install the optional audio and deep learning dependencies with:

```bash
python -m pip install -r requirements.txt
```
TensorFlow availability depends on the Python version and operating system.
For the optional deep learning stack, use a Python version supported by the
TensorFlow release you select; the NumPy baseline remains the recommended
first verification step.

## Using real speech datasets

### Supported dataset direction

The project is structured to support common research datasets:

1. **RAVDESS**: emotional speech and song from multiple actors.
2. **TESS**: speech recordings with emotion labels.
3. **EMODB**: acted German speech with emotion annotations.

Always download datasets from their official or approved distribution source
and review the applicable license before use.

### Recommended workflow

1. Convert dataset filenames or directory labels into `(audio_path, emotion)`
   records.
2. Validate sample rate, channel count, duration, and file integrity.
3. Extract and normalize MFCC based features.
4. Split by speaker/actor rather than by file to prevent speaker leakage.
5. Train on the training split and tune using validation data.
6. Evaluate once on held out speakers.
7. Report accuracy, macro F1, per class precision and recall, and a confusion
   matrix.

Speaker independent evaluation is essential: a random file level split can
produce overly optimistic results when the same speaker appears in both train
and test data.

## Repository structure

```text
.
├── emotion_recognition.py          # Dataset, features, models, and CLI
├── README.md                       # Project documentation
├── requirements.txt                # All project dependencies
├── .gitignore                      # Local environments and generated caches
└── data/
    ├── dummy_emotions.csv          # Generated demo dataset
    └── dummy_emotions.metrics.json # Generated evaluation output
```

## Reproducibility and engineering notes

1. Randomness is controlled with a fixed seed for the demo workflow.
2. Input data is validated before training.
3. Model metrics are written as JSON for automation and integration.
4. Generated data is kept small so the project can be tested offline.
5. Real deployments should version datasets, preprocessing configuration,
  model artifacts, and evaluation reports separately.

## Limitations and responsible use

Emotion recognition from speech is an inference about vocal expression, not a
reliable measurement of a person's internal emotional state. Accuracy can vary
substantially with language, culture, speaker identity, acting style,
microphone quality, background noise, and annotation methodology.

This project must not be used as the sole basis for decisions involving
employment, healthcare, education, insurance, policing, access to services, or
other high impact outcomes. Before deployment, conduct subgroup analysis,
privacy review, consent review, security testing, and human impact assessment.

## License

No project license has been selected yet. Add a `LICENSE` file before
publishing or distributing this repository.

# Trained Model

This directory contains the pre-trained Convolutional Neural Network used by the Doodle Detective application.

## Model Details

| Property | Value |
|----------|-------|
| **Framework** | TensorFlow / Keras |
| **Format** | `.keras` (Keras v3) |
| **Input** | 28 × 28 grayscale bitmap |
| **Output** | 3-class softmax: Apple, Cat, Star |
| **Parameters** | ~243K total |
| **File size** | 1.4 MB |

## Usage

The model is loaded automatically by `run_gui.py` — no manual setup is required.

```python
from src.predict import load_model
model = load_model("models/doodle_classifier.keras")
```

## Why is the model included?

Including the trained model means the application is immediately usable after installation.  You can draw and receive predictions without needing to download the QuickDraw dataset or retrain the network.

The training notebook (`notebooks/DoodleDetective_Workflow.ipynb`) explains the architecture and training process if you wish to retrain or modify the model.
# Doodle Detective 🎨🔍

[![Python](https://img.shields.io/badge/Python-3.10_|_3.11_|_3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**An end-to-end deep learning project that classifies hand-drawn doodles using a Convolutional Neural Network and a desktop GUI built with Tkinter.**

> This is my first deep learning project, built during my AI & Data Science degree at BMSCE. It explores the complete workflow from dataset preparation and CNN training to deploying a trained model in a desktop application.

---

## Screenshots

| Drawing Canvas | Prediction Result | Confidence Chart |
|:---:|:---:|:---:|
| ![Application interface](assets/screenshots/app_interface.png) | ![Drawing an apple](assets/screenshots/drawing_apple.png) | ![Confidence chart](assets/screenshots/confidence_chart.png) |
| Complete application interface | Draw an apple, cat, or star | Bar chart shows model confidence per class |

| Full Interface | Drawing Examples |
|:---:|:---:|
| ![Prediction detail](assets/screenshots/prediction_detail.png) | ![Cat drawing](assets/screenshots/drawing_cat.png) ![Star drawing](assets/screenshots/drawing_star.png) |
| Prediction with confidence, fun fact and stats | The model classifies all three supported shapes |

---

## Overview

Doodle Detective lets you draw an **apple**, **cat**, or **star** on a canvas, and a CNN trained on the QuickDraw dataset predicts what you drew — in real time, with confidence scores, fun facts, and drawing statistics.

| Input | Output |
|-------|--------|
| Hand-drawn sketch (280×280) | Predicted class + confidence % |

### Dataset

<p align="center">
  <img src="assets/results/dataset_distribution.png" alt="PCA projection of the QuickDraw apple, cat and star drawings" width="700">
</p>

The model is trained on 405,543 QuickDraw bitmaps (Apple 144,722 · Cat 123,202 · Star 137,619). The plot shows the drawings projected onto their first two principal components, visualised in Tableau. The classes overlap heavily in this linear 2-D view, which is why a CNN that learns spatial features is needed rather than a simple linear classifier.

## Tech Stack

- **Python** — Core programming language
- **TensorFlow / Keras** — Deep learning framework for the CNN
- **Tkinter** — Desktop GUI toolkit (included with Python)
- **Pillow (PIL)** — Image processing and canvas management
- **NumPy** — Numerical array operations for preprocessing
- **Matplotlib** — Confidence score bar chart visualisation
- **Jupyter Notebook** — Interactive tutorial notebook

---

## Features

- **Interactive drawing canvas** — draw with adjustable pen thickness
- **Real-time prediction** — model runs automatically when you release the mouse
- **Confidence visualisation** — horizontal bar chart showing probabilities for each class
- **Fun facts** — learn something about the predicted object
- **Drawing statistics** — track how many of each class you've drawn
- **Modular codebase** — clean separation of concerns (preprocessing, prediction, GUI, visualisation)

---

## Architecture

### CNN Model

The model is a small Convolutional Neural Network trained on the [Google QuickDraw](https://quickdraw.withgoogle.com/data) dataset:

```
Input (28, 28, 1)
      ↓
Conv2D (32 filters, 3×3) + ReLU
      ↓
MaxPooling2D (2×2)
      ↓
Conv2D (64 filters, 3×3) + ReLU
      ↓
MaxPooling2D (2×2)
      ↓
Flatten
      ↓
Dense (64) + ReLU
      ↓
Dense (3) + Softmax
      ↓
Output: [Apple, Cat, Star]
```

- **Training data:** 405,543 QuickDraw drawings (Apple 144,722 · Cat 123,202 · Star 137,619)
- **Split:** 80/20 stratified train/test (324,434 / 81,109), `random_state=42`
- **Optimiser:** Adam, batch size 128, 5 epochs
- **Loss:** Sparse Categorical Crossentropy
- **Parameters:** 121,475
- **Format:** Saved as `.keras` (Keras v3 format)

### Application Pipeline

```
User draws on Tkinter canvas (280×280)
              ↓
PIL Image (grayscale, white background)
              ↓
Resize to 28×28 → Normalise [0,1] → Invert colours → Reshape (1, 28, 28, 1)
              ↓
CNN predicts probabilities for [Apple, Cat, Star]
              ↓
GUI updates: prediction label, confidence %, bar chart, fun fact, stats
```

---

## Repository Structure

```
DoodleDetective/
├── README.md                         # This file
├── LICENSE                           # MIT License
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Ignored files
├── run_gui.py                        # Entry point — launches the desktop app
│
├── src/                              # Python package
│   ├── __init__.py
│   ├── config.py                     # Constants: paths, class names, fun facts, colours
│   ├── preprocess.py                 # Image preprocessing (resize, normalise, invert)
│   ├── predict.py                    # Model loading + inference
│   ├── gui.py                        # Tkinter application (drawing canvas, results panel)
│   ├── visualization.py              # Matplotlib confidence bar chart
│   └── utils.py                      # Helper functions
│
├── models/
│   ├── README.md                     # Model details and usage
│   └── doodle_classifier.keras       # Pre-trained CNN model
│
├── notebooks/
│   ├── DoodleDetective_Workflow.ipynb  # Guided tutorial notebook
│   └── training.ipynb                  # Original training run (with outputs)
│
└── assets/
    ├── screenshots/                  # Application screenshots
    └── results/                      # Confusion matrix, dataset plot
```

### Why this structure?

- **`src/`** — Each module has a single responsibility. The pipeline reads top-to-bottom: config → preprocess → predict → gui → visualisation.
- **`models/`** — Standard location for trained artifacts. Includes its own `README.md` explaining the model.
- **`notebooks/`** — A guided tutorial notebook, plus the original training notebook with its recorded outputs.
- **`assets/`** — Screenshots and evaluation figures for documentation.
- **Root scripts** — `run_gui.py` is a minimal entry point that says "run this to start the app."

---

## Installation

### Prerequisites

- Python 3.10–3.12 (TensorFlow does not yet support Python 3.13)
- Tkinter (included with Python on macOS and Windows; on Linux: `sudo apt install python3-tk`)

### Setup

```bash
# Clone the repository
git clone https://github.com/kaustubh-commits/DoodleDetective.git
cd DoodleDetective

# (Optional) Create a virtual environment
python3 -m venv .venv

# Activate the environment
source .venv/bin/activate         # macOS / Linux
# .venv\Scripts\activate          # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Running the GUI

The trained model is included in the repository, so the application works immediately after installation — no dataset downloads or retraining required.

### macOS / Linux

```bash
python3 run_gui.py
```

### Windows

```bash
python run_gui.py
```

This opens a 1000×650 window where you can draw and receive predictions.

---

## Notebook Walkthrough

### Training run

[`notebooks/training.ipynb`](notebooks/training.ipynb) is the actual notebook the shipped model was trained in, with its original outputs (dataset shapes, train/test split, per-epoch accuracy). The final model is the one defined and trained in its second training cell (Dense 64, 5 epochs); the earlier cell is a first experiment with dropout and a Dense 128 layer. To re-run it, download `full_numpy_bitmap_apple.npy`, `full_numpy_bitmap_cat.npy` and `full_numpy_bitmap_star.npy` from the [QuickDraw numpy bitmaps](https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap) into the same folder as the notebook.

### Tutorial

Open the tutorial notebook to see the complete ML workflow:

```bash
jupyter notebook notebooks/DoodleDetective_Workflow.ipynb
```

The notebook covers:

1. **Introduction** — What the project does and why it matters
2. **Problem Statement** — The classification task and its challenges
3. **Dataset Overview** — Google QuickDraw dataset format and statistics
4. **Dataset Exploration** — Loading and visualising sample drawings
5. **Image Preprocessing** — Why we resize, normalise, and invert colours
6. **CNN Architecture** — Understanding the model layers and parameters
7. **Training Process** — Optimiser, loss function, and training loop
8. **Model Evaluation** — Accuracy, confusion matrix, and classification report
9. **Saving the Model** — The `.keras` format and why it's used
10. **Running Inference** — Testing the model on individual drawings
11. **Connecting to the GUI** — How the model integrates with Tkinter
12. **Conclusion** — Key takeaways and future directions

---

## Results

**Test accuracy: 98.61%** (loss 0.039) on 81,109 held-out QuickDraw drawings. This matches the final epoch of the [training run](notebooks/training.ipynb) and was re-verified by reloading the saved model on the same seeded split.

| Class | Precision | Recall | F1-score | Test samples |
|-------|:---------:|:------:|:--------:|-------------:|
| Apple | 0.994 | 0.985 | 0.990 | 28,945 |
| Cat   | 0.975 | 0.987 | 0.981 | 24,640 |
| Star  | 0.988 | 0.987 | 0.987 | 27,524 |

<p align="center">
  <img src="assets/results/confusion_matrix.png" alt="Confusion matrix on the test set" width="520">
</p>

Most errors involve **Cat**: apples and stars are occasionally predicted as cats (~1.1% each), which shows up as Cat's lower precision. Apple and Star are rarely confused with each other.

> **Note:** The test split was also passed as `validation_data` during training to monitor progress, so this figure is not from a fully untouched set. No hyperparameter tuning or early stopping was done against it.

Performance on drawings made in the GUI can be lower than these numbers, because mouse-drawn strokes on a 280×280 canvas don't match the centred, scaled QuickDraw bitmaps exactly (see *Future Improvements*).

---

## Future Improvements

These are ideas I'd explore if I were to continue this project:

- **More QuickDraw classes** — Expand beyond 3 classes to 10+ categories
- **Better preprocessing** — Centre the strokes within the canvas before prediction
- **Web deployment** — Convert the model to TensorFlow.js for browser-based inference
- **Mobile app** — Deploy on iOS/Android using TensorFlow Lite
- **Improved UI** — Add undo/redo, colour options, eraser tool
- **Data augmentation during training** — Improve robustness to varied drawing styles

---

## Acknowledgements

- **Google QuickDraw** — The dataset that made this project possible ([quickdraw.withgoogle.com/data](https://quickdraw.withgoogle.com/data))
- **TensorFlow / Keras** — Deep learning framework used for the CNN
- **Tkinter** — Python's built-in GUI toolkit
- **Matplotlib** — Visualisation library for the confidence chart
- **Pillow (PIL)** — Image processing library

---

*Doodle Detective is my first deep learning project, built during my AI & Data Science degree at BMSCE. Questions or feedback? Open an issue!*
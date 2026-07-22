"""
Model loading and inference for the Doodle Detective.

Provides two public functions:
- ``load_model()`` — loads the pre-trained Keras model from disk.
- ``predict_doodle()`` — runs inference on a preprocessed image tensor
  and returns the predicted class, confidence, and full probability vector.
"""

import os
from typing import Optional, Dict, Union

import numpy as np
import tensorflow as tf

from src.config import MODEL_PATH, CLASS_NAMES


def load_model(model_path: str = MODEL_PATH) -> tf.keras.Model:
    """
    Load the pre-trained Keras model from disk.

    Parameters
    ----------
    model_path : str
        Path to the ``.keras`` model file. Defaults to the project's
        standard model location.

    Returns
    -------
    tf.keras.Model
        The loaded Keras model.

    Raises
    ------
    FileNotFoundError
        If the model file does not exist at the given path.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at: {model_path}\n"
            f"Run the training notebook or place a trained .keras file there."
        )

    print(f"[DoodleDetective] Loading model from {model_path}")
    return tf.keras.models.load_model(model_path)


def predict_doodle(
    model: tf.keras.Model,
    image_tensor: np.ndarray,
) -> Dict[str, Union[str, float, np.ndarray]]:
    """
    Run the model on a preprocessed image tensor and return results.

    Parameters
    ----------
    model : tf.keras.Model
        The loaded Keras model.
    image_tensor : np.ndarray
        4D tensor of shape ``(1, 28, 28, 1)`` as produced by
        ``preprocess.preprocess_for_model()``.

    Returns
    -------
    dict
        Keys:
        - ``"class_name"`` (str): Predicted class label.
        - ``"confidence"`` (float): Confidence score for the predicted class
          (as a percentage 0–100).
        - ``"probabilities"`` (np.ndarray): Raw probability vector of shape
          ``(3,)`` — one value per class in ``CLASS_NAMES`` order.
    """
    predictions: np.ndarray = model.predict(image_tensor, verbose=0)
    probabilities: np.ndarray = predictions[0]

    predicted_idx: int = int(np.argmax(probabilities))
    class_name: str = CLASS_NAMES[predicted_idx]
    confidence: float = float(probabilities[predicted_idx]) * 100.0

    return {
        "class_name": class_name,
        "confidence": confidence,
        "probabilities": probabilities,
    }
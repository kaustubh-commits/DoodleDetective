"""
Image preprocessing for the Doodle Detective CNN.

Transforms a raw PIL drawing (280x280 grayscale) into the normalised
28x28 tensor expected by the trained Keras model.

The pipeline mirrors the original project's approach:
1. Resize to 28x28 using LANCZOS resampling.
2. Convert to numpy array.
3. Normalise pixel values from [0, 255] to [0.0, 1.0].
4. Invert so that strokes (originally black on white) become
   white on black, matching the QuickDraw training data.
5. Reshape to (1, 28, 28, 1) for batch inference.
"""

import numpy as np
from PIL import Image

from src.config import CANVAS_SIZE, INPUT_SIZE, NUM_CHANNELS


def resize_image(image: Image.Image, size: int = INPUT_SIZE) -> Image.Image:
    """
    Resize a PIL image to the target size using LANCZOS resampling.

    Parameters
    ----------
    image : PIL.Image
        The input image (typically 280x280 grayscale).
    size : int
        Target width and height in pixels.

    Returns
    -------
    PIL.Image
        Resized image.
    """
    return image.resize((size, size), Image.LANCZOS)


def image_to_array(image: Image.Image) -> np.ndarray:
    """
    Convert a PIL grayscale image to a normalised numpy array.

    Parameters
    ----------
    image : PIL.Image
        Grayscale PIL image.

    Returns
    -------
    np.ndarray
        2D array with values in [0.0, 1.0].
    """
    return np.array(image, dtype=np.float32) / 255.0


def invert_colors(array: np.ndarray) -> np.ndarray:
    """
    Invert pixel values so that strokes become bright on a dark background.

    The original project drew black lines on a white canvas, but the
    QuickDraw dataset has white strokes on a black background.

    Parameters
    ----------
    array : np.ndarray
        Image array with values in [0.0, 1.0].

    Returns
    -------
    np.ndarray
        Inverted array.
    """
    return 1.0 - array


def add_batch_dimension(array: np.ndarray) -> np.ndarray:
    """
    Add the batch and channel dimensions expected by Keras.

    Input shape (28, 28) → output shape (1, 28, 28, 1).

    Parameters
    ----------
    array : np.ndarray
        2D image array.

    Returns
    -------
    np.ndarray
        4D tensor ready for ``model.predict()``.
    """
    return array.reshape(1, INPUT_SIZE, INPUT_SIZE, NUM_CHANNELS)


def preprocess_for_model(image: Image.Image) -> np.ndarray:
    """
    Full preprocessing pipeline: resize → array → invert → reshape.

    This is the single function called by the GUI and the notebook.

    Parameters
    ----------
    image : PIL.Image
        Grayscale PIL image drawn by the user (280x280).

    Returns
    -------
    np.ndarray
        4D tensor of shape (1, 28, 28, 1) ready for model inference.
    """
    resized = resize_image(image)
    array = image_to_array(resized)
    inverted = invert_colors(array)
    return add_batch_dimension(inverted)
"""
Configuration constants for the Doodle Detective application.

This module centralises all magic numbers, file paths, colour codes,
class names, and fun facts so they can be maintained in one place.
"""

import os
from typing import List, Dict, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

#: Root directory of the DoodleDetective project (two levels up from src/).
ROOT_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Directory where trained models are stored.
MODELS_DIR: str = os.path.join(ROOT_DIR, "models")

#: Path to the pre-trained Keras model file.
MODEL_PATH: str = os.path.join(MODELS_DIR, "doodle_classifier.keras")

# ---------------------------------------------------------------------------
# Image dimensions
# ---------------------------------------------------------------------------

#: Width and height of the drawing canvas in pixels.
CANVAS_SIZE: int = 280

#: Input size expected by the CNN (grayscale, 28x28).
INPUT_SIZE: int = 28

#: Number of colour channels (grayscale).
NUM_CHANNELS: int = 1

# ---------------------------------------------------------------------------
# Model / classes
# ---------------------------------------------------------------------------

#: The three classes the model was trained on.
CLASS_NAMES: List[str] = ["Apple", "Cat", "Star"]

#: Probability thresholds used by the GUI.
CONFIDENCE_HIGH: float = 0.50   # above this → count toward stats
CONFIDENCE_LOW: float = 0.30    # above this → show a fun fact

# ---------------------------------------------------------------------------
# UI colours
# ---------------------------------------------------------------------------

BG_COLOR: str = "#F0F0F0"
CANVAS_COLOR: str = "#FFFFFF"
STROKE_COLOR: str = "#000000"

#: Default stroke width for the drawing pen.
DEFAULT_STROKE_WIDTH: int = 15
MIN_STROKE_WIDTH: int = 5
MAX_STROKE_WIDTH: int = 30

# ---------------------------------------------------------------------------
# Window dimensions
# ---------------------------------------------------------------------------

WINDOW_WIDTH: int = 1000
WINDOW_HEIGHT: int = 650

# ---------------------------------------------------------------------------
# Fun facts  (preserved from the original project)
# ---------------------------------------------------------------------------

FUN_FACTS: Dict[str, List[str]] = {
    "Apple": [
        "Apples float in water because they're 25% air!",
        "There are over 7,500 varieties of apples grown around the world.",
        "The science of apple growing is called pomology.",
        "It takes about 36 apples to create one gallon of apple cider.",
        "An apple tree can live for more than 100 years!",
    ],
    "Cat": [
        "Cats sleep for 70% of their lives!",
        "A cat's purr vibrates at a frequency that promotes bone healing.",
        "Cats have 32 muscles in each ear to control movement.",
        "A group of cats is called a 'clowder'.",
        "Cats can't taste sweetness — they lack sweet taste receptors.",
    ],
    "Star": [
        "The light from the nearest star takes over 4 years to reach Earth.",
        "There are more stars in the universe than grains of sand on Earth!",
        "Stars don't actually twinkle — that's caused by our atmosphere.",
        "Some stars are so large that they could fit more than a billion Suns inside them!",
        "A neutron star can rotate up to 600 times per second!",
    ],
}
"""
Confidence score visualization for the Doodle Detective.

Produces a horizontal bar chart showing the model's confidence for each
class. This chart is embedded in the Tkinter GUI via ``FigureCanvasTkAgg``.

Preserves the original project's colour scheme and layout.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend required for Tkinter embedding
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from src.config import CLASS_NAMES

#: Bar colours, one per class — preserved from the original project.
BAR_COLORS: list[str] = ["#FF9999", "#99CC99", "#9999FF"]


def create_confidence_chart(
    probabilities: np.ndarray,
    figsize: tuple[float, float] = (4.0, 2.5),
    facecolor: str = "#F0F8FF",
) -> Figure:
    """
    Build a horizontal bar chart of class confidence scores.

    Parameters
    ----------
    probabilities : np.ndarray
        Array of shape ``(3,)`` with values in ``[0.0, 1.0]``, one per
        class in ``CLASS_NAMES`` order.
    figsize : tuple[float, float]
        Width and height of the figure in inches.
    facecolor : str
        Background colour of the figure.

    Returns
    -------
    matplotlib.figure.Figure
        A Matplotlib figure ready for embedding in a Tkinter canvas.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    fig.patch.set_facecolor(facecolor)

    y_pos = np.arange(len(CLASS_NAMES))

    # Horizontal bars
    ax.barh(y_pos, probabilities, align="center", color=BAR_COLORS)

    # Y-axis: class names, top-to-bottom
    ax.set_yticks(y_pos)
    ax.set_yticklabels(CLASS_NAMES)
    ax.invert_yaxis()

    # Percentage label on each bar
    for i, val in enumerate(probabilities):
        ax.text(val + 0.01, i, f"{val:.0%}", va="center")

    ax.set_title("Confidence Scores")
    ax.set_xlim(0.0, 1.0)

    fig.tight_layout()
    return fig


def update_chart(fig: Figure, probabilities: np.ndarray) -> None:
    """
    Redraw an existing confidence chart with new probabilities.

    Reuses the same figure to avoid recreating the Tkinter embedding.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure returned by ``create_confidence_chart()``.
    probabilities : np.ndarray
        Updated probability vector.
    """
    ax = fig.axes[0]
    ax.clear()

    y_pos = np.arange(len(CLASS_NAMES))
    ax.barh(y_pos, probabilities, align="center", color=BAR_COLORS)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(CLASS_NAMES)
    ax.invert_yaxis()

    for i, val in enumerate(probabilities):
        ax.text(val + 0.01, i, f"{val:.0%}", va="center")

    ax.set_title("Confidence Scores")
    ax.set_xlim(0.0, 1.0)
    fig.tight_layout()
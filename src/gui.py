"""
Graphical user interface for the Doodle Detective application.

Built with Tkinter and Matplotlib, this module provides an interactive
drawing canvas where users can sketch an apple, cat, or star and receive
real-time CNN predictions.

Preserves the original project's layout and behaviour while organising
the code into clearly separated responsibilities.
"""

import random
from typing import Optional

import numpy as np
import tkinter as tk
from tkinter import Canvas, Button, Frame, Label, Scale, HORIZONTAL, ttk
from PIL import Image, ImageDraw

from src.config import (
    CANVAS_SIZE,
    INPUT_SIZE,
    CLASS_NAMES,
    FUN_FACTS,
    BG_COLOR,
    CANVAS_COLOR,
    STROKE_COLOR,
    DEFAULT_STROKE_WIDTH,
    MIN_STROKE_WIDTH,
    MAX_STROKE_WIDTH,
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    MODEL_PATH,
)
from src.preprocess import preprocess_for_model
from src.predict import load_model, predict_doodle
from src.visualization import create_confidence_chart, update_chart

# ---------------------------------------------------------------------------
# Colour palette — ensures readable text across all UI sections
# ---------------------------------------------------------------------------
# Dark text colour for all labels on light backgrounds
TEXT_PRIMARY: str = "#2C3E50"
TEXT_SECONDARY: str = "#555555"
TEXT_ACCENT: str = "#2980B9"

# Panel background colours (preserved from original)
PANEL_PREDICTION: str = "#E8F4FD"
PANEL_FACT: str = "#FFF8EE"
PANEL_CHART: str = "#F0F8FF"
PANEL_STATS: str = "#EEF9EE"

# Button colours with guaranteed readable text
BTN_CLEAR_BG: str = "#E74C3C"
BTN_CLEAR_FG: str = "#FFFFFF"
BTN_PREDICT_BG: str = "#27AE60"
BTN_PREDICT_FG: str = "#FFFFFF"

# Section heading colour (consistent across panels)
HEADING_COLOR: str = "#2C3E50"


class DoodleDetectiveApp:
    """
    Main application window for the Doodle Detective.

    Provides a 280×280 drawing canvas, prediction results panel,
    confidence bar chart, fun facts, and drawing statistics.

    Parameters
    ----------
    root : tkinter.Tk
        The root Tkinter window.
    model_path : str, optional
        Path to the trained ``.keras`` model file. Defaults to the
        project-standard location (``models/doodle_classifier.keras``).
    """

    def __init__(self, root: tk.Tk, model_path: Optional[str] = None) -> None:
        self.root = root
        self.root.title("Doodle Detective")
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.configure(bg=BG_COLOR)

        # ------------------------------------------------------------------
        # Drawing state
        # ------------------------------------------------------------------
        self.drawing: bool = False
        self.last_x: int = 0
        self.last_y: int = 0
        self.stroke_width: int = DEFAULT_STROKE_WIDTH

        # PIL image that mirrors the Tkinter canvas (for model input)
        self.image: Image.Image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=255)
        self.draw: ImageDraw.ImageDraw = ImageDraw.Draw(self.image)

        # Drawing statistics
        self.drawing_counts: dict[str, int] = {name: 0 for name in CLASS_NAMES}

        # ------------------------------------------------------------------
        # Load the trained model
        # ------------------------------------------------------------------
        self.model = load_model(model_path or MODEL_PATH)

        # ------------------------------------------------------------------
        # Build the UI
        # ------------------------------------------------------------------
        self._build_ui()

    # ======================================================================
    # UI Construction
    # ======================================================================

    def _build_ui(self) -> None:
        """Construct the entire user interface."""
        main_container = Frame(self.root, bg=BG_COLOR)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self._build_title(main_container)
        left_panel, right_panel = self._build_content(main_container)
        self._build_left_panel(left_panel)
        self._build_right_panel(right_panel)

    def _build_title(self, parent: Frame) -> None:
        """Top section: title and subtitle."""
        top_frame = Frame(parent, bg=BG_COLOR)
        top_frame.pack(fill=tk.X, pady=5)

        title_label = Label(
            top_frame,
            text="Doodle Detective",
            font=("Arial", 24, "bold"),
            bg=BG_COLOR,
            fg=TEXT_PRIMARY,
        )
        title_label.pack()

        subtitle = Label(
            top_frame,
            text="Draw an apple, cat, or star and watch the AI guess what it is!",
            font=("Arial", 12),
            bg=BG_COLOR,
            fg=TEXT_SECONDARY,
        )
        subtitle.pack(pady=5)

    def _build_content(self, parent: Frame) -> tuple[Frame, Frame]:
        """Middle section: left (drawing) and right (results) panels.

        Returns
        -------
        tuple[Frame, Frame]
            (left_panel, right_panel) as explicit references.
        """
        content = Frame(parent, bg=BG_COLOR)
        content.pack(fill=tk.BOTH, expand=True)

        left_panel = Frame(content, bg=BG_COLOR, bd=2, relief=tk.RIDGE)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        right_panel = Frame(content, bg=BG_COLOR, bd=2, relief=tk.RIDGE)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        return left_panel, right_panel

    def _build_left_panel(self, parent: Frame) -> None:
        """Left panel: pen controls and drawing canvas."""
        # -- Pen thickness slider -----------------------------------------
        tools_frame = Frame(parent, bg=BG_COLOR)
        tools_frame.pack(fill=tk.X, pady=5)

        thickness_label = Label(
            tools_frame,
            text="Pen Size:",
            bg=BG_COLOR,
            font=("Arial", 10),
            fg=TEXT_PRIMARY,
        )
        thickness_label.pack(side=tk.LEFT, padx=5)

        self.thickness_slider = Scale(
            tools_frame,
            from_=MIN_STROKE_WIDTH,
            to=MAX_STROKE_WIDTH,
            orient=HORIZONTAL,
            length=150,
            bg=BG_COLOR,
            fg=TEXT_PRIMARY,
            troughcolor="#DDDDDD",
            command=self._update_thickness,
        )
        self.thickness_slider.set(DEFAULT_STROKE_WIDTH)
        self.thickness_slider.pack(side=tk.LEFT, padx=5)

        # -- Drawing canvas -----------------------------------------------
        canvas_frame = Frame(parent, bg="#DDDDDD", bd=1, relief=tk.SUNKEN)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.canvas = Canvas(
            canvas_frame,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg=CANVAS_COLOR,
            cursor="pencil",
            highlightthickness=0,
        )
        self.canvas.pack(padx=10, pady=10)

        # Mouse bindings
        self.canvas.bind("<Button-1>", self._start_draw)
        self.canvas.bind("<B1-Motion>", self._draw_line)
        self.canvas.bind("<ButtonRelease-1>", self._end_draw)

        # -- Buttons ------------------------------------------------------
        btn_frame = Frame(parent, bg=BG_COLOR)
        btn_frame.pack(fill=tk.X, pady=5)

        self.clear_button = Button(
            btn_frame,
            text="Clear Canvas",
            font=("Arial", 12, "bold"),
            bg=BTN_CLEAR_BG,
            fg=BTN_CLEAR_FG,
            activebackground="#C0392B",
            activeforeground="#FFFFFF",
            relief=tk.RAISED,
            bd=2,
            padx=10,
            command=self.clear_canvas,
        )
        self.clear_button.pack(side=tk.LEFT, padx=10, expand=True, fill=tk.X)

        self.predict_button = Button(
            btn_frame,
            text="Predict",
            font=("Arial", 12, "bold"),
            bg=BTN_PREDICT_BG,
            fg=BTN_PREDICT_FG,
            activebackground="#1E8449",
            activeforeground="#FFFFFF",
            relief=tk.RAISED,
            bd=2,
            padx=10,
            command=self._predict_and_update,
        )
        self.predict_button.pack(side=tk.RIGHT, padx=10, expand=True, fill=tk.X)

    def _build_right_panel(self, parent: Frame) -> None:
        """Right panel: prediction results, fun fact, chart, stats."""
        # -- Prediction result --------------------------------------------
        pred_frame = Frame(parent, bg=PANEL_PREDICTION, bd=2, relief=tk.GROOVE)
        pred_frame.pack(fill=tk.X, pady=10, padx=10)

        Label(
            pred_frame,
            text="I think you drew a:",
            font=("Arial", 12, "bold"),
            bg=PANEL_PREDICTION,
            fg=HEADING_COLOR,
        ).pack(pady=5)

        self.prediction_label = Label(
            pred_frame,
            text="Draw something!",
            font=("Arial", 24, "bold"),
            bg=PANEL_PREDICTION,
            fg=TEXT_ACCENT,
        )
        self.prediction_label.pack(pady=5)

        self.confidence_label = Label(
            pred_frame,
            text="",
            font=("Arial", 11),
            bg=PANEL_PREDICTION,
            fg=TEXT_PRIMARY,
        )
        self.confidence_label.pack(pady=5)

        # -- Fun fact -----------------------------------------------------
        fact_frame = Frame(parent, bg=PANEL_FACT, bd=2, relief=tk.GROOVE)
        fact_frame.pack(fill=tk.X, pady=10, padx=10)

        Label(
            fact_frame,
            text="Fun Fact:",
            font=("Arial", 12, "bold"),
            bg=PANEL_FACT,
            fg=HEADING_COLOR,
        ).pack(pady=5)

        self.fact_label = Label(
            fact_frame,
            text="Draw to see a fun fact!",
            font=("Arial", 11),
            bg=PANEL_FACT,
            fg=TEXT_PRIMARY,
            wraplength=300,
            justify=tk.LEFT,
        )
        self.fact_label.pack(pady=5, padx=10, fill=tk.X)

        # -- Confidence chart (Matplotlib) --------------------------------
        viz_frame = Frame(parent, bg=PANEL_CHART, bd=2, relief=tk.GROOVE)
        viz_frame.pack(fill=tk.BOTH, expand=True, pady=10, padx=10)

        Label(
            viz_frame,
            text="Confidence Scores",
            font=("Arial", 12, "bold"),
            bg=PANEL_CHART,
            fg=HEADING_COLOR,
        ).pack(pady=5)

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        self.chart_figure = create_confidence_chart(
            np.array([0.33, 0.33, 0.34])
        )
        self.chart_canvas = FigureCanvasTkAgg(self.chart_figure, master=viz_frame)
        self.chart_canvas.get_tk_widget().pack(
            fill=tk.BOTH, expand=True, padx=5, pady=5
        )

        # -- Drawing stats ------------------------------------------------
        stats_frame = Frame(parent, bg=PANEL_STATS, bd=2, relief=tk.GROOVE)
        stats_frame.pack(fill=tk.X, pady=10, padx=10)

        Label(
            stats_frame,
            text="Your Drawing Stats:",
            font=("Arial", 12, "bold"),
            bg=PANEL_STATS,
            fg=HEADING_COLOR,
        ).pack(pady=5)

        self.stats_label = Label(
            stats_frame,
            text=self._format_stats(),
            font=("Arial", 12),
            bg=PANEL_STATS,
            fg=TEXT_PRIMARY,
        )
        self.stats_label.pack(pady=5)

    # ======================================================================
    # Drawing Event Handlers
    # ======================================================================

    def _update_thickness(self, val: str) -> None:
        """Update stroke width from slider."""
        self.stroke_width = int(float(val))

    def _start_draw(self, event: tk.Event) -> None:
        """Begin drawing at the cursor position."""
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y

    def _draw_line(self, event: tk.Event) -> None:
        """Draw a line segment on both the Tkinter canvas and PIL image."""
        if not self.drawing:
            return

        # Tkinter canvas
        self.canvas.create_line(
            self.last_x,
            self.last_y,
            event.x,
            event.y,
            fill=STROKE_COLOR,
            width=self.stroke_width,
            capstyle=tk.ROUND,
            smooth=True,
        )

        # PIL image (for model input)
        self.draw.line(
            [self.last_x, self.last_y, event.x, event.y],
            fill=0,
            width=self.stroke_width,
        )

        self.last_x = event.x
        self.last_y = event.y

    def _end_draw(self, event: tk.Event) -> None:
        """End drawing and trigger prediction."""
        self.drawing = False
        self._predict_and_update()

    # ======================================================================
    # Prediction & UI Update
    # ======================================================================

    def _predict_and_update(self) -> None:
        """Preprocess the canvas, run inference, and update all UI elements."""
        # Preprocess
        tensor = preprocess_for_model(self.image)

        # Predict
        result = predict_doodle(self.model, tensor)

        predicted_class: str = result["class_name"]
        confidence: float = result["confidence"]
        probabilities: np.ndarray = result["probabilities"]

        # -- Update labels ------------------------------------------------
        self.prediction_label.config(text=predicted_class)
        self.confidence_label.config(text=f"Confidence: {confidence:.1f}%")

        # -- Update chart -------------------------------------------------
        update_chart(self.chart_figure, probabilities)
        self.chart_canvas.draw()

        # -- Update drawing stats (only if confident enough) --------------
        if confidence > CONFIDENCE_HIGH * 100:
            self.drawing_counts[predicted_class] += 1
            self.stats_label.config(text=self._format_stats())

        # -- Update fun fact (only if reasonably confident) ---------------
        if confidence > CONFIDENCE_LOW * 100:
            fact = random.choice(FUN_FACTS[predicted_class])
            self.fact_label.config(text=fact)
        else:
            self.fact_label.config(
                text="Hmm, I'm not quite sure what that is. Try drawing clearer!"
            )

    # ======================================================================
    # Public API
    # ======================================================================

    def clear_canvas(self) -> None:
        """Clear the drawing canvas and reset all output labels."""
        self.canvas.delete("all")
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=255)
        self.draw = ImageDraw.Draw(self.image)

        self.prediction_label.config(text="Draw something!")
        self.confidence_label.config(text="")
        self.fact_label.config(text="Draw to see a fun fact!")

        # Reset chart to uniform distribution
        update_chart(self.chart_figure, np.array([0.33, 0.33, 0.34]))
        self.chart_canvas.draw()

    def _format_stats(self) -> str:
        """Format the drawing statistics as a display string."""
        return " | ".join(
            f"{name}: {count}"
            for name, count in self.drawing_counts.items()
        )
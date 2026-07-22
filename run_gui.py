#!/usr/bin/env python3
"""
Entry point for the Doodle Detective desktop application.

Usage
-----
    python run_gui.py

This loads the pre-trained Keras model and opens a Tkinter window
where you can draw and receive real-time predictions.
"""

import sys
import os

# Ensure the project root is on sys.path so that ``import src`` works
# when running ``python run_gui.py`` from the project root.
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import tkinter as tk
from src.gui import DoodleDetectiveApp


def main() -> None:
    """Launch the Doodle Detective application."""
    root = tk.Tk()
    app = DoodleDetectiveApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
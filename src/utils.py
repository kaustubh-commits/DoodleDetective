"""
Utility functions for the Doodle Detective application.

Small, reusable helpers that don't belong to any single module.
"""

import os


def ensure_dir(path: str) -> None:
    """Create a directory if it doesn't already exist."""
    os.makedirs(path, exist_ok=True)


def safe_path(*parts: str) -> str:
    """
    Join path components safely.

    Parameters
    ----------
    *parts : str
        One or more path segments to join.

    Returns
    -------
    str
        Normalised absolute path.
    """
    return os.path.normpath(os.path.join(*parts))
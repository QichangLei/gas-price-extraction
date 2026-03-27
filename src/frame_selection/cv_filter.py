"""
cv_filter.py
─────────────
Stage 1 of the frame selection cascade.

Uses traditional CV to cheaply eliminate frames that are too blurry,
too dark, or structurally empty to ever contain a readable price sign.
No API calls — runs on every candidate frame for free.

Metrics used:
  sharpness    — Laplacian variance; high = sharp edges (text is readable)
  edge_density — fraction of Canny edge pixels; high = structured content
"""

from __future__ import annotations
import cv2
import numpy as np

# ── Default thresholds ────────────────────────────────────────────────────────
# Tuned empirically: real price signs in dashcam footage score well above these.
MIN_SHARPNESS    = 50.0    # Laplacian variance; below this = too blurry
MIN_EDGE_DENSITY = 0.015   # fraction of edge pixels; below this = too empty


def compute_sharpness(frame_bgr: np.ndarray) -> float:
    """Laplacian variance of the grayscale frame. Higher = sharper."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_edge_density(frame_bgr: np.ndarray) -> float:
    """Fraction of pixels that are Canny edges. Higher = more structure/text."""
    gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return float(np.count_nonzero(edges)) / edges.size


def passes_cv_filter(
    frame_bgr:        np.ndarray,
    min_sharpness:    float = MIN_SHARPNESS,
    min_edge_density: float = MIN_EDGE_DENSITY,
) -> tuple[bool, dict]:
    """
    Run Stage 1 CV filter on a single frame.

    Returns
    -------
    (passed, metrics)
      passed  — True if the frame survives to Stage 2
      metrics — dict with sharpness and edge_density scores
    """
    sharpness    = compute_sharpness(frame_bgr)
    edge_density = compute_edge_density(frame_bgr)
    passed = sharpness >= min_sharpness and edge_density >= min_edge_density
    return passed, {
        "sharpness":    round(sharpness, 2),
        "edge_density": round(edge_density, 4),
    }

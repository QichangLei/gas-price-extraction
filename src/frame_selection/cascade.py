"""
cascade.py
──────────
Three-stage frame selection cascade.

Stage 1 — CV filter (free)
    Laplacian sharpness + Canny edge density.
    Drops blurry / structurally empty frames instantly.

Stage 2 — Cheap model (gemini-2.0-flash-lite)
    Binary yes/no: "does this frame contain a gas price sign?"
    Runs only on Stage 1 survivors. Returns confidence 0–100.

Stage 3 — Full model (gemini-3-flash-preview)
    Runs on frames that are:
      • Low-confidence from Stage 2 (confidence < CONF_THRESHOLD)
      • GPS-flagged (truck within GPS_RADIUS_M of a known station)
      • Temporal neighbours of any Stage 2 positive (within ±TEMPORAL_WINDOW_SEC)

Selected frames are saved as JPEGs to output_dir.
A metadata CSV is written alongside them.
"""

from __future__ import annotations
import csv
import io
import json
import logging
import re
import time
from pathlib import Path

import cv2
import numpy as np
from google import genai
from google.genai import types

from .cv_filter  import passes_cv_filter
from .gps_filter import load_stations, load_gps_track, get_frame_gps, is_near_station

log = logging.getLogger(__name__)

# ── Models ────────────────────────────────────────────────────────────────────
CHEAP_MODEL  = "gemini-2.5-flash-lite"
FULL_MODEL   = "gemini-3-flash-preview"

# ── Thresholds ────────────────────────────────────────────────────────────────
CONF_THRESHOLD      = 70     # Stage 2 confidence below this → send to Stage 3
GPS_RADIUS_M        = 500.0  # metres — GPS proximity radius
TEMPORAL_WINDOW_SEC = 30.0   # seconds — neighbour window around Stage 2 positives

# ── Prompts ───────────────────────────────────────────────────────────────────
CHEAP_PROMPT = """Look at this dashcam frame. Reply with JSON only — no other text.

{"has_price_sign": true or false, "confidence": 0-100}

true  = a gas station fuel price display board is clearly or partially visible.
false = no gas station price sign visible.

confidence = how certain you are (100 = absolutely sure, 0 = no idea)."""

FULL_PROMPT = """Carefully examine this dashcam frame.
Reply with JSON only — no other text.

{"has_price_sign": true or false, "confidence": 0-100, "reason": "one sentence"}

true  = a gas station fuel price display board is visible (even partially or at an angle).
false = no gas station price sign visible.

Be thorough — check all areas of the frame."""


# ── Gemini helpers ────────────────────────────────────────────────────────────

def _frame_to_jpeg_bytes(frame_bgr: np.ndarray) -> bytes:
    pil_buf = io.BytesIO()
    from PIL import Image
    Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).save(
        pil_buf, format="JPEG", quality=85
    )
    return pil_buf.getvalue()


def _query_model(client: genai.Client, model: str, frame_bgr: np.ndarray,
                 prompt: str, retries: int = 2) -> dict:
    jpeg = _frame_to_jpeg_bytes(frame_bgr)
    for attempt in range(1, retries + 1):
        try:
            response = client.models.generate_content(
                model    = model,
                contents = [
                    types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"),
                    prompt,
                ],
                config = types.GenerateContentConfig(
                    thinking_config = types.ThinkingConfig(thinking_budget=0),
                    temperature     = 1.0,
                ),
            )
            raw = response.text.strip()
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw,
                         flags=re.MULTILINE).strip()
            return json.loads(raw)
        except Exception as exc:
            log.warning("Model %s attempt %d/%d failed: %s", model, attempt, retries, exc)
            if attempt < retries:
                time.sleep(2)
    return {"has_price_sign": False, "confidence": 0, "reason": "api_error"}


# ── Temporal neighbour expansion ──────────────────────────────────────────────

def _expand_temporal_neighbours(
    positive_frame_indices: list[int],
    all_candidate_indices:  list[int],
    fps:                    float,
    window_sec:             float = TEMPORAL_WINDOW_SEC,
) -> set[int]:
    """
    Given a list of Stage 2 positive frame indices, return the set of all
    candidate frame indices within ±window_sec of any positive.
    """
    window_frames = window_sec * fps
    neighbours: set[int] = set()
    for pos_idx in positive_frame_indices:
        for cand_idx in all_candidate_indices:
            if abs(cand_idx - pos_idx) <= window_frames:
                neighbours.add(cand_idx)
    return neighbours


# ── Main cascade ──────────────────────────────────────────────────────────────

def run_cascade(
    video_path:      str | Path,
    output_dir:      str | Path,
    stations_csv:    str | Path | None = None,
    gps_track_csv:   str | Path | None = None,
    every_n:         int   = 5,
    gps_radius_m:    float = GPS_RADIUS_M,
    conf_threshold:  int   = CONF_THRESHOLD,
    temporal_window: float = TEMPORAL_WINDOW_SEC,
    api_key:         str | None = None,
) -> Path:
    """
    Run the three-stage cascade on a video and save selected frames.

    Returns path to the metadata CSV.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load optional GPS data ────────────────────────────────────────────────
    stations  = load_stations(stations_csv)   if stations_csv  else []
    gps_track = load_gps_track(gps_track_csv) if gps_track_csv else []

    if stations:
        log.info("Loaded %d known stations", len(stations))
    if gps_track:
        log.info("Loaded GPS track with %d points", len(gps_track))

    # ── Open video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps         = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total       = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    client      = genai.Client(api_key=api_key) if api_key else genai.Client()

    log.info("Video: %s  (%d frames @ %.1f fps)", Path(video_path).name, total, fps)
    log.info("Sampling every %d frames — %d candidates", every_n,
             len(range(0, total, every_n)))

    # ── Extract candidate frames ──────────────────────────────────────────────
    candidate_indices: list[int]          = []
    candidate_frames:  dict[int, np.ndarray] = {}

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % every_n == 0:
            candidate_indices.append(frame_idx)
            candidate_frames[frame_idx] = frame.copy()
        frame_idx += 1
    cap.release()

    log.info("Extracted %d candidate frames", len(candidate_indices))

    # ── Per-frame metadata accumulator ────────────────────────────────────────
    meta: dict[int, dict] = {
        idx: {
            "frame_idx":      idx,
            "timestamp_sec":  round(idx / fps, 3),
            "lat":            None,
            "lon":            None,
            "dist_m":         None,
            "nearest_station": None,
            "gps_flagged":    False,
            "cv_passed":      False,
            "sharpness":      None,
            "edge_density":   None,
            "stage2_result":  None,
            "stage2_conf":    None,
            "stage3_result":  None,
            "stage3_conf":    None,
            "stage3_reason":  None,
            "selected":       False,
            "selection_stage": None,
        }
        for idx in candidate_indices
    }

    # ── GPS proximity flag ────────────────────────────────────────────────────
    gps_flagged_indices: list[int] = []

    if stations and gps_track:
        for idx in candidate_indices:
            pos = get_frame_gps(idx, fps, gps_track)
            if pos:
                lat, lon = pos
                flagged, station, dist = is_near_station(lat, lon, stations, gps_radius_m)
                meta[idx]["lat"]             = round(lat, 6)
                meta[idx]["lon"]             = round(lon, 6)
                meta[idx]["dist_m"]          = dist
                meta[idx]["nearest_station"] = station["name"] if station else None
                meta[idx]["gps_flagged"]     = flagged
                if flagged:
                    gps_flagged_indices.append(idx)

        log.info("GPS flagged %d/%d frames (within %.0fm of a station)",
                 len(gps_flagged_indices), len(candidate_indices), gps_radius_m)

    # ── Stage 1: CV filter ────────────────────────────────────────────────────
    stage1_passed: list[int] = []

    log.info("── Stage 1: CV filter ──")
    for idx in candidate_indices:
        passed, metrics = passes_cv_filter(candidate_frames[idx])
        meta[idx]["cv_passed"]    = passed
        meta[idx]["sharpness"]    = metrics["sharpness"]
        meta[idx]["edge_density"] = metrics["edge_density"]
        if passed:
            stage1_passed.append(idx)

    log.info("Stage 1: %d/%d frames passed (%.0f%% filtered out)",
             len(stage1_passed), len(candidate_indices),
             100 * (1 - len(stage1_passed) / max(len(candidate_indices), 1)))

    # ── Stage 2: Cheap model ──────────────────────────────────────────────────
    stage2_positives:    list[int] = []   # confirmed has sign
    stage2_low_conf:     list[int] = []   # uncertain — send to Stage 3

    log.info("── Stage 2: Cheap model (%s) ──", CHEAP_MODEL)
    for i, idx in enumerate(stage1_passed):
        result = _query_model(client, CHEAP_MODEL, candidate_frames[idx], CHEAP_PROMPT)
        conf   = int(result.get("confidence", 0))
        has    = bool(result.get("has_price_sign", False))

        meta[idx]["stage2_result"] = has
        meta[idx]["stage2_conf"]   = conf

        if has and conf >= conf_threshold:
            stage2_positives.append(idx)
            meta[idx]["selected"]       = True
            meta[idx]["selection_stage"] = "stage2"
        elif conf < conf_threshold:
            stage2_low_conf.append(idx)

        log.info("  [%4d/%d] frame %4d — sign=%-5s  conf=%3d  %s",
                 i + 1, len(stage1_passed), idx,
                 str(has), conf,
                 "→ SELECTED" if meta[idx]["selected"] else
                 "→ low-conf" if conf < conf_threshold else "")

    log.info("Stage 2: %d confirmed positives, %d low-confidence → Stage 3",
             len(stage2_positives), len(stage2_low_conf))

    # ── Build Stage 3 queue ───────────────────────────────────────────────────
    temporal_neighbours = _expand_temporal_neighbours(
        stage2_positives, candidate_indices, fps, temporal_window
    )

    stage3_queue: set[int] = set()
    stage3_queue.update(stage2_low_conf)
    stage3_queue.update(gps_flagged_indices)
    stage3_queue.update(temporal_neighbours)
    # Don't re-run Stage 3 on already-confirmed Stage 2 positives
    stage3_queue -= set(stage2_positives)
    stage3_queue_sorted = sorted(stage3_queue)

    log.info("── Stage 3: Full model (%s) — %d frames ──", FULL_MODEL,
             len(stage3_queue_sorted))
    log.info("  Sources: %d low-conf + %d GPS-flagged + %d temporal neighbours",
             len(stage2_low_conf), len(gps_flagged_indices),
             len(temporal_neighbours - set(stage2_low_conf) - set(gps_flagged_indices)))

    for i, idx in enumerate(stage3_queue_sorted):
        result = _query_model(client, FULL_MODEL, candidate_frames[idx], FULL_PROMPT)
        conf   = int(result.get("confidence", 0))
        has    = bool(result.get("has_price_sign", False))
        reason = result.get("reason", "")

        meta[idx]["stage3_result"] = has
        meta[idx]["stage3_conf"]   = conf
        meta[idx]["stage3_reason"] = reason

        if has and conf >= conf_threshold:
            meta[idx]["selected"]        = True
            meta[idx]["selection_stage"] = "stage3"

        log.info("  [%4d/%d] frame %4d — sign=%-5s  conf=%3d  %s",
                 i + 1, len(stage3_queue_sorted), idx,
                 str(has), conf,
                 "→ SELECTED" if meta[idx]["selected"] else "")

    # ── Save selected frames ──────────────────────────────────────────────────
    selected = [idx for idx in candidate_indices if meta[idx]["selected"]]
    log.info("Saving %d selected frames to %s", len(selected), output_dir)

    for idx in selected:
        out_path = output_dir / f"frame_{idx:06d}.jpg"
        cv2.imwrite(str(out_path), candidate_frames[idx])

    # ── Write metadata CSV ────────────────────────────────────────────────────
    meta_path = output_dir / "selection_metadata.csv"
    fieldnames = [
        "frame_idx", "timestamp_sec", "lat", "lon", "dist_m", "nearest_station",
        "gps_flagged", "cv_passed", "sharpness", "edge_density",
        "stage2_result", "stage2_conf",
        "stage3_result", "stage3_conf", "stage3_reason",
        "selected", "selection_stage",
    ]
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx in candidate_indices:
            writer.writerow(meta[idx])

    log.info("Metadata CSV: %s", meta_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    total_candidates = len(candidate_indices)
    total_selected   = len(selected)
    api_calls_stage2 = len(stage1_passed)
    api_calls_stage3 = len(stage3_queue_sorted)
    naive_calls      = total_candidates   # without cascade: every candidate to full model

    log.info("─" * 50)
    log.info("SUMMARY")
    log.info("  Candidates (every %d frames): %d", every_n, total_candidates)
    log.info("  Stage 1 survivors:            %d  (%.0f%% filtered)",
             len(stage1_passed), 100*(1 - len(stage1_passed)/max(total_candidates,1)))
    log.info("  Stage 2 API calls (cheap):    %d", api_calls_stage2)
    log.info("  Stage 3 API calls (full):     %d", api_calls_stage3)
    log.info("  Naive full-model calls:        %d", naive_calls)
    log.info("  Full-model call reduction:    %.0f%%",
             100 * (1 - api_calls_stage3 / max(naive_calls, 1)))
    log.info("  Selected frames:              %d", total_selected)
    log.info("─" * 50)

    return meta_path

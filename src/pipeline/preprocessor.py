"""
preprocessor.py
───────────────
Handles image loading, resizing, EXIF GPS extraction, and
(optionally) video → frame extraction.
"""

from __future__ import annotations
import os
import io
import re
import struct
import base64
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from PIL import Image, ExifTags

from .config import (
    SUPPORTED_EXTENSIONS,
    VIDEO_EXTENSIONS,
    MAX_IMAGE_LONG_SIDE,
    JPEG_QUALITY,
    VIDEO_FPS_EXTRACT,
)

log = logging.getLogger(__name__)


# ── Data containers ─────────────────────────────────────────────────────────

@dataclass
class GeoPoint:
    latitude:  Optional[float] = None
    longitude: Optional[float] = None
    captured_at: Optional[str] = None   # ISO-8601 or raw EXIF string


@dataclass
class ProcessedImage:
    image_number: int
    source_path:  Path
    pil_image:    Image.Image
    geo:          GeoPoint = field(default_factory=GeoPoint)

    def to_base64_jpeg(self) -> str:
        """Return the image as a base64-encoded JPEG string."""
        buf = io.BytesIO()
        rgb = self.pil_image.convert("RGB")
        rgb.save(buf, format="JPEG", quality=JPEG_QUALITY)
        return base64.b64encode(buf.getvalue()).decode()


# ── EXIF GPS helpers ─────────────────────────────────────────────────────────

def _rational_to_float(rational) -> float:
    """Convert an EXIF rational (numerator, denominator) to float."""
    if isinstance(rational, tuple):
        return rational[0] / rational[1] if rational[1] else 0.0
    return float(rational)


def _exif_gps_to_decimal(coord, ref: str) -> Optional[float]:
    """Convert DMS EXIF GPS tuple to signed decimal degrees."""
    try:
        degrees = _rational_to_float(coord[0])
        minutes = _rational_to_float(coord[1])
        seconds = _rational_to_float(coord[2])
        decimal = degrees + minutes / 60 + seconds / 3600
        if ref in ("S", "W"):
            decimal = -decimal
        return round(decimal, 7)
    except Exception:
        return None


def extract_exif_geo(img: Image.Image) -> GeoPoint:
    """Extract GPS coordinates and datetime from PIL image EXIF data."""
    geo = GeoPoint()
    try:
        exif_raw = img._getexif()
        if not exif_raw:
            return geo

        # Build tag-name → value map
        exif = {ExifTags.TAGS.get(k, k): v for k, v in exif_raw.items()}

        dt = exif.get("DateTimeOriginal") or exif.get("DateTime")
        if dt:
            geo.captured_at = str(dt)

        gps_info = exif.get("GPSInfo")
        if not gps_info:
            return geo

        gps = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps_info.items()}
        lat = _exif_gps_to_decimal(gps.get("GPSLatitude", ()), gps.get("GPSLatitudeRef", "N"))
        lon = _exif_gps_to_decimal(gps.get("GPSLongitude", ()), gps.get("GPSLongitudeRef", "E"))
        geo.latitude  = lat
        geo.longitude = lon
    except Exception as exc:
        log.debug("EXIF parse failed: %s", exc)
    return geo


# ── Resize helper ─────────────────────────────────────────────────────────────

def _resize_if_needed(img: Image.Image) -> Image.Image:
    w, h = img.size
    if max(w, h) <= MAX_IMAGE_LONG_SIDE:
        return img
    scale = MAX_IMAGE_LONG_SIDE / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    log.debug("Resizing %dx%d → %dx%d", w, h, new_w, new_h)
    return img.resize((new_w, new_h), Image.LANCZOS)


# ── Public API ───────────────────────────────────────────────────────────────

def load_images_from_folder(folder: str | Path,
                             geo_override: Optional[dict[str, GeoPoint]] = None
                             ) -> list[ProcessedImage]:
    """
    Scan *folder* for supported image files (sorted alphabetically),
    load and resize each, extract EXIF GPS, and return a numbered list.

    geo_override: mapping  filename (stem) → GeoPoint  from a sidecar CSV.
                  If provided, overrides EXIF data for that filename.
    """
    folder = Path(folder)
    def _natural_key(path: Path) -> list:

        """Sort '2.png' < '10.png' < '22.png' instead of lexicographic order."""
        return [int(c) if c.isdigit() else c.lower()
            for c in re.split(r'(\d+)', path.name)]

    files = sorted(
        (f for f in folder.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS),
        key=_natural_key
    )
    # files  = sorted(
    #     f for f in folder.iterdir()
    #     if f.suffix.lower() in SUPPORTED_EXTENSIONS
    # )
    if not files:
        log.warning("No supported images found in %s", folder)

    results: list[ProcessedImage] = []
    for idx, fpath in enumerate(files, start=1):
        try:
            img = Image.open(fpath)
            img.load()                          # force decode
            geo = extract_exif_geo(img)
            img = _resize_if_needed(img)

            # sidecar override (higher priority than EXIF)
            if geo_override and fpath.stem in geo_override:
                geo = geo_override[fpath.stem]

            results.append(ProcessedImage(
                image_number=idx,
                source_path=fpath,
                pil_image=img,
                geo=geo,
            ))
            log.info("[%d] Loaded %s  GPS=(%s, %s)",
                     idx, fpath.name, geo.latitude, geo.longitude)
        except Exception as exc:
            log.error("Failed to load %s: %s", fpath, exc)

    return results


def extract_frames_from_video(video_path: str | Path,
                               output_folder: str | Path,
                               fps: float = VIDEO_FPS_EXTRACT) -> Path:
    """
    Extract frames from a video at *fps* frames-per-second and save them
    as JPEG files in *output_folder*.  Requires opencv-python.

    Returns output_folder as a Path (ready for load_images_from_folder).
    """
    try:
        import cv2  # type: ignore
    except ImportError:
        raise ImportError("Install opencv-python:  pip install opencv-python")

    video_path    = Path(video_path)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    cap      = cv2.VideoCapture(str(video_path))
    vid_fps  = cap.get(cv2.CAP_PROP_FPS) or 30
    interval = max(1, int(vid_fps / fps))   # save every Nth frame

    frame_idx = 0
    saved     = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            out_path = output_folder / f"frame_{saved:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1
        frame_idx += 1

    cap.release()
    log.info("Extracted %d frames from %s → %s", saved, video_path, output_folder)
    return output_folder


# ── Sidecar CSV loader ────────────────────────────────────────────────────────

def load_geo_sidecar(csv_path: str | Path) -> dict[str, GeoPoint]:
    """
    Load a sidecar CSV with columns:
        filename, latitude, longitude, captured_at
    Returns a dict keyed by filename stem (no extension).
    """
    import csv
    mapping: dict[str, GeoPoint] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            stem = Path(row["filename"]).stem
            mapping[stem] = GeoPoint(
                latitude    = float(row["latitude"])  if row.get("latitude")  else None,
                longitude   = float(row["longitude"]) if row.get("longitude") else None,
                captured_at = row.get("captured_at"),
            )
    return mapping

"""
select_frames.py
─────────────────
CLI entry point for the three-stage frame selection cascade.

Usage
-----
# Basic — CV + cheap model only (no GPS)
python src/frame_selection/select_frames.py \
    --input  "data/videos/Gas Videos/IMG_0966.MOV" \
    --output output/frames/IMG_0966/

# With mock GPS data
python src/frame_selection/select_frames.py \
    --input      "data/videos/Gas Videos/IMG_0966.MOV" \
    --output     output/frames/IMG_0966/ \
    --stations   "GPS info/stations.csv" \
    --gps-track  "GPS info/IMG_0966_gps_track.csv"

# Then run the full OCR pipeline on selected frames:
python pipeline_csv.py --input output/frames/IMG_0966/
"""

from __future__ import annotations
import argparse
import logging
from pathlib import Path

from .cascade import run_cascade

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt = "%H:%M:%S",
)


def main():
    p = argparse.ArgumentParser(
        description="Three-stage frame selection cascade: CV → cheap model → full model"
    )
    p.add_argument("--input",      required=True,
                   help="Input video file path")
    p.add_argument("--output",     required=True,
                   help="Output folder for selected frames and metadata CSV")
    p.add_argument("--stations",   default=None,
                   help="Path to stations.csv (GPS info/stations.csv)")
    p.add_argument("--gps-track",  default=None,
                   help="Path to per-video GPS track CSV (timestamp_sec, lat, lon)")
    p.add_argument("--every",      type=int,   default=5,
                   help="Sample one frame every N frames (default: 5)")
    p.add_argument("--radius",     type=float, default=500.0,
                   help="GPS proximity radius in metres (default: 500)")
    p.add_argument("--conf",       type=int,   default=70,
                   help="Stage 2 confidence threshold; below this → Stage 3 (default: 70)")
    p.add_argument("--temporal",   type=float, default=30.0,
                   help="Temporal neighbour window in seconds (default: 30)")
    args = p.parse_args()

    meta_path = run_cascade(
        video_path      = args.input,
        output_dir      = args.output,
        stations_csv    = args.stations,
        gps_track_csv   = getattr(args, "gps_track"),
        every_n         = args.every,
        gps_radius_m    = args.radius,
        conf_threshold  = args.conf,
        temporal_window = args.temporal,
    )

    print(f"\n✅  Done. Metadata: {meta_path}")
    print(f"   Run OCR on selected frames:")
    print(f"   python pipeline_csv.py --input {args.output}")


if __name__ == "__main__":
    main()

"""
run_pipeline.py
───────────────
Main entry point for the Gas Price Extraction Pipeline.

Usage examples
--------------
# Basic: process a folder of images

python run_pipeline.py --input ../../data/bright

# With a GPS sidecar CSV
python run_pipeline.py --input ../data/bright --geo-sidecar gps_log.csv

# From a video (extracts frames first)
python run_pipeline.py --video dashcam.mp4 --frames-dir ./frames

# Set minimum confidence threshold
python run_pipeline.py --input ./images --min-confidence 75

# Custom output file
python run_pipeline.py --input ./images --output ./results/prices.csv
"""

from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

from src.pipeline.preprocessor import (
    load_images_from_folder,
    load_geo_sidecar,
    extract_frames_from_video,
)
from src.pipeline.extractor  import extract_prices
from src.pipeline.geo_merger import merge_and_export, print_summary
from src.pipeline.config     import OUTPUT_DIR, FINAL_CSV, GEO_SIDECAR_CSV

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt = "%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="Gas Price Extraction Pipeline  (images or video → geo CSV)"
    )

    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--input",  metavar="FOLDER",
                        help="Folder containing image files")
    source.add_argument("--video",  metavar="FILE",
                        help="Video file to extract frames from")

    p.add_argument("--frames-dir", metavar="FOLDER", default="frames",
                   help="Where to save video frames (default: ./frames)")
    p.add_argument("--geo-sidecar", metavar="CSV",
                   help="CSV with columns: filename,latitude,longitude,captured_at")
    p.add_argument("--output", metavar="CSV",
                   default=str(Path(OUTPUT_DIR) / FINAL_CSV),
                   help="Output CSV file path")
    p.add_argument("--min-confidence", metavar="INT", type=int, default=0,
                   help="Drop entries below this confidence score (default: 0 = keep all)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Step 1: Resolve image folder ─────────────────────────────────────────
    if args.video:
        log.info("Extracting frames from video: %s", args.video)
        image_folder = extract_frames_from_video(args.video, args.frames_dir)
    else:
        image_folder = Path(args.input)
        if not image_folder.is_dir():
            log.error("Input folder not found: %s", image_folder)
            sys.exit(1)

    # ── Step 2: Load GPS sidecar (optional) ──────────────────────────────────
    geo_override = None
    sidecar_path = args.geo_sidecar or GEO_SIDECAR_CSV
    if sidecar_path:
        log.info("Loading GPS sidecar: %s", sidecar_path)
        geo_override = load_geo_sidecar(sidecar_path)

    # ── Step 3: Preprocess images ─────────────────────────────────────────────
    log.info("Loading images from: %s", image_folder)
    images = load_images_from_folder(image_folder, geo_override)
    if not images:
        log.error("No images to process. Exiting.")
        sys.exit(1)
    log.info("Found %d image(s)", len(images))

    # ── Step 4: Call Gemini OCR ───────────────────────────────────────────────
    log.info("Sending images to Gemini for OCR extraction …")
    price_rows = extract_prices(images)
    if not price_rows:
        log.warning("No price data extracted. Check images or API key.")
        sys.exit(0)
    log.info("Total price entries extracted: %d", len(price_rows))

    # ── Step 5: Merge with GPS and export ─────────────────────────────────────
    output_path = merge_and_export(
        price_rows       = price_rows,
        images           = images,
        output_csv       = args.output,
        min_confidence   = args.min_confidence,
    )

    # ── Step 6: Summary ───────────────────────────────────────────────────────
    print_summary(output_path)
    print(f"✅  Done!  Results saved to: {output_path}")


if __name__ == "__main__":
    main()

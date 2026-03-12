"""
geo_merger.py
─────────────
Merges extracted price rows with geographic coordinates and exports
a final CSV ready for map display / remote dataset upload.
"""

from __future__ import annotations
import csv
import logging
import re
from datetime import datetime
from pathlib import Path

from .preprocessor import ProcessedImage

log = logging.getLogger(__name__)


FINAL_HEADER = [
    "Image_Number",
    "Fuel_Type",
    "Price",
    "Gas_Station_Brand",
    "Payment_Type",
    "Confidence",
    "Latitude",
    "Longitude",
    "Captured_At",
    "Source_File",
]


def _build_geo_index(images: list[ProcessedImage]) -> tuple[
        dict[int, ProcessedImage],
        dict[str, ProcessedImage]]:
    """
    Build two lookup dicts:
      by_number   : image_number (int)  → ProcessedImage
      by_filename : filename / stem     → ProcessedImage
    """
    by_number:   dict[int, ProcessedImage] = {}
    by_filename: dict[str, ProcessedImage] = {}
    for img in images:
        by_number[img.image_number] = img
        by_filename[img.source_path.name] = img   # "21.png"
        by_filename[img.source_path.stem] = img   # "21"
    return by_number, by_filename


def _resolve_image(raw: str,
                   by_number:   dict[int, ProcessedImage],
                   by_filename: dict[str, ProcessedImage],
                   ) -> tuple[int, ProcessedImage | None, str]:
    """
    Robustly resolve whatever the model put in the Image_Number field.

    The model may return any of:
      "3"           → integer index  (ideal)
      "21.png"      → filename
      "21"          → filename stem
      "Image 3"     → label echo
      "3 | 21.png"  → combined label

    Returns
    -------
    (resolved_int, ProcessedImage | None, fallback_filename)

    fallback_filename is the best filename we can derive from `raw` even
    if proc lookup fails — guarantees Source_File is never silently "NA"
    when the original filename is recoverable from the model's output.
    """
    raw = str(raw).strip()

    # ── 1. Direct integer index ───────────────────────────────────────────────
    try:
        n    = int(raw)
        proc = by_number.get(n)
        name = proc.source_path.name if proc else "NA"
        return n, proc, name
    except ValueError:
        pass

    # ── 2. Exact filename / stem lookup  ("21.png" or "21") ──────────────────
    if raw in by_filename:
        proc = by_filename[raw]
        return proc.image_number, proc, proc.source_path.name

    # ── 3. Filename embedded in a longer string  ("3 | 21.png") ──────────────
    #    Look for anything that matches a known filename first.
    for candidate in re.findall(r'[\w.-]+\.(?:png|jpg|jpeg|bmp|webp|tiff)', raw, re.IGNORECASE):
        if candidate in by_filename:
            proc = by_filename[candidate]
            return proc.image_number, proc, proc.source_path.name
        # filename found in raw but not in index — still preserve it
        fallback_name = candidate

    # ── 4. Extract first integer from string  ("Image 3") ────────────────────
    m = re.search(r'\d+', raw)
    if m:
        n    = int(m.group())
        proc = by_number.get(n)
        name = proc.source_path.name if proc else locals().get("fallback_name", "NA")
        return n, proc, name

    # ── 5. Give up ────────────────────────────────────────────────────────────
    log.warning("Cannot resolve Image_Number from model value: %r", raw)
    return -1, None, "NA"


def _stamped_path(output_csv: Path) -> Path:
    """
    Append a datetime stamp to the file stem.
    e.g.  output/gas_prices_geo.csv  →  output/gas_prices_geo_20240315_143022.csv
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_csv.with_name(f"{output_csv.stem}_{stamp}{output_csv.suffix}")


def merge_and_export(price_rows: list[dict],
                     images: list[ProcessedImage],
                     output_csv: str | Path,
                     min_confidence: int = 0) -> Path:
    """
    Join price_rows (from extractor) with GPS data from images list.
    Writes the final CSV to a timestamped file so every run is unique.

    e.g.  gas_prices_geo_20240315_143022.csv

    Parameters
    ----------
    price_rows      : rows returned by extractor.extract_prices()
    images          : list of ProcessedImage (carries GPS + source path)
    output_csv      : base output path (stem used as prefix for timestamp)
    min_confidence  : drop rows below this confidence score (0 = keep all)
    """
    output_csv = _stamped_path(Path(output_csv))
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    by_number, by_filename = _build_geo_index(images)
    written  = 0
    skipped  = 0

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FINAL_HEADER)
        writer.writeheader()

        for row in price_rows:
            # ── Confidence filter ─────────────────────────────────────────────
            try:
                conf = int(row.get("Confidence", 0))
            except ValueError:
                conf = 0

            if conf < min_confidence:
                skipped += 1
                continue

            # ── Resolve image → proc + guaranteed filename ────────────────────
            img_num, proc, fallback_name = _resolve_image(
                row.get("Image_Number", "-1"), by_number, by_filename
            )

            # Source_File priority:
            #   1. proc.source_path.name  (matched ProcessedImage — most reliable)
            #   2. fallback_name          (filename extracted from model's raw value)
            #   3. "NA"
            source_file = (
                proc.source_path.name if proc
                else fallback_name
            )

            merged = {
                "Image_Number":      img_num,
                "Fuel_Type":         row.get("Fuel_Type",         "NA"),
                "Price":             row.get("Price",             "NA"),
                "Gas_Station_Brand": row.get("Gas_Station_Brand", "NA"),
                "Payment_Type":      row.get("Payment_Type",      "NA"),
                "Confidence":        conf,
                "Latitude":          proc.geo.latitude    if proc else "NA",
                "Longitude":         proc.geo.longitude   if proc else "NA",
                "Captured_At":       proc.geo.captured_at if proc else "NA",
                "Source_File":       source_file,
            }
            writer.writerow(merged)
            written += 1

    log.info("Exported %d rows to %s  (%d skipped by confidence filter)",
             written, output_csv, skipped)
    return output_csv


def print_summary(output_csv: str | Path) -> None:
    """Print a quick summary of the exported CSV to stdout."""
    output_csv = Path(output_csv)
    with open(output_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"\n{'='*60}")
    print(f"  Gas Price Extraction Summary")
    print(f"  Output: {output_csv}")
    print(f"  Total entries: {len(rows)}")
    print(f"{'='*60}")

    brands = {r["Gas_Station_Brand"] for r in rows if r["Gas_Station_Brand"] != "NA"}
    if brands:
        print(f"  Stations detected: {', '.join(sorted(brands))}")

    geotagged = sum(1 for r in rows if r["Latitude"] not in ("NA", ""))
    print(f"  Geotagged entries: {geotagged}/{len(rows)}")

    missing_file = sum(1 for r in rows if r["Source_File"] == "NA")
    if missing_file:
        log.warning("  %d row(s) have no Source_File — check model output", missing_file)
    print(f"{'='*60}\n")
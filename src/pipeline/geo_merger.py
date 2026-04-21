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
from collections import defaultdict
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


_PAYMENT_KEYWORDS = {"cash", "credit", "debit", "card", "payment"}

_GRADE_ORDER = {"Regular": 0, "Mid-Grade": 1, "Premium": 2,
                "Diesel": 3, "E85": 4, "Unknown": 5}


def _normalise_grade_simple(raw: str) -> str | None:
    """
    Lightweight grade normaliser. Returns None if row should be dropped.
    Strips payment suffixes (e.g. 'Regular CASH' → 'Regular') before matching.
    Only drops a row as a pure payment label if no grade keyword is present.
    """
    if not raw or raw.upper() == "NA":
        return "Unknown"
    r = raw.lower().strip()
    # Strip trailing payment words so "Regular CASH" → "regular"
    for kw in _PAYMENT_KEYWORDS:
        r = r.replace(kw, "").strip()
    # If nothing meaningful is left, it was a pure payment label — drop it
    if not r:
        return None
    if "diesel" in r or r in ("dsl", "dsl.", "d", "ulsd", "uls", "ulsd89"):
        return "Diesel"
    if "premium" in r or "supreme" in r or "v-power" in r or r in ("prem", "prm", "sup", "91", "93"):
        return "Premium"
    if "plus" in r or "mid" in r or r in ("extra", "89"):
        return "Mid-Grade"
    if "regular" in r or "unleaded" in r or "gasoline" in r or r in ("reg", "reg.", "unl", "u", "87", "e10"):
        return "Regular"
    if "e85" in r or r == "85":
        return "E85"
    return "Unknown"


def export_clean_summary(raw_csv: str | Path,
                         strategy_name: str = "modal",
                         **strategy_kwargs) -> Path:
    """
    Read the raw frame-level CSV and write a *_clean.csv alongside it.

    The aggregation strategy is pluggable — see src/pipeline/aggregator.py.
    Pass strategy_name to switch strategies; extra kwargs go to the strategy.

    Example:
        export_clean_summary(path, strategy_name="consensus", threshold=0.6)
    """
    from .aggregator import get_strategy

    raw_csv   = Path(raw_csv)
    clean_csv = raw_csv.with_name(raw_csv.stem + "_clean.csv")

    with open(raw_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # ── Filter and normalise ──────────────────────────────────────────────────
    valid = []
    for idx, r in enumerate(rows):
        price_str = r.get("Price", "NA").strip()
        if price_str in ("NA", "", "nan"):
            continue
        try:
            price_val = float(price_str)
        except ValueError:
            continue

        grade = _normalise_grade_simple(r.get("Fuel_Type", "NA"))
        if grade is None:
            continue

        valid.append({
            "grade":      grade,
            "price":      price_val,
            "brand":      r.get("Gas_Station_Brand", "NA"),
            "payment":    r.get("Payment_Type", "NA") or "NA",
            "confidence": int(r.get("Confidence", 0) or 0),
            "frame_idx":  int(r.get("Image_Number", idx) or idx),
        })

    if not valid:
        log.warning("No valid price rows to aggregate for clean summary.")
        return raw_csv

    # ── Delegate to strategy ──────────────────────────────────────────────────
    strategy   = get_strategy(strategy_name, **strategy_kwargs)
    clean_rows = strategy.aggregate(valid)
    log.info("Strategy '%s' → %d grade(s)", strategy_name, len(clean_rows))

    # ── Deduplicate: drop Unknown rows whose price is already covered ─────────
    # A (price, payment) pair already represented by a named grade should not
    # also appear as Unknown — that is a duplicate caused by frames where
    # Gemini read the price but not the grade label.
    known_keys = {
        (r["Price"], r["Payment_Type"])
        for r in clean_rows
        if r["Fuel_Type"] != "Unknown"
    }
    before = len(clean_rows)
    clean_rows = [
        r for r in clean_rows
        if r["Fuel_Type"] != "Unknown"
        or (r["Price"], r["Payment_Type"]) not in known_keys
    ]
    dropped = before - len(clean_rows)
    if dropped:
        log.info("  Dropped %d Unknown duplicate(s) already covered by named grade(s).", dropped)

    # ── Cluster nearby Unknown prices ─────────────────────────────────────────
    # Prices within CLUSTER_THRESHOLD of each other on the same payment type
    # are merged into one row (modal price wins; frames are summed).
    CLUSTER_THRESHOLD = 0.05
    unknown_rows = [r for r in clean_rows if r["Fuel_Type"] == "Unknown"]
    named_rows   = [r for r in clean_rows if r["Fuel_Type"] != "Unknown"]

    if len(unknown_rows) > 1:
        from collections import defaultdict as _dd
        by_payment: dict = _dd(list)
        for r in unknown_rows:
            by_payment[r["Payment_Type"]].append(r)

        clustered_unknown = []
        for payment, group in by_payment.items():
            group.sort(key=lambda r: r["Price"])
            clusters, current = [], [group[0]]
            for r in group[1:]:
                if r["Price"] - current[0]["Price"] <= CLUSTER_THRESHOLD:
                    current.append(r)
                else:
                    clusters.append(current)
                    current = [r]
            clusters.append(current)

            for cluster in clusters:
                # Modal price = the one with most frames
                rep        = max(cluster, key=lambda r: r["Frames_Detected"])
                total_frm  = sum(r["Frames_Detected"] for r in cluster)
                modal_frm  = rep["Frames_Detected"]
                best_conf  = max(r["Max_Confidence"] for r in cluster)
                clustered_unknown.append({
                    "Gas_Station_Brand": rep["Gas_Station_Brand"],
                    "Fuel_Type":         "Unknown",
                    "Price":             rep["Price"],
                    "Payment_Type":      payment,
                    "Max_Confidence":    best_conf,
                    "Frames_Detected":   total_frm,
                    "Consistency_Pct":   round(modal_frm / total_frm * 100, 1),
                })

        before_cluster = len(clean_rows)
        clean_rows = named_rows + clustered_unknown
        merged = before_cluster - len(clean_rows)
        if merged:
            log.info("  Clustered %d nearby Unknown price(s) → %d row(s) (threshold ±$%.2f).",
                     before_cluster - len(named_rows), len(clustered_unknown), CLUSTER_THRESHOLD)

    # ── Auto-assign Regular / Diesel when all prices remain Unknown ──────────
    # Conditions (all must hold to avoid false assignment):
    #   • Every row in clean_rows is Unknown (no named grade was ever extracted)
    #   • Exactly 1 or 2 Unknown prices (3+ grades are too ambiguous)
    #   • All rows have confidence ≥ AUTO_ASSIGN_MIN_CONF
    #   • For 2 prices: gap between low and high > AUTO_ASSIGN_GAP (typical
    #     Regular–Diesel spread); avoids mislabeling Regular/Plus/Premium triples
    AUTO_ASSIGN_MIN_CONF = 80
    AUTO_ASSIGN_GAP      = 0.80   # dollars

    all_unknown = all(r["Fuel_Type"] == "Unknown" for r in clean_rows)
    if all_unknown and 1 <= len(clean_rows) <= 2:
        conf_ok = all(r["Max_Confidence"] >= AUTO_ASSIGN_MIN_CONF for r in clean_rows)
        prices  = sorted(clean_rows, key=lambda r: r["Price"])

        if len(prices) == 1 and conf_ok:
            prices[0]["Fuel_Type"] = "Regular"
            log.info("  Auto-assigned single Unknown price $%.3f → Regular (conf=%d).",
                     prices[0]["Price"], prices[0]["Max_Confidence"])

        elif len(prices) == 2 and conf_ok:
            gap = prices[1]["Price"] - prices[0]["Price"]
            if gap > AUTO_ASSIGN_GAP:
                prices[0]["Fuel_Type"] = "Regular"
                prices[1]["Fuel_Type"] = "Diesel"
                log.info(
                    "  Auto-assigned Unknown prices: $%.3f → Regular, $%.3f → Diesel "
                    "(gap=%.3f, conf=%d/%d).",
                    prices[0]["Price"], prices[1]["Price"], gap,
                    prices[0]["Max_Confidence"], prices[1]["Max_Confidence"],
                )
            else:
                log.info(
                    "  Skipped auto-assign: 2 Unknown prices but gap=%.3f ≤ %.2f "
                    "(too close to distinguish Regular from Diesel).",
                    gap, AUTO_ASSIGN_GAP,
                )

    # ── Auto-assign Cash / Credit for same-grade unlabelled price pairs ──────
    # If a named grade has exactly 2 rows both with payment=NA and the price
    # difference is within a typical cash/credit spread ($0.05–$0.25), the
    # lower price is likely Cash and the higher is Credit.
    CASH_CREDIT_MIN_GAP = 0.05
    CASH_CREDIT_MAX_GAP = 0.25

    by_grade: dict = defaultdict(list)
    for r in clean_rows:
        if r["Fuel_Type"] != "Unknown":
            by_grade[r["Fuel_Type"]].append(r)

    for grade, group in by_grade.items():
        na_rows = [r for r in group if r["Payment_Type"] == "NA"]
        if len(na_rows) == 2:
            lo, hi = sorted(na_rows, key=lambda r: r["Price"])
            gap = hi["Price"] - lo["Price"]
            if CASH_CREDIT_MIN_GAP <= gap <= CASH_CREDIT_MAX_GAP:
                lo["Payment_Type"] = "Cash"
                hi["Payment_Type"] = "Credit"
                log.info(
                    "  Auto-assigned Cash/Credit for %s: $%.3f → Cash, $%.3f → Credit (gap=%.3f).",
                    grade, lo["Price"], hi["Price"], gap,
                )

    # ── Write ─────────────────────────────────────────────────────────────────
    fieldnames = ["Gas_Station_Brand", "Fuel_Type", "Price", "Payment_Type",
                  "Max_Confidence", "Frames_Detected", "Consistency_Pct"]
    with open(clean_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(clean_rows)

    log.info("Clean summary → %s", clean_csv)
    return clean_csv


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
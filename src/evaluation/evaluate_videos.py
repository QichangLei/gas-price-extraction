"""
evaluate_videos.py
──────────────────
Compare pipeline clean-CSV outputs against ground_truth.csv for a set of videos.
Classifies each discrepancy as:
  - digit_error    : fuel type matched but price is wrong (outside ±tolerance)
  - fuel_type_error: extracted fuel type doesn't match any GT fuel type for that video
  - missing        : GT grade was not extracted at all
  - false_positive : extracted grade has no corresponding GT entry
  - correct        : fuel type matched AND price within tolerance

Usage
-----
    python src/evaluation/evaluate_videos.py
    python src/evaluation/evaluate_videos.py --tolerance 0.01 --output output/evaluation/video_eval.txt
"""

import argparse
import csv
import re
from pathlib import Path
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parents[2]
GT_CSV       = PROJECT_ROOT / "data/videos/labels/ground_truth.csv"
PIPELINE_DIR = PROJECT_ROOT / "output/pipeline"
OUTPUT_DIR   = PROJECT_ROOT / "output/evaluation"

# ── Fuel-type normalisation ────────────────────────────────────────────────────
_GRADE_MAP = {
    "regular": "Regular", "unleaded": "Regular", "gasoline": "Regular",
    "midgrade": "Mid-Grade", "mid-grade": "Mid-Grade", "mid grade": "Mid-Grade", "plus": "Mid-Grade",
    "premium": "Premium", "super": "Premium", "v-power": "Premium", "vpower": "Premium",
    "supreme": "Supreme", "extra": "Extra",   # Exxon-specific brand names
    "diesel": "Diesel", "diesel efficient": "Diesel",
    "e15": "E15", "e85": "E85",
    "95": "95", "mp5": "MP5",  # non-US octane labels (night_moving)
}

def norm_grade(raw: str) -> str:
    r = re.sub(r"\b(cash|credit|price)\b", "", raw.lower()).strip()
    return _GRADE_MAP.get(r, raw.strip())


# ── Loaders ────────────────────────────────────────────────────────────────────

def load_ground_truth(videos: list[str]) -> dict[str, list[dict]]:
    """Returns {video_stem: [{"grade": ..., "price": float, "payment": ...}]}"""
    gt = defaultdict(list)
    with open(GT_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            stem = Path(row["video_file"]).stem.lower()
            if stem in {Path(v).stem.lower() for v in videos}:
                try:
                    gt[stem].append({
                        "grade":   row["grade"],
                        "price":   float(row["price"]),
                        "payment": row.get("payment", "NA") or "NA",
                        "brand":   row.get("brand", "NA"),
                    })
                except ValueError:
                    pass  # skip TODO rows
    return gt


def load_clean_csv(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                rows.append({
                    "grade":   row.get("Fuel_Type", "NA"),
                    "price":   float(row["Price"]),
                    "payment": row.get("Payment_Type", "NA") or "NA",
                    "brand":   row.get("Gas_Station_Brand", "NA"),
                })
            except (ValueError, KeyError):
                pass
    return rows


def find_clean_csv(video_stem: str) -> Path | None:
    """Find the most recent clean CSV for a given video stem."""
    pattern = f"gas_prices_{video_stem}_*_clean.csv"
    matches = sorted(PIPELINE_DIR.glob(pattern))
    return matches[-1] if matches else None


# ── Matching logic ─────────────────────────────────────────────────────────────

def match_rows(gt_rows: list[dict], pred_rows: list[dict], tolerance: float
               ) -> list[dict]:
    """
    Match predicted rows to GT rows. For each GT row attempt to find a
    predicted row with the same normalised grade (and payment if specified).
    Returns a list of result dicts.
    """
    results = []
    used_pred = set()

    for gt in gt_rows:
        gt_grade_norm = norm_grade(gt["grade"])
        gt_pay        = gt["payment"].lower() if gt["payment"] not in ("NA", "") else None
        grade_unknown = gt["grade"].strip().upper() == "NA"

        # For GT rows with grade=NA: match on price only (grade label unreadable)
        # For normal GT rows: match on normalised grade (+ payment if specified)
        DIGIT_ERROR_WINDOW = 0.15   # wider window for grade=NA digit-error detection

        match_idx      = None
        digit_err_idx  = None   # fallback: price-only near-miss → digit_error
        for i, pred in enumerate(pred_rows):
            if i in used_pred:
                continue
            if grade_unknown:
                diff = abs(pred["price"] - gt["price"])
                if diff <= tolerance:
                    match_idx = i
                    break
                elif diff <= DIGIT_ERROR_WINDOW and digit_err_idx is None:
                    digit_err_idx = i
            else:
                pred_grade_norm = norm_grade(pred["grade"])
                if pred_grade_norm == gt_grade_norm:
                    pred_pay = pred["payment"].lower() if pred["payment"] not in ("NA", "") else None
                    if gt_pay is None or pred_pay is None or gt_pay == pred_pay:
                        # Only consume this pred if price is in a plausible range.
                        # A $1.50+ gap means the grade label matched by coincidence;
                        # leave the pred row available for a better price-proximity match.
                        if abs(pred["price"] - gt["price"]) <= 1.50:
                            match_idx = i
                            break

        # For grade=NA: fall back to digit_err_idx if no exact match found
        if match_idx is None and grade_unknown and digit_err_idx is not None:
            match_idx = digit_err_idx

        if match_idx is None:
            results.append({
                "error_type": "missing",
                "gt_grade":   gt["grade"],
                "gt_price":   gt["price"],
                "gt_payment": gt["payment"],
                "pred_grade": "",
                "pred_price": "",
                "pred_payment": "",
                "note": "price not found in pipeline output" if grade_unknown
                        else "GT grade not found in pipeline output",
            })
        else:
            used_pred.add(match_idx)
            pred = pred_rows[match_idx]
            price_ok = abs(pred["price"] - gt["price"]) <= tolerance
            results.append({
                "error_type":   "correct" if price_ok else "digit_error",
                "gt_grade":     gt["grade"],
                "gt_price":     gt["price"],
                "gt_payment":   gt["payment"],
                "pred_grade":   pred["grade"],
                "pred_price":   pred["price"],
                "pred_payment": pred["payment"],
                "note": "(price-only match)" if grade_unknown and price_ok
                        else "" if price_ok
                        else f"expected {gt['price']}, got {pred['price']}",
            })

    # Remaining preds with no GT match = false positives (skip "Unknown" grade)
    for i, pred in enumerate(pred_rows):
        if i not in used_pred and norm_grade(pred["grade"]) != "Unknown":
            results.append({
                "error_type":   "false_positive",
                "gt_grade":     "",
                "gt_price":     "",
                "gt_payment":   "",
                "pred_grade":   pred["grade"],
                "pred_price":   pred["price"],
                "pred_payment": pred["payment"],
                "note": "no matching GT entry",
            })

    return results


# ── Main ───────────────────────────────────────────────────────────────────────

VIDEOS = [
    "gas_station_1.mp4",
    "IMG_0966.MOV",
    "IMG_0970.MOV",
    "IMG_0977.mov",
    "IMG_0979.mov",
    "IMG_1080.mov",
    "IMG_1084.mov",
    "IMG_1087.mov",
    "IMG_1088.mov",
    "IMG_1090.mov",
    "IMG_1091.mov",
    "IMG_1092.mov",
    "IMG_1093.mov",
    "IMG_1094.mov",
    "IMG_1097.mov",
    "IMG_1098.mov",
]


def run_evaluation(tolerance: float, output_path: Path) -> None:
    gt_all = load_ground_truth(VIDEOS)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    lines   = []
    summary = defaultdict(int)

    lines.append("=" * 70)
    lines.append("  Video Pipeline Evaluation Report")
    lines.append(f"  Price tolerance: ±${tolerance:.3f}")
    lines.append("=" * 70)

    for video in VIDEOS:
        stem     = Path(video).stem.lower()
        gt_rows  = gt_all.get(stem, [])
        clean_csv = find_clean_csv(Path(video).stem)

        lines.append(f"\n{'─'*70}")
        lines.append(f"  VIDEO: {video}")
        lines.append(f"{'─'*70}")

        if not gt_rows:
            lines.append("  [SKIP] No labeled ground truth rows (TODO entries).")
            continue

        if clean_csv is None:
            lines.append(f"  [SKIP] No clean pipeline output found in {PIPELINE_DIR}")
            continue

        lines.append(f"  Pipeline output : {clean_csv.name}")
        lines.append(f"  GT rows         : {len(gt_rows)}")

        pred_rows = load_clean_csv(clean_csv)
        results   = match_rows(gt_rows, pred_rows, tolerance)

        for r in results:
            summary[r["error_type"]] += 1
            if r["error_type"] == "correct":
                lines.append(
                    f"  ✅  CORRECT       | GT: {r['gt_grade']:<20} ${r['gt_price']:.3f}  {r['gt_payment']}"
                )
            elif r["error_type"] == "digit_error":
                lines.append(
                    f"  ❌  DIGIT ERROR   | GT: {r['gt_grade']:<20} ${r['gt_price']:.3f}  "
                    f"→ got ${r['pred_price']:.3f}  ({r['note']})"
                )
            elif r["error_type"] == "fuel_type_error":
                lines.append(
                    f"  ❌  FUEL TYPE ERR | GT: {r['gt_grade']:<20} → got '{r['pred_grade']}'"
                )
            elif r["error_type"] == "missing":
                lines.append(
                    f"  ⚠️   MISSING       | GT: {r['gt_grade']:<20} ${r['gt_price']:.3f}  "
                    f"{r['gt_payment']} — not extracted"
                )
            elif r["error_type"] == "false_positive":
                lines.append(
                    f"  ➕  FALSE POS     | Pred: {r['pred_grade']:<18} ${r['pred_price']:.3f} "
                    f"— no GT match"
                )

    # ── Summary ────────────────────────────────────────────────────────────────
    total = sum(summary.values())
    lines.append(f"\n{'='*70}")
    lines.append("  OVERALL SUMMARY")
    lines.append(f"{'='*70}")
    lines.append(f"  Total comparisons : {total}")
    lines.append(f"  ✅  Correct        : {summary['correct']}  "
                 f"({100*summary['correct']/total:.1f}%)" if total else "  N/A")
    lines.append(f"  ❌  Digit errors   : {summary['digit_error']}")
    lines.append(f"  ❌  Fuel type errs : {summary['fuel_type_error']}")
    lines.append(f"  ⚠️   Missing        : {summary['missing']}")
    lines.append(f"  ➕  False positives: {summary['false_positive']}")
    lines.append("")

    report = "\n".join(lines)
    print(report)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved → {output_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tolerance", type=float, default=0.05,
                   help="Price match tolerance in dollars (default: 0.05)")
    p.add_argument("--output", default=str(OUTPUT_DIR / "video_eval_report.txt"))
    args = p.parse_args()
    run_evaluation(args.tolerance, Path(args.output))


if __name__ == "__main__":
    main()

"""
Gas Price Extraction Evaluator
================================
Compares extracted gas prices (from Gemini pipeline) against ground truth labels.

Ground truth format (0.csv .. 19.csv):
    Price,Grade,Cash/Credit
    3.359,Regular,Both

Extracted format (pipeline output CSV):
    Image_Number,Fuel_Type,Price,Gas_Station_Brand,Payment_Type,Confidence,...,Source_File

HOW img_idx IS DETERMINED (two naming conventions can coexist):
  - If Source_File is a valid filename like "10.png"  -> use stem number (10)
  - If Source_File is blank / "NA"                    -> use Image_Number directly
    (in this case Image_Number IS the file index, e.g. 0 -> 0.csv)
  NOTE: when Source_File IS present, Image_Number is a batch counter that is
  offset by +1 from the file index (Image_Number=1 -> 0.png). Never use
  Image_Number as fallback for rows that already have a valid Source_File.

Usage:
    python evaluate_gas_prices.py \
        --extracted  path/to/extracted.csv \
        --gt_dir     path/to/labels/ \
        --tolerance  0.01 \
        --output     evaluation_report.csv
"""

import argparse
import os
import re
import pandas as pd
import numpy as np
from pathlib import Path


# -------------------------------------------------
# Grade normalisation map
# -------------------------------------------------

GRADE_CANONICAL = {
    "regular":   "Regular",
    "midgrade":  "Mid-Grade",
    "mid-grade": "Mid-Grade",
    "mid grade": "Mid-Grade",
    "plus":      "Mid-Grade",
    "super":     "Mid-Grade",
    "premium":   "Premium",
    "supreme":   "Premium",
    "v-power":   "Premium",
    "extra":     "Mid-Grade",   # Exxon Extra ~ mid-grade
    "diesel":    "Diesel",
    "unleaded":  "Regular",
    "ultra":     "Premium",     # e.g. Ultra 93
    "prem":      "Premium",
    "reg":       "Regular",
}

def normalise_grade(raw: str) -> str:
    """Map free-text fuel labels to a canonical grade."""
    if pd.isna(raw) or str(raw).strip().upper() == "NA":
        return "Unknown"
    raw_lower = str(raw).lower().strip()
    for suffix in (" self", " gasoline", " unleaded", " #2", " cash", " card"):
        raw_lower = raw_lower.replace(suffix, "")
    raw_lower = raw_lower.strip()
    for key, canonical in GRADE_CANONICAL.items():
        if key in raw_lower:
            return canonical
    return raw.strip()


def normalise_payment(raw: str) -> str:
    """Normalise payment type to Cash / Credit / Both / NA."""
    if pd.isna(raw) or str(raw).strip().upper() == "NA":
        return "NA"
    r = str(raw).lower()
    has_cash   = "cash" in r
    has_credit = any(x in r for x in ("credit", "debit", "card"))
    has_both   = "both" in r
    if has_both or (has_cash and has_credit):
        return "Both"
    if has_cash:
        return "Cash"
    if has_credit:
        return "Credit"
    return "NA"


# -------------------------------------------------
# img_idx parser
# -------------------------------------------------

def _parse_img_idx(row) -> int:
    """
    Determine the GT file index for a row.

    Priority:
      1. Source_File contains a valid filename -> use its numeric stem.
      2. Source_File is absent/NA              -> use Image_Number directly.

    Two naming conventions can exist in the same CSV:
      - Rows with Source_File="NA" or empty: Image_Number IS the file index (e.g. 0 -> 0.csv).
      - Rows with Source_File="0.png":       file index is 0, Image_Number may be 1 (off by one).
    """
    sf = str(row.get("Source_File", "")).strip()
    if sf and sf.upper() != "NA":
        stem   = sf.split(".")[0]              # "10.png" -> "10"
        digits = re.sub(r"[^0-9]", "", stem)   # safety: strip any non-digit chars
        if digits:
            return int(digits)
    # No valid Source_File -> Image_Number is the file index
    return int(row["Image_Number"])


# -------------------------------------------------
# Loading helpers
# -------------------------------------------------

def load_extracted(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["img_idx"]      = df.apply(_parse_img_idx, axis=1)
    df["grade_norm"]   = df["Fuel_Type"].apply(normalise_grade)
    df["payment_norm"] = df["Payment_Type"].apply(normalise_payment)
    df["Price"]        = pd.to_numeric(df["Price"], errors="coerce")
    df["_row_id"]      = range(len(df))   # stable unique id for EXTRA detection
    return df


def load_ground_truth(gt_dir: str) -> pd.DataFrame:
    """Load all 0.csv ... N.csv from gt_dir into a single DataFrame."""
    records = []
    gt_path = Path(gt_dir)
    csv_files = sorted(
        [p for p in gt_path.glob("*.csv") if p.stem.isdigit()],
        key=lambda p: int(p.stem)
    )
    if not csv_files:
        raise FileNotFoundError(f"No numeric ground-truth CSV files found in: {gt_dir}")
    for csv_file in csv_files:
        idx = int(csv_file.stem)
        try:
            tmp = pd.read_csv(csv_file)
            tmp.columns = tmp.columns.str.strip()
            tmp["img_idx"] = idx
            records.append(tmp)
        except Exception as e:
            print(f"  [WARN] Could not load {csv_file}: {e}")
    gt = pd.concat(records, ignore_index=True)
    gt.rename(columns={"Grade": "grade_raw", "Cash/Credit": "payment_raw"}, inplace=True)
    gt["grade_norm"]   = gt["grade_raw"].apply(normalise_grade)
    gt["payment_norm"] = gt["payment_raw"].apply(normalise_payment)
    gt["Price"]        = pd.to_numeric(gt["Price"], errors="coerce")
    return gt


# -------------------------------------------------
# Matching logic
# -------------------------------------------------

def match_entries(ext_df: pd.DataFrame,
                  gt_df: pd.DataFrame,
                  price_tol: float = 0.01) -> pd.DataFrame:
    """
    For each ground-truth entry, find the best matching extracted entry
    on the same img_idx and normalised grade.
    """
    results = []

    for img_idx, gt_group in gt_df.groupby("img_idx"):
        ext_group  = ext_df[ext_df["img_idx"] == img_idx].copy()

        # Representative source filename for display
        sf_series  = ext_group["Source_File"] if "Source_File" in ext_group.columns else pd.Series([])
        valid_sf   = sf_series[sf_series.notna() & (sf_series.str.upper() != "NA")]
        source_file = valid_sf.iloc[0] if len(valid_sf) > 0 else f"{img_idx}.png"

        for _, gt_row in gt_group.iterrows():
            grade_matches = ext_group[ext_group["grade_norm"] == gt_row["grade_norm"]]

            if grade_matches.empty:
                results.append({
                    "img_idx":         img_idx,
                    "source_file":     source_file,
                    "gt_grade":        gt_row.get("grade_raw", gt_row["grade_norm"]),
                    "gt_price":        gt_row["Price"],
                    "gt_payment":      gt_row["payment_norm"],
                    "ext_grade":       None,
                    "ext_price":       None,
                    "ext_payment":     None,
                    "ext_brand":       None,
                    "ext_confidence":  None,
                    "price_error":     None,
                    "price_match":     False,
                    "grade_matched":   False,
                    "payment_match":   False,
                    "status":          "MISSED",
                    "_matched_row_id": None,
                })
                continue

            grade_matches = grade_matches.copy()
            grade_matches["price_diff"] = (grade_matches["Price"] - gt_row["Price"]).abs()
            best = grade_matches.loc[grade_matches["price_diff"].idxmin()]

            price_err  = float(best["price_diff"])
            price_ok   = price_err <= price_tol
            payment_ok = (best["payment_norm"] == gt_row["payment_norm"]
                          or gt_row["payment_norm"] == "NA"
                          or best["payment_norm"] == "NA")

            if price_ok:
                status = "CORRECT"
            elif price_err <= 0.05:
                status = "CLOSE"
            else:
                status = "WRONG_PRICE"

            results.append({
                "img_idx":         img_idx,
                "source_file":     source_file,
                "gt_grade":        gt_row.get("grade_raw", gt_row["grade_norm"]),
                "gt_price":        gt_row["Price"],
                "gt_payment":      gt_row["payment_norm"],
                "ext_grade":       best["Fuel_Type"],
                "ext_price":       best["Price"],
                "ext_payment":     best["payment_norm"],
                "ext_brand":       best.get("Gas_Station_Brand", None),
                "ext_confidence":  best.get("Confidence", None),
                "price_error":     round(price_err, 4),
                "price_match":     price_ok,
                "grade_matched":   True,
                "payment_match":   payment_ok,
                "status":          status,
                "_matched_row_id": int(best["_row_id"]),
            })

    # Flag hallucinated (extra) extractions using stable row IDs
    matched_row_ids = {r["_matched_row_id"] for r in results if r["_matched_row_id"] is not None}

    for _, ext_row in ext_df.iterrows():
        if ext_row["_row_id"] not in matched_row_ids:
            sf = ext_row.get("Source_File", None)
            results.append({
                "img_idx":         ext_row["img_idx"],
                "source_file":     sf if (pd.notna(sf) and str(sf).upper() != "NA") else f"{ext_row['img_idx']}.png",
                "gt_grade":        None,
                "gt_price":        None,
                "gt_payment":      None,
                "ext_grade":       ext_row["Fuel_Type"],
                "ext_price":       ext_row["Price"],
                "ext_payment":     ext_row["payment_norm"],
                "ext_brand":       ext_row.get("Gas_Station_Brand", None),
                "ext_confidence":  ext_row.get("Confidence", None),
                "price_error":     None,
                "price_match":     False,
                "grade_matched":   False,
                "payment_match":   False,
                "status":          "EXTRA",
                "_matched_row_id": None,
            })

    df = pd.DataFrame(results).sort_values(["img_idx", "gt_price"])
    df.drop(columns=["_matched_row_id"], inplace=True, errors="ignore")
    return df


# -------------------------------------------------
# Summary metrics
# -------------------------------------------------

def compute_metrics(matches: pd.DataFrame, price_tol: float) -> dict:
    gt_rows    = matches[matches["status"] != "EXTRA"]
    extra_rows = matches[matches["status"] == "EXTRA"]

    total_gt    = len(gt_rows)
    correct     = int((gt_rows["status"] == "CORRECT").sum())
    close       = int((gt_rows["status"] == "CLOSE").sum())
    wrong_price = int((gt_rows["status"] == "WRONG_PRICE").sum())
    missed      = int((gt_rows["status"] == "MISSED").sum())
    extra       = len(extra_rows)

    denom_p = correct + wrong_price + extra
    precision = correct / denom_p if denom_p > 0 else 0.0
    recall    = correct / total_gt  if total_gt > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    price_errors = gt_rows["price_error"].dropna()
    mae     = price_errors.mean()      if len(price_errors) > 0 else float("nan")
    rmse    = np.sqrt((price_errors**2).mean()) if len(price_errors) > 0 else float("nan")
    max_err = price_errors.max()       if len(price_errors) > 0 else float("nan")

    tol_label = str(price_tol)
    return {
        "total_gt_entries":              total_gt,
        "correct_matches":               correct,
        "close_matches_5cent":           close,
        "wrong_price":                   wrong_price,
        "missed":                        missed,
        "extra_hallucinated":            extra,
        f"precision@{tol_label}":        round(precision, 4),
        f"recall@{tol_label}":           round(recall, 4),
        f"F1@{tol_label}":               round(f1, 4),
        "MAE_dollars":                   round(mae,     4),
        "RMSE_dollars":                  round(rmse,    4),
        "max_error_dollars":             round(max_err, 4),
    }


def per_image_summary(matches: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for img_idx, grp in matches.groupby("img_idx"):
        gt_rows = grp[grp["status"] != "EXTRA"]
        errs    = gt_rows["price_error"].dropna()
        rows.append({
            "img_idx":          img_idx,
            "source_file":      grp["source_file"].iloc[0],
            "gt_count":         len(gt_rows),
            "correct":          int((gt_rows["status"] == "CORRECT").sum()),
            "close":            int((gt_rows["status"] == "CLOSE").sum()),
            "wrong_price":      int((gt_rows["status"] == "WRONG_PRICE").sum()),
            "missed":           int((gt_rows["status"] == "MISSED").sum()),
            "extra":            int((grp["status"] == "EXTRA").sum()),
            "mean_price_error": round(errs.mean(), 4) if len(errs) > 0 else None,
        })
    return pd.DataFrame(rows).set_index("img_idx")


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate gas price extraction accuracy.")
    parser.add_argument("--extracted",  required=True,
                        help="Path to pipeline output CSV.")
    parser.add_argument("--gt_dir",     required=True,
                        help="Directory containing ground-truth CSVs (0.csv ... N.csv).")
    parser.add_argument("--tolerance",  type=float, default=0.01,
                        help="Price match tolerance in dollars (default 0.01).")
    parser.add_argument("--output",     default="evaluation_report.csv",
                        help="Output path for the detailed match table.")
    args = parser.parse_args()

    print("=" * 60)
    print("  Gas Price Extraction Evaluator")
    print("=" * 60)

    print(f"\n[1] Loading extracted prices: {args.extracted}")
    ext_df = load_extracted(args.extracted)

    # Diagnostics: show how img_idx was resolved
    has_sf  = ext_df["Source_File"].notna() & (ext_df["Source_File"].str.upper() != "NA") \
              if "Source_File" in ext_df.columns else pd.Series([False] * len(ext_df))
    n_from_sf  = has_sf.sum()
    n_from_num = (~has_sf).sum()
    print(f"    -> {len(ext_df)} rows: {n_from_sf} img_idx from Source_File, "
          f"{n_from_num} from Image_Number (no Source_File)")
    print(f"    -> Unique img_idx values: {sorted(ext_df['img_idx'].unique())}")

    print(f"\n[2] Loading ground truth: {args.gt_dir}")
    gt_df = load_ground_truth(args.gt_dir)
    print(f"    -> {len(gt_df)} GT entries across img_idx: {sorted(gt_df['img_idx'].unique())}")

    # Warn about GT files with no matching extracted rows
    gt_only  = set(gt_df["img_idx"].unique()) - set(ext_df["img_idx"].unique())
    ext_only = set(ext_df["img_idx"].unique()) - set(gt_df["img_idx"].unique())
    if gt_only:
        print(f"  [WARN] GT files with NO extracted rows: {sorted(gt_only)}")
    if ext_only:
        print(f"  [INFO] Extracted img_idx with no GT file (will be EXTRA): {sorted(ext_only)}")

    print(f"\n[3] Matching (tolerance +-${args.tolerance:.3f}) ...")
    matches = match_entries(ext_df, gt_df, price_tol=args.tolerance)

    print(f"\n[4] Metrics:")
    metrics = compute_metrics(matches, price_tol=args.tolerance)

    print("\n" + "-" * 42)
    for k, v in metrics.items():
        print(f"  {k:<34s}: {v}")
    print("-" * 42)

    print("\n[5] Per-image breakdown:")
    per_img = per_image_summary(matches)
    print(per_img.to_string())

    report_path   = args.output
    per_img_path  = report_path.replace(".csv", "_per_image.csv")
    metrics_path  = report_path.replace(".csv", "_metrics.csv")

    matches.to_csv(report_path, index=False)
    per_img.to_csv(per_img_path)
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    print(f"\n[6] Saved:")
    print(f"    Detailed matches  -> {report_path}")
    print(f"    Per-image summary -> {per_img_path}")
    print(f"    Overall metrics   -> {metrics_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
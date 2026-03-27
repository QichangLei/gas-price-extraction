"""
make_price_panel.py
────────────────────
Generate a slide-ready price summary panel from a pipeline output CSV
or from manually specified prices.

Usage:
    python src/annotation/make_price_panel.py \
        --csv output/pipeline/gas_prices_geo_20260312_181901.csv \
        --brand "TA / TA EXPRESS" \
        --output output/annotation/IMG_0977_price_panel.png
"""

from __future__ import annotations
import argparse
import datetime
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# ── Palette ───────────────────────────────────────────────────────────────────
BG          = (10,  13,  22)
HEADER_BG   = (10,  75,  48)
ROW_EVEN    = (14,  18,  30)
ROW_ODD     = (18,  23,  40)
SUBHDR_BG   = (16,  22,  38)
C_ACCENT    = (0,   220, 130)
C_WHITE     = (255, 255, 255)
C_DIM       = (140, 160, 185)
C_BRAND     = (255, 215,  55)
C_CASH      = ( 85, 220, 120)
C_CREDIT    = ( 85, 180, 255)
C_PRICE     = ( 55, 215, 255)


def _load_font(size: int, bold: bool = False):
    paths_bold    = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                     "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]
    paths_regular = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                     "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"]
    for p in (paths_bold if bold else paths_regular):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _text_w(font, text: str) -> int:
    try:
        return int(font.getlength(text))
    except AttributeError:
        return len(text) * (font.size // 2)


def generate_panel(
    rows: list[dict],          # [{"fuel": str, "cash": str|None, "credit": str|None}]
    brand: str,
    output_path: str,
    title: str = "GAS PRICE AI  ·  GEMINI 3 FLASH",
) -> None:
    """
    Draw a cash/credit price table panel suitable for slides.

    rows example:
        [{"fuel": "Regular", "cash": "2.169", "credit": "2.269"},
         {"fuel": "Diesel",  "cash": "3.649", "credit": "3.709"}]
    """
    W        = 820
    PAD      = 32
    HDR_H    = 62
    BRAND_H  = 72
    SUBHDR_H = 36
    ROW_H    = 60
    FOOTER_H = 40
    H = HDR_H + BRAND_H + SUBHDR_H + len(rows) * ROW_H + FOOTER_H + PAD

    img  = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    # Fonts
    f_title   = _load_font(19, bold=True)
    f_sub     = _load_font(13)
    f_brand   = _load_font(30, bold=True)
    f_col_hdr = _load_font(13)
    f_fuel    = _load_font(18)
    f_price   = _load_font(24, bold=True)
    f_footer  = _load_font(12)

    # ── Header bar ────────────────────────────────────────────────────────────
    draw.rectangle([0, 0, W, HDR_H], fill=HEADER_BG)
    draw.text((PAD, 20), "GAS PRICE AI", font=f_title, fill=C_ACCENT)
    sub = "POWERED BY GEMINI 3 FLASH"
    draw.text((W - PAD - _text_w(f_sub, sub), 24), sub, font=f_sub, fill=C_DIM)
    draw.line([(0, HDR_H), (W, HDR_H)], fill=C_ACCENT, width=2)

    cy = HDR_H

    # ── Brand ─────────────────────────────────────────────────────────────────
    draw.rectangle([0, cy, W, cy + BRAND_H], fill=ROW_EVEN)
    draw.text((PAD, cy + 8),  "STATION",        font=f_col_hdr, fill=C_DIM)
    draw.text((PAD, cy + 26), brand.upper(),    font=f_brand,   fill=C_BRAND)
    cy += BRAND_H
    draw.line([(PAD, cy), (W - PAD, cy)], fill=(*C_ACCENT, 100), width=1)

    # ── Column headers ────────────────────────────────────────────────────────
    draw.rectangle([0, cy, W, cy + SUBHDR_H], fill=SUBHDR_BG)
    COL_FUEL   = PAD
    COL_CASH   = 380
    COL_CREDIT = 600
    draw.text((COL_FUEL,   cy + 10), "FUEL TYPE",     font=f_col_hdr, fill=C_DIM)
    draw.text((COL_CASH,   cy + 10), "CASH",          font=f_col_hdr, fill=C_CASH)
    draw.text((COL_CREDIT, cy + 10), "CREDIT / CARD", font=f_col_hdr, fill=C_CREDIT)
    cy += SUBHDR_H

    # ── Price rows ────────────────────────────────────────────────────────────
    for i, row in enumerate(rows):
        bg = ROW_EVEN if i % 2 == 0 else ROW_ODD
        draw.rectangle([0, cy, W, cy + ROW_H], fill=bg)

        mid = cy + ROW_H // 2

        # Fuel type
        draw.text((COL_FUEL, mid - 12), row["fuel"], font=f_fuel, fill=C_WHITE)

        # Cash price
        if row.get("cash"):
            draw.text((COL_CASH, mid - 14), f"${row['cash']}", font=f_price, fill=C_CASH)

        # Credit price
        if row.get("credit"):
            draw.text((COL_CREDIT, mid - 14), f"${row['credit']}", font=f_price, fill=C_CREDIT)

        # Subtle row separator
        draw.line([(PAD, cy + ROW_H - 1), (W - PAD, cy + ROW_H - 1)],
                  fill=(30, 38, 60), width=1)

        cy += ROW_H

    # ── Accent border ─────────────────────────────────────────────────────────
    draw.line([(PAD, cy), (W - PAD, cy)], fill=(*C_ACCENT, 80), width=1)
    cy += 6

    # ── Footer ────────────────────────────────────────────────────────────────
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    draw.text((PAD, cy + 10),
              f"Extracted via Gemini Vision OCR  ·  {ts}",
              font=f_footer, fill=C_DIM)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    print(f"Panel saved: {output_path}")


# ── Load from CSV ─────────────────────────────────────────────────────────────

def load_from_csv(csv_path: str) -> tuple[list[dict], str]:
    """
    Read pipeline output CSV and build cash/credit rows per fuel grade.
    Returns (rows, brand).
    """
    import csv, re

    GRADE_MAP = {
        "regular": "Regular", "unleaded": "Regular", "reg": "Regular",
        "midgrade": "Mid-Grade", "mid-grade": "Mid-Grade", "plus": "Mid-Grade",
        "premium": "Premium", "supreme": "Premium", "v-power": "Premium",
        "diesel": "Diesel",
    }

    def normalise(raw: str) -> str:
        r = raw.lower()
        # strip payment suffix
        r = re.sub(r"\s*(cash|credit|card)\s*$", "", r).strip()
        for k, v in GRADE_MAP.items():
            if k in r:
                return v
        return raw.strip().title()

    def payment(raw: str) -> str | None:
        r = raw.lower()
        if "credit" in r or "card" in r:
            return "credit"
        if "cash" in r:
            return "cash"
        return None

    by_grade: dict[str, dict] = defaultdict(lambda: {"cash": [], "credit": [], "either": []})
    brands = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                price = float(row.get("Price", ""))
            except ValueError:
                continue
            fuel  = row.get("Fuel_Type", "")
            grade = normalise(fuel)
            pay   = payment(fuel) or payment(row.get("Payment_Type", ""))
            brand = row.get("Gas_Station_Brand", "")
            if brand and brand != "NA":
                brands.append(brand)
            if pay == "cash":
                by_grade[grade]["cash"].append(price)
            elif pay == "credit":
                by_grade[grade]["credit"].append(price)
            else:
                by_grade[grade]["either"].append(price)

    def majority(lst):
        if not lst:
            return None
        from collections import Counter
        val, _ = Counter(f"{v:.3f}" for v in lst).most_common(1)[0]
        return val

    rows = []
    grade_order = ["Regular", "Mid-Grade", "Premium", "Diesel"]
    seen = set(by_grade.keys())
    ordered = [g for g in grade_order if g in seen] + \
              [g for g in seen if g not in grade_order]

    for grade in ordered:
        d = by_grade[grade]
        rows.append({
            "fuel":   grade,
            "cash":   majority(d["cash"])   or majority(d["either"]),
            "credit": majority(d["credit"]) or majority(d["either"]),
        })

    from collections import Counter
    brand = Counter(brands).most_common(1)[0][0] if brands else "Unknown"
    return rows, brand


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",    required=True, help="Pipeline output CSV")
    p.add_argument("--brand",  default=None,  help="Override station brand name")
    p.add_argument("--output", default="output/annotation/price_panel.png")
    args = p.parse_args()

    rows, brand = load_from_csv(args.csv)
    generate_panel(rows, args.brand or brand, args.output)


if __name__ == "__main__":
    main()

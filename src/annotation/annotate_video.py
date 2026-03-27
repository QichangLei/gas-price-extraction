#!/usr/bin/env python3
"""
annotate_video.py
─────────────────
Annotates a gas station video using Gemini Vision:
  - Tactical corner-bracket bounding box around the detected price sign
  - Professional HUD-style info panel (brand, fuel prices, confidence)

Usage:
    python annotate_video.py --input ../../data/videos/gas_station_1.mp4
    python annotate_video.py --input video.mp4 --output out.mp4 --every 10

Requirements:
    pip install google-genai opencv-python pillow numpy
    export GEMINI_API_KEY="your-key-here"
"""

from __future__ import annotations
import argparse
import csv
import io
from collections import Counter, defaultdict
import json
import logging
import re
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from google import genai
from google.genai import types

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL          = "gemini-3-flash-preview"
SAMPLE_EVERY_N = 15

# Color palette  (R, G, B)
C_ACCENT  = (0,   220, 130)   # mint-green  — main accent
C_WHITE   = (255, 255, 255)
C_DIM     = (140, 160, 185)   # muted blue-grey for secondary text
C_BRAND   = (255, 215,  55)   # golden-yellow for station name
C_PRICE   = ( 55, 215, 255)   # bright cyan for price values
C_CONF_H  = ( 75, 220,  90)   # green  — confidence ≥ 85
C_CONF_M  = (245, 195,  55)   # amber  — confidence 70–84
C_CONF_L  = (240,  75,  75)   # red    — confidence < 70

# Panel RGBA backgrounds  (R, G, B, A)  — A=255 means fully opaque
PANEL_BG       = (10,  13,  22, 218)   # dark navy,  85 % opaque
HEADER_BG      = (10,  75,  48, 240)   # dark green, 94 % opaque
SEPARATOR_COL  = (*C_ACCENT, 120)      # semi-transparent accent line

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Prompt ────────────────────────────────────────────────────────────────────

ANNOTATION_PROMPT = """You are a gas price sign detector and OCR engine for video frames.

Analyze this frame and respond with JSON only — no markdown, no explanation.

{
  "sign_detected": true or false,
  "bounding_box": [y_min, x_min, y_max, x_max],
  "brand": "Shell",
  "prices": [
    {"fuel_type": "Regular", "price": "3.459", "payment_type": "Cash", "confidence": 92}
  ]
}

RULES

bounding_box:
  - Normalized 0-1000 coordinates. 0 = top/left edge, 1000 = bottom/right edge.
  - Format: [y_min, x_min, y_max, x_max]
  - Enclose the entire gas price display board tightly.
  - Set to null if no gas price sign is visible.

brand:
  - The gas station brand / chain name if a logo or name is clearly visible (e.g. "Shell", "BP", "Chevron").
  - Set to null if not visible or uncertain. Never guess from colors alone.

prices:
  - price: always 3 decimals (e.g. 3.459). Convert 9/10 fraction: 2.99⁹ → 2.999
  - fuel_type: copy exactly as shown on the sign.
  - payment_type: only if explicitly written (Cash / Credit), else null.
  - confidence: integer. >90 = clear, 80-90 = slight blur, 70-80 = hard, <70 = barely readable.
  - Empty array [] if no prices are visible.

If sign_detected is false, set bounding_box to null, brand to null, and prices to [].
Do not guess. Do not explain. JSON only.
"""


# ── Gemini helpers ────────────────────────────────────────────────────────────

def _frame_to_jpeg_bytes(frame_bgr: np.ndarray) -> bytes:
    pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def query_gemini(client: genai.Client, frame_bgr: np.ndarray) -> dict:
    """Send a single frame to Gemini and return the parsed JSON result."""
    jpeg = _frame_to_jpeg_bytes(frame_bgr)
    response = client.models.generate_content(
        model=MODEL,
        contents=[
            types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"),
            ANNOTATION_PROMPT,
        ],
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            temperature=1.0,
        ),
    )
    raw = response.text.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    return json.loads(raw)


# ── Font loader ───────────────────────────────────────────────────────────────

def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Try common system font paths; fall back to PIL built-in. Never below 12px."""
    size = max(12, size)
    paths_bold = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    ]
    paths_regular = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]
    for p in (paths_bold if bold else paths_regular):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    # Pillow >= 10 supports size param on load_default
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _conf_color(conf) -> tuple:
    if conf is None:
        return C_DIM
    try:
        c = int(conf)
    except (ValueError, TypeError):
        return C_DIM
    return C_CONF_H if c >= 85 else (C_CONF_M if c >= 70 else C_CONF_L)


def _rounded_rect(draw: ImageDraw.ImageDraw, xy, fill, radius: int = 5):
    """Draw a filled rounded rectangle (works on Pillow ≥ 8.2; falls back to plain rect)."""
    try:
        draw.rounded_rectangle(xy, radius=radius, fill=fill)
    except AttributeError:
        draw.rectangle(xy, fill=fill)


def _draw_corner_box(overlay: ImageDraw.ImageDraw,
                     x0, y0, x1, y1, scale: float) -> None:
    """
    Tactical corner-bracket markers — the modern detection-UI aesthetic.
    A thin ghost rectangle provides spatial context.
    """
    cl = max(12, int(min(x1 - x0, y1 - y0) * 0.13))  # corner length
    t  = max(2, int(3 * scale))                         # line thickness
    accent_solid = (*C_ACCENT, 255)
    accent_ghost = (*C_ACCENT, 55)

    # Ghost perimeter
    overlay.rectangle([x0, y0, x1, y1], outline=accent_ghost, width=1)

    # Corner L-brackets
    for px, py, sx, sy in [(x0, y0, 1, 1), (x1, y0, -1, 1),
                            (x0, y1, 1, -1), (x1, y1, -1, -1)]:
        overlay.line([(px, py), (px + sx * cl, py)], fill=accent_solid, width=t)
        overlay.line([(px, py), (px, py + sy * cl)], fill=accent_solid, width=t)

    # Corner dots for extra polish
    r = max(2, t - 1)
    for px, py in [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]:
        overlay.ellipse([px - r, py - r, px + r, py + r], fill=accent_solid)


def _draw_bbox_label(overlay: ImageDraw.ImageDraw,
                     x0, y0, brand: str | None, scale: float) -> None:
    """Small label tag just above the bounding box."""
    font     = _load_font(max(13, int(14 * scale)), bold=True)
    label    = brand.upper() if brand else "GAS PRICE SIGN"
    pad_x, pad_y = int(8 * scale), int(4 * scale)

    try:
        tw = font.getlength(label)
    except AttributeError:
        tw = len(label) * int(7 * scale)

    th    = int(11 * scale)
    lx0   = x0
    ly1   = max(y0 - 2, int(20 * scale))
    ly0   = ly1 - th - pad_y * 2

    _rounded_rect(overlay, [lx0, ly0, lx0 + int(tw) + pad_x * 2, ly1],
                  fill=(*C_ACCENT, 230), radius=3)
    overlay.text((lx0 + pad_x, ly0 + pad_y), label,
                 font=font, fill=(10, 13, 22, 255))


def _draw_info_panel(overlay: ImageDraw.ImageDraw,
                     frame_w: int, frame_h: int,
                     brand: str | None,
                     prices: list[dict],
                     scale: float,
                     side: str = "left") -> None:
    """
    HUD-style info panel at bottom-left.

    Layout:
      ┌──────────────────────────────┐
      │  GAS PRICE AI  GEMINI 3 FLASH│  ← header (dark green)
      ├──────────────────────────────┤
      │  STATION   Shell             │  ← brand (golden) — hidden if null
      ├──────────────────────────────┤
      │  FUEL TYPE       PRICE  CONF │  ← column headers (dimmed)
      │  Regular        $3.459   ●   │
      │  Mid-Grade      $3.659   ●   │
      │  Premium        $3.859   ●   │
      └──────────────────────────────┘
    """
    # ── Fonts — base sizes are for 1080p; min clamps ensure legibility at 480p ──
    f_header  = _load_font(max(15, int(18 * scale)), bold=True)
    f_brand   = _load_font(max(20, int(24 * scale)), bold=True)
    f_label   = _load_font(max(12, int(13 * scale)), bold=False)
    f_fuel    = _load_font(max(14, int(17 * scale)), bold=False)
    f_price   = _load_font(max(15, int(18 * scale)), bold=True)

    # ── Dimensions — min clamps keep panel readable on small videos ───────────
    PANEL_W   = max(220, int(240 * scale))
    HDR_H     = max(36,  int(34 * scale))
    BRAND_H   = max(54,  int(50 * scale)) if brand else 0
    SEP_H     = max(1,   int(1  * scale))
    ROW_H     = max(32,  int(30 * scale))
    SEC_PAD   = max(8,   int(8  * scale))
    CORNER    = max(5,   int(5  * scale))
    MARGIN    = max(14,  int(14 * scale))

    PANEL_H = (HDR_H
               + (BRAND_H + SEP_H + SEC_PAD if brand else 0)
               + SEC_PAD + len(prices) * ROW_H + SEC_PAD)

    px  = MARGIN if side == "left" else frame_w - PANEL_W - MARGIN
    py  = frame_h - PANEL_H - MARGIN

    # ── Outer panel (dark navy) ────────────────────────────────────────────────
    _rounded_rect(overlay, [px, py, px + PANEL_W, py + PANEL_H],
                  fill=PANEL_BG, radius=CORNER)

    # ── Header bar (dark green) ────────────────────────────────────────────────
    _rounded_rect(overlay,
                  [px, py, px + PANEL_W, py + HDR_H],
                  fill=HEADER_BG, radius=CORNER)
    # Clip bottom corners of header so it blends into panel
    overlay.rectangle([px, py + CORNER, px + PANEL_W, py + HDR_H],
                      fill=HEADER_BG)

    overlay.text((px + int(10 * scale), py + int(9 * scale)),
                 "GAS PRICE AI",
                 font=f_header, fill=(*C_ACCENT, 255))
    right_label = "GEMINI 3 FLASH"
    try:
        rw = f_label.getlength(right_label)
    except AttributeError:
        rw = len(right_label) * int(6 * scale)
    overlay.text((px + PANEL_W - int(rw) - int(8 * scale),
                  py + int(10 * scale)),
                 right_label,
                 font=f_label, fill=(*C_DIM, 200))

    cursor_y = py + HDR_H

    # ── Brand section ─────────────────────────────────────────────────────────
    if brand:
        cursor_y += int(5 * scale)
        overlay.text((px + int(10 * scale), cursor_y),
                     "STATION",
                     font=f_label, fill=(*C_DIM, 180))
        cursor_y += int(13 * scale)
        overlay.text((px + int(10 * scale), cursor_y),
                     brand,
                     font=f_brand, fill=(*C_BRAND, 255))
        cursor_y += int(18 * scale) + SEC_PAD

        # Thin separator line
        overlay.line([(px + int(8 * scale), cursor_y),
                      (px + PANEL_W - int(8 * scale), cursor_y)],
                     fill=SEPARATOR_COL, width=max(1, SEP_H))
        cursor_y += SEP_H + SEC_PAD

    else:
        cursor_y += SEC_PAD

    # ── Price rows ─────────────────────────────────────────────────────────────
    DOT_R    = max(4, int(5  * scale))
    DOT_X    = px + PANEL_W - int(18 * scale)
    PRICE_X  = px + PANEL_W - int(25 * scale)

    for p in prices:
        ft   = (p.get("fuel_type") or "Unknown")[:14]
        pr   = p.get("price") or "—"
        pt   = p.get("payment_type")
        conf = p.get("confidence")
        mid_y = cursor_y + ROW_H // 2

        # Fuel type
        overlay.text((px + int(10 * scale), cursor_y + int(4 * scale)),
                     ft, font=f_fuel, fill=(*C_WHITE, 240))

        # Price value (right-aligned to PRICE_X)
        price_str = f"${pr}"
        if pt:
            price_str += f" {pt[:2]}"
        try:
            pw = f_price.getlength(price_str)
        except AttributeError:
            pw = len(price_str) * int(8 * scale)
        overlay.text((PRICE_X - int(pw) - int(4 * scale),
                      cursor_y + int(3 * scale)),
                     price_str, font=f_price, fill=(*C_PRICE, 255))

        # Confidence dot
        dot_color = (*_conf_color(conf), 255)
        overlay.ellipse([DOT_X - DOT_R, mid_y - DOT_R,
                         DOT_X + DOT_R, mid_y + DOT_R],
                        fill=dot_color)

        cursor_y += ROW_H

    # ── Thin accent border around entire panel ─────────────────────────────────
    try:
        overlay.rounded_rectangle([px, py, px + PANEL_W, py + PANEL_H],
                                   radius=CORNER,
                                   outline=(*C_ACCENT, 90), width=1)
    except (AttributeError, TypeError):
        overlay.rectangle([px, py, px + PANEL_W, py + PANEL_H],
                          outline=(*C_ACCENT, 90), width=1)


# ── Master drawing function ───────────────────────────────────────────────────

def draw_annotations(frame_bgr: np.ndarray, result: dict) -> np.ndarray:
    """Composite all HUD elements onto the frame using PIL alpha blending."""
    h, w  = frame_bgr.shape[:2]
    scale = min(w, h) / 720.0          # normalize: 1.0 = 720 p, 1.5 = 1080 p

    base    = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)

    brand  = result.get("brand") or None
    prices = result.get("prices") or []

    # ── Bounding box ──────────────────────────────────────────────────────────
    panel_side = "left"   # default panel position
    bbox = result.get("bounding_box")
    if bbox and len(bbox) == 4 and all(v is not None for v in bbox):
        y0n, x0n, y1n, x1n = bbox
        x0 = int(x0n / 1000 * w)
        y0 = int(y0n / 1000 * h)
        x1 = int(x1n / 1000 * w)
        y1 = int(y1n / 1000 * h)

        # Place panel on the opposite side from the sign
        bbox_center_x = (x0 + x1) / 2
        panel_side = "right" if bbox_center_x < w / 2 else "left"

        _draw_corner_box(draw, x0, y0, x1, y1, scale)
        _draw_bbox_label(draw, x0, y0, brand, scale)

    # ── Info panel ────────────────────────────────────────────────────────────
    if prices or brand:
        _draw_info_panel(draw, w, h, brand, prices, scale, side=panel_side)

    composite = Image.alpha_composite(base, overlay)
    return cv2.cvtColor(np.array(composite.convert("RGB")), cv2.COLOR_RGB2BGR)


# ── Price validation & summary export ────────────────────────────────────────

_GRADE_MAP = {
    "regular":   "Regular",  "unleaded": "Regular",  "reg":     "Regular",
    "midgrade":  "Mid-Grade","mid-grade":"Mid-Grade", "plus":    "Mid-Grade",
    "premium":   "Premium",  "supreme":  "Premium",   "v-power": "Premium",
    "diesel":    "Diesel",
}
_GRADE_ORDER = ["Regular", "Mid-Grade", "Premium", "Diesel"]


def _normalise_fuel(fuel_type: str, payment_type: str) -> tuple[str, str]:
    """
    Split a potentially mixed fuel_type string (e.g. "Regular CASH") into
    a canonical grade ("Regular") and payment ("Cash" / "Credit" / "NA").
    Also accepts a separate payment_type field as fallback.
    """
    ft = fuel_type.strip()
    ft_lower = ft.lower()

    # Extract payment from fuel_type string
    pay = None
    if "credit" in ft_lower or "card" in ft_lower:
        pay = "Credit"
        ft_lower = re.sub(r"\s*(credit|card)\s*", " ", ft_lower).strip()
    elif "cash" in ft_lower:
        pay = "Cash"
        ft_lower = re.sub(r"\s*cash\s*", " ", ft_lower).strip()

    # Fallback to explicit payment_type field
    if pay is None:
        pt = str(payment_type or "").strip().lower()
        if "credit" in pt or "card" in pt:
            pay = "Credit"
        elif "cash" in pt:
            pay = "Cash"

    # Normalize grade
    for key, canonical in _GRADE_MAP.items():
        if key in ft_lower:
            return canonical, pay or "NA"
    return ft.title(), pay or "NA"


def validate_prices(price_log: list[dict],
                    brand_log: list[str],
                    threshold: float = 0.60,
                    min_presence: float = 0.05) -> tuple[list[dict], str | None]:
    """
    Majority-vote validation across all Gemini API calls.

    Normalises fuel grade and payment type before grouping, so entries like
    "Regular CASH" and fuel_type="Regular" + payment_type="Cash" are merged.

    Two-gate filter per (grade, payment) group:
      1. min_presence  — must appear in >= min_presence fraction of all API calls
      2. threshold     — top price must account for >= threshold of appearances

    Returns (validated_rows, majority_brand).
    """
    total_api_calls = len(brand_log)

    # ── Brand majority vote ───────────────────────────────────────────────────
    valid_brands = [b for b in brand_log if b]
    majority_brand = Counter(valid_brands).most_common(1)[0][0] if valid_brands else None

    # ── Aggregate by (grade, payment) ────────────────────────────────────────
    by_key: dict[tuple, list[str]] = defaultdict(list)
    for entry in price_log:
        ft  = (entry.get("fuel_type")    or "").strip()
        pt  = (entry.get("payment_type") or "").strip()
        pr  = str(entry.get("price")     or "").strip()
        if not ft or not pr or pr == "nan":
            continue
        grade, pay = _normalise_fuel(ft, pt)
        by_key[(grade, pay)].append(pr)

    validated = []
    for (grade, pay), prices in by_key.items():
        total_seen = len(prices)
        presence   = total_seen / total_api_calls if total_api_calls > 0 else 0

        if presence < min_presence:
            log.debug("  SKIP %-15s %-8s — presence %.1f%% < %.0f%%",
                      grade, pay, presence * 100, min_presence * 100)
            continue

        top_two    = Counter(prices).most_common(2)
        top_price, top_count = top_two[0]
        agreement  = top_count / total_seen

        # ── NA-payment split: sign shows two prices but never labels cash/credit ──
        # When a grade has no payment label and the top price doesn't achieve
        # agreement (because two different prices split the votes ~50/50),
        # treat the two dominant values as Cash (lower) and Credit (higher).
        # Only trigger when: (a) exactly top-2 together cover ≥70% of the bucket
        # (genuine two-way split, not scattered noise), and (b) each appeared ≥2×.
        if pay == "NA" and agreement < threshold and len(top_two) == 2:
            c1, c2 = top_two[0][1], top_two[1][1]
            combined_agreement = (c1 + c2) / total_seen
            if combined_agreement >= 0.70 and c1 >= 2 and c2 >= 2:
                p1 = float(top_two[0][0])
                p2 = float(top_two[1][0])
                cash_val   = f"{min(p1, p2):.3f}"
                credit_val = f"{max(p1, p2):.3f}"
                log.debug("  SPLIT %-15s NA → Cash=%s Credit=%s", grade, cash_val, credit_val)
                for split_pay, split_price in (("Cash", cash_val), ("Credit", credit_val)):
                    validated.append({
                        "fuel_type":        grade,
                        "payment_type":     split_pay,
                        "validated_price":  split_price,
                        "occurrence_count": c1 if split_pay == "Cash" else c2,
                        "total_detections": total_seen,
                        "total_api_calls":  total_api_calls,
                        "presence_pct":     round(presence * 100, 1),
                        "agreement_pct":    round(combined_agreement * 100, 1),
                    })
                continue

        if agreement < threshold:
            log.debug("  SKIP %-15s %-8s — agreement %.1f%% < %.0f%%",
                      grade, pay, agreement * 100, threshold * 100)
            continue

        validated.append({
            "fuel_type":        grade,
            "payment_type":     pay,
            "validated_price":  top_price,
            "occurrence_count": top_count,
            "total_detections": total_seen,
            "total_api_calls":  total_api_calls,
            "presence_pct":     round(presence * 100, 1),
            "agreement_pct":    round(agreement * 100, 1),
        })

    # Sort: grade order first, then Cash before Credit
    pay_order = {"Cash": 0, "Credit": 1, "NA": 2}
    validated.sort(key=lambda r: (
        _GRADE_ORDER.index(r["fuel_type"]) if r["fuel_type"] in _GRADE_ORDER else 99,
        pay_order.get(r["payment_type"], 2),
    ))

    return validated, majority_brand


def save_price_csv(validated: list[dict], brand: str | None,
                   csv_path: str) -> None:
    """Save validated prices to CSV."""
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fields = ["fuel_type", "payment_type", "validated_price",
                  "occurrence_count", "total_detections", "total_api_calls",
                  "presence_pct", "agreement_pct"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in validated:
            writer.writerow({k: row.get(k, "NA") for k in fields})
    log.info("CSV saved: %s", csv_path)


def save_summary_image(validated: list[dict], brand: str | None,
                       img_path: str, threshold: float,
                       min_presence: float = 0.05) -> None:
    """
    Generate a slide-ready cash/credit price table summary card.

    Layout:
      ┌─────────────────────────────────────────┐
      │  GAS PRICE AI          GEMINI 3 FLASH  │  ← header
      ├─────────────────────────────────────────┤
      │  STATION   EXXON                        │  ← brand
      ├─────────────────────────────────────────┤
      │  FUEL TYPE        CASH      CREDIT/CARD │  ← column headers
      │  Regular         $2.169      $2.269     │
      │  Diesel          $3.649      $3.709     │
      ├─────────────────────────────────────────┤
      │  Extracted via Gemini Vision OCR  · ... │  ← footer
      └─────────────────────────────────────────┘
    """
    import datetime

    # ── Group validated rows by (grade → {cash, credit}) ─────────────────────
    by_grade: dict[str, dict] = {}
    for row in validated:
        g = row["fuel_type"]
        if g not in by_grade:
            by_grade[g] = {"cash": None, "credit": None, "row": row}
        pay = row.get("payment_type", "NA")
        if pay == "Cash":
            by_grade[g]["cash"] = row["validated_price"]
        elif pay == "Credit":
            by_grade[g]["credit"] = row["validated_price"]
        else:
            # No payment type — fill both if not already set
            if not by_grade[g]["cash"]:
                by_grade[g]["cash"] = row["validated_price"]
            if not by_grade[g]["credit"]:
                by_grade[g]["credit"] = row["validated_price"]

    ordered_grades = ([g for g in _GRADE_ORDER if g in by_grade] +
                      [g for g in by_grade if g not in _GRADE_ORDER])

    # ── Canvas ────────────────────────────────────────────────────────────────
    W        = 820
    PAD      = 32
    HDR_H    = 62
    BRAND_H  = 72
    SUBHDR_H = 36
    ROW_H    = 58
    FOOTER_H = 40
    H = HDR_H + BRAND_H + SUBHDR_H + len(ordered_grades) * ROW_H + FOOTER_H + PAD

    img  = Image.new("RGB", (W, H), (10, 13, 22))
    draw = ImageDraw.Draw(img)

    f_title   = _load_font(19, bold=True)
    f_sub     = _load_font(13)
    f_brand   = _load_font(30, bold=True)
    f_col_hdr = _load_font(13)
    f_fuel    = _load_font(18)
    f_price   = _load_font(24, bold=True)
    f_footer  = _load_font(12)

    C_CASH_COL   = (85,  220, 120)
    C_CREDIT_COL = (85,  180, 255)

    def tw(font, text):
        try:    return int(font.getlength(text))
        except: return len(text) * (font.size // 2)

    # ── Header ────────────────────────────────────────────────────────────────
    draw.rectangle([0, 0, W, HDR_H], fill=(10, 75, 48))
    draw.text((PAD, 20), "GAS PRICE AI", font=f_title, fill=C_ACCENT)
    sub = "POWERED BY GEMINI 3 FLASH"
    draw.text((W - PAD - tw(f_sub, sub), 24), sub, font=f_sub, fill=C_DIM)
    draw.line([(0, HDR_H), (W, HDR_H)], fill=C_ACCENT, width=2)

    cy = HDR_H

    # ── Brand ─────────────────────────────────────────────────────────────────
    draw.rectangle([0, cy, W, cy + BRAND_H], fill=(14, 18, 30))
    draw.text((PAD, cy + 8),  "STATION", font=f_col_hdr, fill=C_DIM)
    draw.text((PAD, cy + 26), (brand or "UNKNOWN").upper(), font=f_brand, fill=C_BRAND)
    cy += BRAND_H
    draw.line([(PAD, cy), (W - PAD, cy)], fill=(*C_ACCENT, 100), width=1)

    # ── Column headers ────────────────────────────────────────────────────────
    COL_FUEL   = PAD
    COL_CASH   = 370
    COL_CREDIT = 590
    draw.rectangle([0, cy, W, cy + SUBHDR_H], fill=(16, 22, 38))
    draw.text((COL_FUEL,   cy + 10), "FUEL TYPE",     font=f_col_hdr, fill=C_DIM)
    draw.text((COL_CASH,   cy + 10), "CASH",          font=f_col_hdr, fill=C_CASH_COL)
    draw.text((COL_CREDIT, cy + 10), "CREDIT / CARD", font=f_col_hdr, fill=C_CREDIT_COL)
    cy += SUBHDR_H

    # ── Price rows ────────────────────────────────────────────────────────────
    for i, grade in enumerate(ordered_grades):
        d   = by_grade[grade]
        bg  = (12, 16, 26) if i % 2 == 0 else (16, 20, 34)
        draw.rectangle([0, cy, W, cy + ROW_H], fill=bg)
        mid = cy + ROW_H // 2

        draw.text((COL_FUEL, mid - 11), grade, font=f_fuel, fill=C_WHITE)

        if d["cash"]:
            draw.text((COL_CASH,   mid - 13), f"${d['cash']}",
                      font=f_price, fill=C_CASH_COL)
        if d["credit"]:
            draw.text((COL_CREDIT, mid - 13), f"${d['credit']}",
                      font=f_price, fill=C_CREDIT_COL)

        draw.line([(PAD, cy + ROW_H - 1), (W - PAD, cy + ROW_H - 1)],
                  fill=(30, 38, 60), width=1)
        cy += ROW_H

    # ── Footer ────────────────────────────────────────────────────────────────
    draw.line([(PAD, cy), (W - PAD, cy)], fill=(*C_ACCENT, 60), width=1)
    cy += 6
    footer = (f"Filters: ≥{int(threshold*100)}% agreement  ·  ≥{int(min_presence*100)}% presence  ·  "
              f"Extracted {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    draw.text((PAD, cy + 8), footer, font=f_footer, fill=C_DIM)

    Path(img_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(img_path)
    log.info("Summary image saved: %s", img_path)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def annotate_video(input_path: str, output_path: str,
                   sample_every_n: int = SAMPLE_EVERY_N,
                   max_frames: int | None = None) -> None:
    """
    Process video frame-by-frame:
      - Every `sample_every_n` frames: call Gemini for detection + OCR
      - All other frames: reuse the last known annotation (smooth & low-cost)
      - max_frames: stop early after N frames (useful for quick tests)
    """
    client = genai.Client()   # reads GEMINI_API_KEY from environment

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    limit  = min(total, max_frames) if max_frames else total

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    log.info("Input : %s  (%dx%d @ %.1f fps, %d frames)", input_path, width, height, fps, total)
    log.info("Output: %s", output_path)
    log.info("Processing %d/%d frames — Gemini every %d frames (~every %.2fs)",
             limit, total, sample_every_n, sample_every_n / fps)

    frame_idx   = 0
    api_calls   = 0
    last_result: dict = {"sign_detected": False, "bounding_box": None,
                         "brand": None, "prices": []}
    price_log: list[dict] = []   # all extracted price entries across API calls
    brand_log: list[str]  = []   # all brand values returned by Gemini

    while frame_idx < limit:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_every_n == 0:
            try:
                t0      = time.time()
                result  = query_gemini(client, frame)
                elapsed = time.time() - t0
                api_calls  += 1
                last_result = result

                # Collect prices & brand for post-run validation
                for p in (result.get("prices") or []):
                    price_log.append(p)
                brand_log.append(result.get("brand") or "")

                log.info("[%4d/%d]  sign=%-5s  brand=%-12s  prices=%d  (%.1fs)",
                         frame_idx, limit,
                         str(result.get("sign_detected")),
                         str(result.get("brand") or "—"),
                         len(result.get("prices") or []),
                         elapsed)
            except Exception as exc:
                log.warning("[%4d] Gemini error: %s", frame_idx, exc)

        annotated = draw_annotations(frame, last_result)
        writer.write(annotated)
        frame_idx += 1

    cap.release()
    writer.release()
    log.info("Done. %d frames written, %d Gemini API calls.", frame_idx, api_calls)
    log.info("Saved: %s", output_path)

    # ── Post-run: validate prices & export summary ────────────────────────────
    validated, majority_brand = validate_prices(price_log, brand_log, threshold=0.60)

    if validated:
        stem     = Path(output_path).stem   # e.g. "out_20260301_220759"
        out_dir  = Path(output_path).parent
        csv_path = str(out_dir / f"{stem}_prices.csv")
        img_path = str(out_dir / f"{stem}_summary.png")

        save_price_csv(validated, majority_brand, csv_path)
        save_summary_image(validated, majority_brand, img_path,
                           threshold=0.60, min_presence=0.05)

        log.info("── Validated Prices (≥60%% agreement, ≥5%% presence) ──")
        for r in validated:
            log.info("  %-18s  $%s   agree=%.1f%%  presence=%.1f%%  (%d/%d)",
                     r["fuel_type"], r["validated_price"],
                     r["agreement_pct"], r["presence_pct"],
                     r["occurrence_count"], r["total_detections"])
    else:
        log.warning("No prices passed the 60%% validation threshold.")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import datetime

    parser = argparse.ArgumentParser(
        description="Annotate a gas station video with Gemini — bounding box + HUD price panel"
    )
    parser.add_argument("--input",  required=True,
                        help="Input video path (e.g. ../../data/videos/gas_station_1.mp4)")
    parser.add_argument("--output", default=None,
                        help="Output base name (default: <stem>_annotated). "
                             "A timestamp is always appended to avoid overwrites.")
    parser.add_argument("--every",  type=int, default=SAMPLE_EVERY_N,
                        help=f"Call Gemini every N frames (default: {SAMPLE_EVERY_N})")
    parser.add_argument("--frames", type=int, default=None,
                        help="Process only the first N frames (for quick tests)")
    args = parser.parse_args()

    stamp    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    in_path  = args.input
    base     = args.output or str(Path(in_path).with_stem(Path(in_path).stem + "_annotated"))
    # Insert timestamp before the extension: out.mp4 → out_20260301_193000.mp4
    base_p   = Path(base).with_suffix("")
    out_path = str(base_p) + f"_{stamp}.mp4"

    annotate_video(in_path, out_path, sample_every_n=args.every, max_frames=args.frames)

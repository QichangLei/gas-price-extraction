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

def validate_prices(price_log: list[dict],
                    brand_log: list[str],
                    threshold: float = 0.60,
                    min_presence: float = 0.05) -> tuple[list[dict], str | None]:
    """
    Majority-vote validation across all Gemini API calls.

    Two-gate filter:
      1. min_presence  — fuel type must appear in >= min_presence fraction of
                         ALL API calls (scales with video length; filters one-offs)
      2. threshold     — of those appearances, the top price must account for
                         >= threshold fraction (filters inconsistent readings)

    Returns (validated_rows, majority_brand).
    """
    total_api_calls = len(brand_log)   # one entry per Gemini call

    # ── Brand majority vote ───────────────────────────────────────────────────
    valid_brands = [b for b in brand_log if b]
    majority_brand = Counter(valid_brands).most_common(1)[0][0] if valid_brands else None

    # ── Per fuel-type price aggregation ──────────────────────────────────────
    by_fuel: dict[str, list[str]] = defaultdict(list)
    for entry in price_log:
        ft = (entry.get("fuel_type") or "").strip()
        pr = str(entry.get("price") or "").strip()
        if ft and pr and pr != "nan":
            by_fuel[ft].append(pr)

    validated = []
    for fuel_type, prices in by_fuel.items():
        total_seen = len(prices)

        # Gate 1: must appear in enough frames relative to total run length
        presence = total_seen / total_api_calls if total_api_calls > 0 else 0
        if presence < min_presence:
            log.debug("  SKIP %-15s — presence %.1f%% < %.0f%% minimum",
                      fuel_type, presence * 100, min_presence * 100)
            continue

        # Gate 2: top price must dominate its own detections
        top_price, top_count = Counter(prices).most_common(1)[0]
        agreement = top_count / total_seen
        if agreement < threshold:
            log.debug("  SKIP %-15s — agreement %.1f%% < %.0f%% threshold",
                      fuel_type, agreement * 100, threshold * 100)
            continue

        validated.append({
            "fuel_type":        fuel_type,
            "validated_price":  top_price,
            "occurrence_count": top_count,
            "total_detections": total_seen,
            "total_api_calls":  total_api_calls,
            "presence_pct":     round(presence * 100, 1),
            "agreement_pct":    round(agreement * 100, 1),
        })

    # Sort by price value ascending (Regular → Mid → Premium)
    try:
        validated.sort(key=lambda r: float(r["validated_price"]))
    except ValueError:
        validated.sort(key=lambda r: r["fuel_type"])

    return validated, majority_brand


def save_price_csv(validated: list[dict], brand: str | None,
                   csv_path: str) -> None:
    """Save validated prices to CSV."""
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fields = ["fuel_type", "validated_price", "occurrence_count",
                  "total_detections", "total_api_calls",
                  "presence_pct", "agreement_pct"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in validated:
            writer.writerow({k: row[k] for k in fields})
    log.info("CSV saved: %s", csv_path)


def save_summary_image(validated: list[dict], brand: str | None,
                       img_path: str, threshold: float,
                       min_presence: float = 0.05) -> None:
    """
    Generate a clean investor-facing summary card image.

    Layout:
      ┌─────────────────────────────────────┐
      │  GAS PRICE AI  ·  GEMINI 3 FLASH   │  ← header
      ├─────────────────────────────────────┤
      │  Station:  EXXON                   │  ← brand
      ├─────────────────────────────────────┤
      │  FUEL TYPE     PRICE    CONFIDENCE │  ← column headers
      │  Regular       $3.459   ████ 94%   │
      │  Premium       $3.859   ████ 91%   │
      ├─────────────────────────────────────┤
      │  Validated at ≥60% agreement · ... │  ← footer
      └─────────────────────────────────────┘
    """
    # ── Canvas ────────────────────────────────────────────────────────────────
    W, H_BASE  = 760, 200
    ROW_H      = 68
    H          = H_BASE + len(validated) * ROW_H
    img        = Image.new("RGB", (W, H), color=(10, 13, 22))
    draw       = ImageDraw.Draw(img)

    # ── Fonts ─────────────────────────────────────────────────────────────────
    f_title   = _load_font(18, bold=True)
    f_sub     = _load_font(13, bold=False)
    f_brand   = _load_font(26, bold=True)
    f_col_hdr = _load_font(12, bold=False)
    f_fuel    = _load_font(18, bold=False)
    f_price   = _load_font(26, bold=True)
    f_pct     = _load_font(15, bold=True)
    f_footer  = _load_font(12, bold=False)

    PAD = 30

    # ── Header bar ────────────────────────────────────────────────────────────
    HDR_H = 54
    draw.rectangle([0, 0, W, HDR_H], fill=(10, 75, 48))
    draw.text((PAD, 16), "GAS PRICE AI", font=f_title, fill=(*C_ACCENT, 255))
    sub = "POWERED BY GEMINI 3 FLASH"
    try:
        sw = f_sub.getlength(sub)
    except AttributeError:
        sw = len(sub) * 7
    draw.text((W - PAD - int(sw), 20), sub, font=f_sub, fill=(*C_DIM,))
    # Thin accent underline
    draw.line([(0, HDR_H), (W, HDR_H)], fill=(*C_ACCENT,), width=2)

    cursor_y = HDR_H

    # ── Brand section ─────────────────────────────────────────────────────────
    BRAND_H = 64
    draw.rectangle([0, cursor_y, W, cursor_y + BRAND_H], fill=(14, 18, 30))
    draw.text((PAD, cursor_y + 10), "STATION", font=f_col_hdr, fill=(*C_DIM,))
    draw.text((PAD, cursor_y + 26), brand.upper() if brand else "UNKNOWN",
              font=f_brand, fill=(*C_BRAND,))
    cursor_y += BRAND_H

    # Separator
    draw.line([(PAD, cursor_y), (W - PAD, cursor_y)],
              fill=(*C_ACCENT, 100), width=1)
    cursor_y += 1

    # ── Column headers ────────────────────────────────────────────────────────
    COL_HDR_H = 28
    draw.rectangle([0, cursor_y, W, cursor_y + COL_HDR_H], fill=(14, 18, 30))
    draw.text((PAD,       cursor_y + 8), "FUEL TYPE",  font=f_col_hdr, fill=(*C_DIM,))
    draw.text((320,       cursor_y + 8), "PRICE",      font=f_col_hdr, fill=(*C_DIM,))
    draw.text((500,       cursor_y + 8), "AGREEMENT",  font=f_col_hdr, fill=(*C_DIM,))
    cursor_y += COL_HDR_H

    # ── Price rows ────────────────────────────────────────────────────────────
    BAR_W_MAX = 180   # max width of the agreement bar
    for i, row in enumerate(validated):
        bg = (12, 16, 26) if i % 2 == 0 else (16, 20, 34)
        draw.rectangle([0, cursor_y, W, cursor_y + ROW_H], fill=bg)

        mid_y = cursor_y + ROW_H // 2

        # Fuel type
        draw.text((PAD, mid_y - 10), row["fuel_type"],
                  font=f_fuel, fill=(*C_WHITE,))

        # Price
        draw.text((320, mid_y - 14), f"${row['validated_price']}",
                  font=f_price, fill=(*C_PRICE,))

        # Agreement bar + percentage
        pct     = row["agreement_pct"] / 100.0
        bar_w   = int(BAR_W_MAX * pct)
        bar_col = C_CONF_H if pct >= 0.85 else (C_CONF_M if pct >= 0.70 else C_CONF_L)
        bar_y0  = mid_y - 7
        bar_y1  = mid_y + 7
        # Track background
        draw.rectangle([500, bar_y0, 500 + BAR_W_MAX, bar_y1], fill=(30, 35, 50))
        # Filled bar
        draw.rectangle([500, bar_y0, 500 + bar_w, bar_y1], fill=(*bar_col,))
        # Percentage label
        draw.text((500 + BAR_W_MAX + 10, mid_y - 9),
                  f"{row['agreement_pct']}%", font=f_pct, fill=(*bar_col,))

        # Count sub-label
        draw.text((PAD, mid_y + 12),
                  f"{row['occurrence_count']} / {row['total_detections']} detections  "
                  f"({row['presence_pct']}% of frames)",
                  font=f_col_hdr, fill=(*C_DIM,))

        cursor_y += ROW_H

    # ── Footer ────────────────────────────────────────────────────────────────
    import datetime
    draw.line([(PAD, cursor_y), (W - PAD, cursor_y)],
              fill=(*C_ACCENT, 60), width=1)
    cursor_y += 4
    footer = (f"Filters: ≥{int(threshold*100)}% price agreement  &  ≥{int(min_presence*100)}% frame presence  ·  "
              f"Generated {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    draw.text((PAD, cursor_y + 6), footer, font=f_footer, fill=(*C_DIM,))

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

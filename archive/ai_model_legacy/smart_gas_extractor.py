"""
Smart Gas Price Extractor
Combines computer vision image analysis with adaptive Claude prompts.
Pipeline:
  1. Analyze image conditions (brightness, time-of-day, blur, contrast)
  2. Route to specialized prompt based on conditions
  3. Parse and validate extracted prices
"""

import anthropic
import base64
import cv2
import numpy as np
import json
import re
import csv
from pathlib import Path
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional
import time


# ─────────────────────────────────────────────
# Image Condition Analysis
# ─────────────────────────────────────────────
#define conditions
class ImageCondition(Enum):
    DAYTIME_CLEAR    = "daytime_clear"
    DAYTIME_GLARE    = "daytime_glare"
    NIGHTTIME_LED    = "nighttime_led"
    NIGHTTIME_DIM    = "nighttime_dim"
    LOW_CONTRAST     = "low_contrast"
    MOTION_BLUR      = "motion_blur"
    DISTANT          = "distant"
    OVEREXPOSED      = "overexposed"


@dataclass
class ImageAnalysis:
    condition: ImageCondition
    brightness: float          # 0–255
    contrast: float            # std dev of pixel values
    blur_score: float          # Laplacian variance (higher = sharper)
    has_led_glow: bool         # bright spots on dark background
    estimated_distance: str    # "close" | "medium" | "far"
    notes: list[str] = field(default_factory=list)


def analyze_image(img_path: str) -> ImageAnalysis:
    """Analyze image conditions using OpenCV."""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # ── Brightness (mean pixel value) ──
    brightness = float(np.mean(gray))

    # ── Contrast (std dev) ──
    contrast = float(np.std(gray))

    # ── Blur (Laplacian variance – lower = blurrier) ──
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # ── LED glow detection (bright spots on dark background) ──
    # Look for isolated bright regions with dark surroundings
    _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    bright_ratio = np.sum(bright_mask > 0) / (h * w)
    has_led_glow = brightness < 80 and bright_ratio > 0.02 and bright_ratio < 0.4

    # ── Estimated sign distance (heuristic based on edge density) ──
    #rough heuristic, not true depth estimation. 
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (h * w)
    if edge_density < 0.03:
        estimated_distance = "far"
    elif edge_density < 0.08:
        estimated_distance = "medium"
    else:
        estimated_distance = "close"

    # ── Glare detection ──
    overexposed_ratio = np.sum(gray > 240) / (h * w)
    has_glare = overexposed_ratio > 0.05

    # ── Classify condition ──
    notes = []
    if brightness > 180:
        condition = ImageCondition.OVEREXPOSED
        notes.append(f"Very bright image ({brightness:.0f})")
    elif has_glare and brightness > 100:
        condition = ImageCondition.DAYTIME_GLARE
        notes.append(f"Glare detected ({overexposed_ratio:.1%} overexposed pixels)")
    elif blur_score < 50:
        condition = ImageCondition.MOTION_BLUR
        notes.append(f"Blurry image (score={blur_score:.1f})")
    elif has_led_glow:
        condition = ImageCondition.NIGHTTIME_LED
        notes.append("LED display pattern detected")
    elif brightness < 60:
        condition = ImageCondition.NIGHTTIME_DIM
        notes.append(f"Low brightness ({brightness:.0f})")
    elif contrast < 30:
        condition = ImageCondition.LOW_CONTRAST
        notes.append(f"Low contrast ({contrast:.1f})")
    elif estimated_distance == "far":
        condition = ImageCondition.DISTANT
        notes.append("Sign appears far away")
    else:
        condition = ImageCondition.DAYTIME_CLEAR
        notes.append("Normal daytime conditions")

    return ImageAnalysis(
        condition=condition,
        brightness=brightness,
        contrast=contrast,
        blur_score=blur_score,
        has_led_glow=has_led_glow,
        estimated_distance=estimated_distance,
        notes=notes
    )


# ─────────────────────────────────────────────
# Adaptive Prompt Library
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise gas price extraction system. 
Extract fuel prices from gas station sign images with high accuracy.
Always respond with valid JSON only — no explanation, no markdown."""

PROMPTS = {
    ImageCondition.DAYTIME_CLEAR: """Extract all visible fuel prices from this gas station sign.

The image is clear and well-lit. Look for:
- Regular/Unleaded, Mid-Grade, Premium, Diesel price displays
- Prices in format like 3.459 or 3.45 9/10 (tenths digit is common)
- Both cash and credit prices if shown separately

Return JSON:
{
  "prices": [
    {"fuel_type": "Regular", "price": "3.459", "price_type": "cash"},
    {"fuel_type": "Premium", "price": "3.759", "price_type": "cash"}
  ],
  "confidence": "high|medium|low",
  "notes": "any observations"
}""",

    ImageCondition.NIGHTTIME_LED: """Extract fuel prices from this NIGHTTIME gas station sign with LED displays.

LED digits on dark backgrounds can cause:
- Digit confusion: 1↔7, 3↔8, 6↔8, 0↔8
- Glow bleed making digits appear connected
- Missing segments looking like different digits

CRITICAL RULES for LED displays:
- Gas prices are always between $2.00 and $7.00 in the US
- The 3rd decimal is almost always 9 (e.g., X.XX9)
- Prices increase from Regular → Mid → Premium by ~$0.30-$0.50 each
- If a digit seems wrong, check if swapping it makes a realistic price

Return JSON:
{
  "prices": [
    {"fuel_type": "Regular", "price": "3.459", "price_type": "cash"},
    {"fuel_type": "Diesel", "price": "3.899", "price_type": "cash"}
  ],
  "confidence": "high|medium|low",
  "notes": "LED-specific observations, any digit ambiguities noted"
}""",

    ImageCondition.NIGHTTIME_DIM: """Extract fuel prices from this LOW-LIGHT nighttime gas station image.

The image is dark and may have limited visibility. Strategies:
- Look for any lit price boards, backlit signs, or illuminated panels
- Focus on regions with any visible text or numbers
- Use price context: Regular < Mid-Grade < Premium, all typically $2-6

Be conservative — only report prices you can see with reasonable confidence.

Return JSON:
{
  "prices": [
    {"fuel_type": "Regular", "price": "3.459", "price_type": "cash"}
  ],
  "confidence": "high|medium|low",
  "notes": "visibility issues encountered"
}""",

    ImageCondition.DAYTIME_GLARE: """Extract fuel prices from this gas station sign with GLARE or harsh lighting.

Glare may wash out some digits. Strategies:
- Look for price digits in shadowed or less-affected areas
- Use partial price info + context to infer obscured digits
- Price validation: Regular < Mid < Premium, spread of ~$0.30 each

Return JSON:
{
  "prices": [
    {"fuel_type": "Regular", "price": "3.459", "price_type": "cash"}
  ],
  "confidence": "high|medium|low",
  "notes": "which areas affected by glare"
}""",

    ImageCondition.MOTION_BLUR: """Extract fuel prices from this BLURRY gas station sign image.

Despite blur, try to:
- Identify price regions by layout (gas signs follow standard formats)
- Use digit shape context even if edges are soft
- Apply price sanity checks aggressively ($2-6 range, 9/10 endings)
- Report partial prices if full prices unreadable

Return JSON:
{
  "prices": [
    {"fuel_type": "Regular", "price": "3.459", "price_type": "cash"}
  ],
  "confidence": "high|medium|low",
  "notes": "blur level impact on specific areas"
}""",

    ImageCondition.DISTANT: """Extract fuel prices from this GAS STATION SIGN photographed from a distance.

The sign may appear small. Focus on:
- The price display board (usually rectangular, centered on sign)
- Large digit clusters — these are the prices
- Use zoom/magnification mentally on the sign area

Price format reminders: X.XXX where last digit is usually 9.

Return JSON:
{
  "prices": [
    {"fuel_type": "Regular", "price": "3.459", "price_type": "cash"}
  ],
  "confidence": "high|medium|low",
  "notes": "estimated sign size/legibility"
}""",

    ImageCondition.LOW_CONTRAST: """Extract fuel prices from this LOW CONTRAST gas station image.

Low contrast makes text harder to distinguish from background. Look for:
- Slight color/brightness differences that outline digits
- Price board area by its rectangular shape
- Any shadows or reflections that reveal digit edges

Return JSON:
{
  "prices": [
    {"fuel_type": "Regular", "price": "3.459", "price_type": "cash"}
  ],
  "confidence": "high|medium|low",
  "notes": "contrast issues and how they affected extraction"
}""",

    ImageCondition.OVEREXPOSED: """Extract fuel prices from this OVEREXPOSED gas station image.

Bright areas may wash out digits. Strategies:
- Look for price digits in less-saturated regions
- Dark digits on bright background: focus on edges of bright zones
- Use surrounding context (fuel type labels) to locate price areas

Return JSON:
{
  "prices": [
    {"fuel_type": "Regular", "price": "3.459", "price_type": "cash"}
  ],
  "confidence": "high|medium|low",
  "notes": "overexposure impact"
}""",
}


# ─────────────────────────────────────────────
# Extraction Pipeline
# ─────────────────────────────────────────────

@dataclass
class ExtractionResult:
    image_path: str
    condition: str
    brightness: float
    blur_score: float
    prices: list[dict]
    confidence: str
    notes: str
    raw_response: str
    error: Optional[str] = None
    processing_time_ms: float = 0.0


def encode_image(img_path: str) -> tuple[str, str]:
    """Base64-encode image for Claude API."""
    with open(img_path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")
    ext = Path(img_path).suffix.lower()
    media_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                 ".png": "image/png", ".webp": "image/webp"}
    media_type = media_map.get(ext, "image/jpeg")
    return data, media_type


def extract_prices(img_path: str, client: anthropic.Anthropic,
                   model: str = "claude-sonnet-4-20250514") -> ExtractionResult:
    """Full pipeline: analyze → select prompt → call Claude → parse."""
    t0 = time.time()

    # Step 1: Analyze image
    try:
        analysis = analyze_image(img_path)
    except Exception as e:
        return ExtractionResult(
            image_path=img_path, condition="unknown", brightness=0,
            blur_score=0, prices=[], confidence="low",
            notes="", raw_response="", error=f"CV analysis failed: {e}"
        )

    # Step 2: Select prompt
    prompt = PROMPTS.get(analysis.condition, PROMPTS[ImageCondition.DAYTIME_CLEAR])

    # Step 3: Call Claude
    try:
        img_data, media_type = encode_image(img_path)
        response = client.messages.create(
            model=model,
            max_tokens=600,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": media_type, "data": img_data
                    }},
                    {"type": "text", "text": prompt}
                ]
            }]
        )
        raw = response.content[0].text
    except Exception as e:
        return ExtractionResult(
            image_path=img_path, condition=analysis.condition.value,
            brightness=analysis.brightness, blur_score=analysis.blur_score,
            prices=[], confidence="low", notes=str(analysis.notes),
            raw_response="", error=f"API call failed: {e}",
            processing_time_ms=(time.time() - t0) * 1000
        )

    # Step 4: Parse JSON response
    prices, confidence, notes = [], "low", ""
    try:
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        parsed = json.loads(clean)
        prices = parsed.get("prices", [])
        confidence = parsed.get("confidence", "low")
        notes = parsed.get("notes", "")
    except json.JSONDecodeError:
        # Fallback: regex extraction
        price_pattern = r'\b[2-6]\.\d{3}\b'
        found = re.findall(price_pattern, raw)
        prices = [{"fuel_type": "unknown", "price": p, "price_type": "unknown"} for p in found]
        notes = "JSON parse failed, used regex fallback"

    elapsed = (time.time() - t0) * 1000
    return ExtractionResult(
        image_path=img_path,
        condition=analysis.condition.value,
        brightness=round(analysis.brightness, 1),
        blur_score=round(analysis.blur_score, 1),
        prices=prices,
        confidence=confidence,
        notes=f"{'; '.join(analysis.notes)} | {notes}".strip(" |"),
        raw_response=raw,
        processing_time_ms=round(elapsed, 1)
    )


# ─────────────────────────────────────────────
# Batch Processing
# ─────────────────────────────────────────────

def process_batch(
    image_dir: str,
    output_dir: str,
    model: str = "claude-sonnet-4-20250514",
    max_images: Optional[int] = None,
    delay_between: float = 0.5,
    extensions: tuple = (".jpg", ".jpeg", ".png", ".webp")
) -> list[ExtractionResult]:
    """Process a directory of images with adaptive prompting."""

    client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    images = [
        p for p in Path(image_dir).iterdir()
        if p.suffix.lower() in extensions
    ]
    if max_images:
        images = images[:max_images]

    print(f"Processing {len(images)} images from {image_dir}")
    print(f"Output → {output_dir}")
    print("-" * 60)

    results = []
    condition_counts: dict[str, int] = {}

    for i, img_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {img_path.name}", end=" ... ", flush=True)
        result = extract_prices(str(img_path), client, model)
        results.append(result)

        cond = result.condition
        condition_counts[cond] = condition_counts.get(cond, 0) + 1

        status = f"✓ {len(result.prices)} prices [{result.confidence}]" if not result.error else f"✗ {result.error}"
        print(f"{cond} → {status} ({result.processing_time_ms:.0f}ms)")

        if delay_between > 0:
            time.sleep(delay_between)

    # ── Save outputs ──
    _save_json(results, out_path / "results.json")
    _save_csv(results, out_path / "results.csv")
    _save_summary(results, condition_counts, out_path / "summary.txt")

    print("\n" + "=" * 60)
    print("CONDITION DISTRIBUTION:")
    for cond, count in sorted(condition_counts.items(), key=lambda x: -x[1]):
        pct = count / len(results) * 100
        print(f"  {cond:<25} {count:>4}  ({pct:.1f}%)")
    print(f"\nResults saved to: {output_dir}")
    return results


def _save_json(results: list[ExtractionResult], path: Path):
    with open(path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)


def _save_csv(results: list[ExtractionResult], path: Path):
    rows = []
    for r in results:
        base = {
            "image_path": r.image_path,
            "condition": r.condition,
            "brightness": r.brightness,
            "blur_score": r.blur_score,
            "confidence": r.confidence,
            "notes": r.notes,
            "error": r.error or "",
            "processing_time_ms": r.processing_time_ms,
        }
        if r.prices:
            for p in r.prices:
                rows.append({**base, **p})
        else:
            rows.append({**base, "fuel_type": "", "price": "", "price_type": ""})

    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _save_summary(results, condition_counts, path: Path):
    total = len(results)
    with_prices = sum(1 for r in results if r.prices)
    errors = sum(1 for r in results if r.error)

    lines = [
        "GAS PRICE EXTRACTION SUMMARY",
        "=" * 50,
        f"Total images:       {total}",
        f"With prices found:  {with_prices} ({with_prices/total*100:.1f}%)",
        f"Errors:             {errors}",
        f"Success rate:       {(total-errors)/total*100:.1f}%",
        "",
        "CONDITION BREAKDOWN:",
    ]
    for cond, count in sorted(condition_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {cond:<30} {count:>4}  ({count/total*100:.1f}%)")

    high = sum(1 for r in results if r.confidence == "high")
    med  = sum(1 for r in results if r.confidence == "medium")
    low  = sum(1 for r in results if r.confidence == "low")
    lines += ["", "CONFIDENCE DISTRIBUTION:",
              f"  High:   {high}", f"  Medium: {med}", f"  Low:    {low}"]

    path.write_text("\n".join(lines))


# ─────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smart Gas Price Extractor")
    parser.add_argument("image_dir",  help="Directory of gas station images")
    parser.add_argument("output_dir", help="Where to save results")
    parser.add_argument("--model",    default="claude-sonnet-4-20250514")
    parser.add_argument("--max",      type=int, default=None, help="Max images to process")
    parser.add_argument("--delay",    type=float, default=0.5, help="Delay between API calls (s)")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Only run CV analysis, no API calls")
    args = parser.parse_args()

    if args.analyze_only:
        # Quick condition survey without API calls
        images = list(Path(args.image_dir).glob("*.jpg")) + list(Path(args.image_dir).glob("*.png"))
        if args.max:
            images = images[:args.max]
        print(f"Analyzing {len(images)} images...\n")
        counts: dict[str, int] = {}
        for img in images:
            try:
                a = analyze_image(str(img))
                counts[a.condition.value] = counts.get(a.condition.value, 0) + 1
                print(f"{img.name:<40} {a.condition.value:<25} brightness={a.brightness:.0f} blur={a.blur_score:.0f}")
            except Exception as e:
                print(f"{img.name:<40} ERROR: {e}")
        print("\nCondition Summary:")
        for c, n in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"  {c:<30} {n}")
    else:
        process_batch(
            image_dir=args.image_dir,
            output_dir=args.output_dir,
            model=args.model,
            max_images=args.max,
            delay_between=args.delay
        )

"""
Smart Gas Price Extractor v2
Uses Lei's strict OCR system prompt + condition-specific injection blocks.
Pipeline:
  1. OpenCV analysis → classify image condition
  2. Inject condition hints into strict OCR prompt
  3. Parse TXT/CSV structured output
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
# Image Condition Analysis (unchanged)
# ─────────────────────────────────────────────

class ImageCondition(Enum):
    DAYTIME_CLEAR  = "daytime_clear"
    DAYTIME_GLARE  = "daytime_glare"
    NIGHTTIME_LED  = "nighttime_led"
    NIGHTTIME_DIM  = "nighttime_dim"
    LOW_CONTRAST   = "low_contrast"
    MOTION_BLUR    = "motion_blur"
    DISTANT        = "distant"
    OVEREXPOSED    = "overexposed"


@dataclass
class ImageAnalysis:
    condition: ImageCondition
    brightness: float
    contrast: float
    blur_score: float
    has_led_glow: bool
    estimated_distance: str
    notes: list[str] = field(default_factory=list)


def analyze_image(img_path: str) -> ImageAnalysis:
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    brightness = float(np.mean(gray))
    contrast   = float(np.std(gray))
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    bright_ratio   = np.sum(bright_mask > 0) / (h * w)
    has_led_glow   = brightness < 80 and 0.02 < bright_ratio < 0.4

    edges        = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (h * w)
    if edge_density < 0.03:
        estimated_distance = "far"
    elif edge_density < 0.08:
        estimated_distance = "medium"
    else:
        estimated_distance = "close"

    overexposed_ratio = np.sum(gray > 240) / (h * w)
    has_glare = overexposed_ratio > 0.05

    notes = []
    if brightness > 180:
        condition = ImageCondition.OVEREXPOSED
        notes.append(f"Very bright ({brightness:.0f})")
    elif has_glare and brightness > 100:
        condition = ImageCondition.DAYTIME_GLARE
        notes.append(f"Glare ({overexposed_ratio:.1%} overexposed)")
    elif blur_score < 50:
        condition = ImageCondition.MOTION_BLUR
        notes.append(f"Blurry (score={blur_score:.1f})")
    elif has_led_glow:
        condition = ImageCondition.NIGHTTIME_LED
        notes.append("LED display detected")
    elif brightness < 60:
        condition = ImageCondition.NIGHTTIME_DIM
        notes.append(f"Dark ({brightness:.0f})")
    elif contrast < 30:
        condition = ImageCondition.LOW_CONTRAST
        notes.append(f"Low contrast ({contrast:.1f})")
    elif estimated_distance == "far":
        condition = ImageCondition.DISTANT
        notes.append("Sign far away")
    else:
        condition = ImageCondition.DAYTIME_CLEAR
        notes.append("Normal daytime")

    return ImageAnalysis(condition, brightness, contrast, blur_score,
                         has_led_glow, estimated_distance, notes)


# ─────────────────────────────────────────────
# Lei's Strict OCR System Prompt
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are a strict visual information extraction system.
Your task is to read text from images and output structured records.
You are NOT allowed to interpret, infer, normalize, or guess missing information.

You must behave like an OCR parser, not a conversational assistant.

Input Description

You will receive 1 gas station sign image labeled: Image 1

The image may contain zero, one, or multiple fuel price entries.

Extraction Targets

For each distinct fuel price entry visible in the image, extract exactly:
- Image label
- Fuel type text appearing next to the price
- Numeric price value
- Gas station brand (logo or text only if visible)
- Payment condition (Cash, Credit, or explicit wording)
- Confidence score (your certainty that the text was read correctly)

Strict Detection Rules

Fuel Type
- Only copy the label physically associated with the price
- Examples: Regular, Unleaded, Midgrade, Plus, Premium, Diesel, etc.
- If unclear or cropped → NA
- Do NOT standardize wording (e.g., "Unl" ≠ "Regular")

Price
- Must be a number formatted like fuel pricing (e.g., 2.999, 3.459, 4.109)
- Ignore road prices, store prices, lottery, or advertisement numbers
- If cents fraction (9/10) format appears → convert to 3 decimal (e.g., 2.99⁹ → 2.999)

Brand
- Only if logo or brand name is directly visible
- Do NOT infer from color scheme or location

Payment Type
- Only if explicitly written (Cash, Credit, Cash Price, Credit Price, etc.)
- Otherwise NA

Confidence
- Represents OCR reading certainty only (not reasoning)
- Integer percentage between 50–99%
- Higher only if text is sharp and unambiguous

Forbidden Behaviors

Do NOT:
- Guess missing labels
- Normalize fuel categories
- Assume brand by color
- Merge prices across rows
- Explain reasoning
- Add commentary
- Reformat output

Output Format (STRICT)

TXT block for every detected entry:

Image: [Image number]
Fuel Type: [exact visible text or NA]
Price: $[price with exactly 3 decimals]
Gas Station Brand: [brand or NA]
Payment Type: [Cash/Credit/NA]
Confidence: [integer%]

(blank line between entries)

CSV section after all TXT blocks:

CSV_START
Image_Number,Fuel_Type,Price,Gas_Station_Brand,Payment_Type,Confidence
[one row per entry]
CSV_END

Missing Data Rule
If any field is uncertain → output NA
Never fabricate information."""


# ─────────────────────────────────────────────
# Condition Injection Blocks
# These append ONLY OCR-level reading hints to the user message.
# They do NOT override the system prompt or add reasoning.
# ─────────────────────────────────────────────

CONDITION_HINTS: dict[ImageCondition, str] = {
    ImageCondition.DAYTIME_CLEAR: "",  # No hint needed

    ImageCondition.NIGHTTIME_LED: """
[OCR READING HINT — LED DISPLAY DETECTED]
This image contains LED digit displays on a dark background.
Known LED OCR failure modes to watch for:
- Segment confusion: 1↔7, 3↔8, 6↔8, 0↔8
- Glow bleed may make digits appear fused
- Missing segments may make a digit resemble another
Do not correct or infer — only report what you can read.
Apply your normal confidence scoring; flag ambiguous digits with lower confidence.""",

    ImageCondition.NIGHTTIME_DIM: """
[OCR READING HINT — LOW LIGHT]
This image is dark. Focus on any illuminated or backlit regions.
Only report prices with characters you can actually discern.
Do not guess digits obscured by darkness.""",

    ImageCondition.DAYTIME_GLARE: """
[OCR READING HINT — GLARE / HARSH LIGHT]
Parts of this image may be washed out by sunlight or reflection.
Only report digits that are clearly readable.
Do not fill in digits hidden by glare.""",

    ImageCondition.MOTION_BLUR: """
[OCR READING HINT — MOTION BLUR]
This image contains blur that may soften digit edges.
Only report characters you can distinguish despite the blur.
Do not reconstruct digits from blurred shapes.""",

    ImageCondition.DISTANT: """
[OCR READING HINT — DISTANT SIGN]
The sign appears small in the frame. Focus on the price display area.
Only report characters large enough to read with confidence.
Do not guess digits that are too small to distinguish.""",

    ImageCondition.LOW_CONTRAST: """
[OCR READING HINT — LOW CONTRAST]
Text and background have low visual separation.
Only report characters where edges are sufficiently visible.
Do not report digits you cannot distinguish from background.""",

    ImageCondition.OVEREXPOSED: """
[OCR READING HINT — OVEREXPOSED]
Bright regions may obscure digit detail.
Focus on less-saturated areas where characters remain visible.
Do not reconstruct digits lost to overexposure.""",
}


# ─────────────────────────────────────────────
# Output Parsers
# ─────────────────────────────────────────────

def parse_txt_block(text: str) -> list[dict]:
    """Parse TXT-format entries from Claude's response."""
    entries = []
    blocks = re.split(r'\n\s*\n', text.strip())
    for block in blocks:
        if "Image:" not in block:
            continue
        entry = {}
        for line in block.strip().splitlines():
            if line.startswith("Image:"):
                entry["image_number"] = line.split(":", 1)[1].strip()
            elif line.startswith("Fuel Type:"):
                entry["fuel_type"] = line.split(":", 1)[1].strip()
            elif line.startswith("Price:"):
                raw_price = line.split(":", 1)[1].strip().lstrip("$")
                entry["price"] = raw_price
            elif line.startswith("Gas Station Brand:"):
                entry["brand"] = line.split(":", 1)[1].strip()
            elif line.startswith("Payment Type:"):
                entry["payment_type"] = line.split(":", 1)[1].strip()
            elif line.startswith("Confidence:"):
                entry["confidence"] = line.split(":", 1)[1].strip().rstrip("%")
        if entry:
            entries.append(entry)
    return entries


def parse_csv_section(text: str) -> list[dict]:
    """Parse CSV_START...CSV_END block from Claude's response."""
    match = re.search(r'CSV_START\n(.+?)\nCSV_END', text, re.DOTALL)
    if not match:
        return []
    lines = match.group(1).strip().splitlines()
    if len(lines) < 2:
        return []
    reader = csv.DictReader(lines)
    return list(reader)


# ─────────────────────────────────────────────
# Extraction Pipeline
# ─────────────────────────────────────────────
#define conditions
@dataclass
class ExtractionResult:
    image_path: str
    condition: str
    brightness: float
    blur_score: float
    entries: list[dict]          # parsed TXT entries
    csv_rows: list[dict]         # parsed CSV rows
    raw_response: str
    error: Optional[str] = None
    processing_time_ms: float = 0.0


def encode_image(img_path: str) -> tuple[str, str]:
    with open(img_path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")
    ext = Path(img_path).suffix.lower()
    media_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                 ".png": "image/png", ".webp": "image/webp"}
    return data, media_map.get(ext, "image/jpeg")


def extract_prices(img_path: str, client: anthropic.Anthropic,
                   model: str = "claude-sonnet-4-20250514") -> ExtractionResult:
    t0 = time.time()

    # Step 1: Analyze image condition
    try:
        analysis = analyze_image(img_path)
    except Exception as e:
        return ExtractionResult(
            image_path=img_path, condition="unknown", brightness=0, blur_score=0,
            entries=[], csv_rows=[], raw_response="", error=f"CV analysis failed: {e}"
        )

    # Step 2: Build user message = base instruction + condition hint
    hint = CONDITION_HINTS.get(analysis.condition, "")
    user_text = "Extract all fuel prices from Image 1."
    if hint:
        user_text += "\n" + hint.strip()

    # Step 3: Call Claude with Lei's strict system prompt
    try:
        img_data, media_type = encode_image(img_path)
        response = client.messages.create(
            model=model,
            max_tokens=800,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": media_type, "data": img_data
                    }},
                    {"type": "text", "text": user_text}
                ]
            }]
        )
        raw = response.content[0].text
    except Exception as e:
        return ExtractionResult(
            image_path=img_path, condition=analysis.condition.value,
            brightness=analysis.brightness, blur_score=analysis.blur_score,
            entries=[], csv_rows=[], raw_response="", error=f"API call failed: {e}",
            processing_time_ms=(time.time() - t0) * 1000
        )

    # Step 4: Parse both output formats
    entries  = parse_txt_block(raw)
    csv_rows = parse_csv_section(raw)

    return ExtractionResult(
        image_path=img_path,
        condition=analysis.condition.value,
        brightness=round(analysis.brightness, 1),
        blur_score=round(analysis.blur_score, 1),
        entries=entries,
        csv_rows=csv_rows,
        raw_response=raw,
        processing_time_ms=round((time.time() - t0) * 1000, 1)
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

    client = anthropic.Anthropic()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    images = [p for p in Path(image_dir).iterdir()
              if p.suffix.lower() in extensions]
    if max_images:
        images = images[:max_images]

    print(f"Processing {len(images)} images  →  {output_dir}")
    print("-" * 70)

    results = []
    condition_counts: dict[str, int] = {}
    all_csv_rows: list[dict] = []

    for i, img_path in enumerate(images, 1):
        print(f"[{i:>4}/{len(images)}] {img_path.name:<40}", end=" ", flush=True)
        result = extract_prices(str(img_path), client, model)
        results.append(result)

        cond = result.condition
        condition_counts[cond] = condition_counts.get(cond, 0) + 1

        if result.error:
            print(f"✗ {result.error}")
        else:
            n = len(result.entries)
            print(f"{cond:<22} → {n} entr{'y' if n==1 else 'ies'}  ({result.processing_time_ms:.0f}ms)")

        # Attach image filename to CSV rows for traceability
        for row in result.csv_rows:
            row["source_file"] = img_path.name
            all_csv_rows.append(row)

        if delay_between > 0 and i < len(images):
            time.sleep(delay_between)

    # ── Save outputs ──
    _save_master_csv(all_csv_rows, out_path / "all_prices.csv")
    _save_raw_responses(results, out_path / "raw_responses.jsonl")
    _save_summary(results, condition_counts, out_path / "summary.txt")

    print("\n" + "=" * 70)
    print("CONDITION DISTRIBUTION:")
    for cond, count in sorted(condition_counts.items(), key=lambda x: -x[1]):
        print(f"  {cond:<28} {count:>4}  ({count/len(results)*100:.1f}%)")
    total_entries = sum(len(r.entries) for r in results)
    print(f"\nTotal price entries extracted: {total_entries}")
    print(f"Output files: {output_dir}/")
    return results


def _save_master_csv(rows: list[dict], path: Path):
    if not rows:
        return
    fieldnames = ["source_file", "Image_Number", "Fuel_Type", "Price",
                  "Gas_Station_Brand", "Payment_Type", "Confidence"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _save_raw_responses(results: list[ExtractionResult], path: Path):
    with open(path, "w") as f:
        for r in results:
            record = {
                "image_path": r.image_path,
                "condition": r.condition,
                "brightness": r.brightness,
                "blur_score": r.blur_score,
                "error": r.error,
                "processing_time_ms": r.processing_time_ms,
                "raw_response": r.raw_response,
                "parsed_entries": r.entries,
            }
            f.write(json.dumps(record) + "\n")


def _save_summary(results, condition_counts, path: Path):
    total = len(results)
    errors = sum(1 for r in results if r.error)
    with_entries = sum(1 for r in results if r.entries)
    total_entries = sum(len(r.entries) for r in results)

    lines = [
        "GAS PRICE EXTRACTION SUMMARY",
        "=" * 50,
        f"Total images:           {total}",
        f"Images with entries:    {with_entries} ({with_entries/total*100:.1f}%)",
        f"Total price entries:    {total_entries}",
        f"Avg entries/image:      {total_entries/max(with_entries,1):.2f}",
        f"Errors:                 {errors}",
        "",
        "CONDITION BREAKDOWN:",
    ]
    for cond, count in sorted(condition_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {cond:<32} {count:>4}  ({count/total*100:.1f}%)")
    path.write_text("\n".join(lines))


# ─────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smart Gas Price Extractor v2")
    parser.add_argument("image_dir",  help="Directory of gas station images")
    parser.add_argument("output_dir", help="Where to save results")
    parser.add_argument("--model",  default="claude-sonnet-4-20250514")
    parser.add_argument("--max",    type=int, default=None, help="Max images")
    parser.add_argument("--delay",  type=float, default=0.5, help="Delay between calls (s)")
    parser.add_argument("--analyze-only", action="store_true",
                        help="CV condition survey only, no API calls")
    args = parser.parse_args()

    if args.analyze_only:
        images = []
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            images += list(Path(args.image_dir).glob(f"*{ext}"))
        if args.max:
            images = images[:args.max]
        print(f"Analyzing {len(images)} images (no API calls)...\n")
        counts: dict[str, int] = {}
        for img in images:
            try:
                a = analyze_image(str(img))
                counts[a.condition.value] = counts.get(a.condition.value, 0) + 1
                hint_flag = "★" if CONDITION_HINTS.get(a.condition) else " "
                print(f"{hint_flag} {img.name:<40} {a.condition.value:<25} "
                      f"bright={a.brightness:.0f} blur={a.blur_score:.0f}")
            except Exception as e:
                print(f"  {img.name:<40} ERROR: {e}")
        print("\nSummary:")
        for c, n in sorted(counts.items(), key=lambda x: -x[1]):
            has_hint = "★" if CONDITION_HINTS.get(ImageCondition(c)) else " "
            print(f"  {has_hint} {c:<32} {n}")
        print("\n★ = condition hint will be injected into prompt")
    else:
        process_batch(
            image_dir=args.image_dir,
            output_dir=args.output_dir,
            model=args.model,
            max_images=args.max,
            delay_between=args.delay
        )

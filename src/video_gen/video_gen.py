"""
video_gen.py
─────────────
Generate synthetic gas station dashcam videos via Google Veo and automatically
write ground truth labels to data/videos/labels/ground_truth.csv.

Because the prices and brand are specified here as parameters, no manual
labeling is needed — the ground truth is known before the video is generated.

Usage
-----
    python src/video_gen/video_gen.py

    # Custom station:
    python src/video_gen/video_gen.py \
        --brand Chevron \
        --grades "Regular:3.199:Cash,Regular:3.299:Credit,Premium:3.799:Cash,Diesel:4.099:NA" \
        --time day

    # Dry run (print prompt + GT without generating):
    python src/video_gen/video_gen.py --dry-run
"""

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parents[2] / ".env", override=True)

from google import genai
from google.genai import types
from PIL import Image as PILImage
import io

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT   = Path(__file__).parents[2]
GT_CSV         = PROJECT_ROOT / "data/videos/labels/ground_truth.csv"
VIDEO_DIR      = PROJECT_ROOT / "data/videos/Gas Videos"
REFERENCE_DIR  = PROJECT_ROOT / "data/AI_reference_images"


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_price_sign_text(grades: list[dict]) -> str:
    """
    Build the sign text description from a list of grade dicts.
    e.g. [{"grade": "Regular", "price": 3.95}, ...]
    → "'REGULAR  3.95', 'DIESEL  4.59'"
    """
    rows = []
    for g in grades:
        label = g["grade"].upper()
        price = f"{g['price']:.2f}"
        rows.append(f"'{label}  {price}'")
    return ", ".join(rows)


def build_prompt(brand: str, grades: list[dict], time_of_day: str = "day",
                 has_ref_image: bool = False) -> str:
    lighting = (
        "natural daylight, clear sky"
        if time_of_day == "day"
        else "dusk or night, station lights illuminated, neon glow on the sign"
    )
    sign_text = build_price_sign_text(grades)
    n_rows    = len(grades)

    ref_style_note = (
        "The visual style, color scheme, font, and physical design of the price sign "
        "should match the reference image provided. "
        "CRITICAL: do NOT copy any numbers or text from the reference image — "
        "the sign must display exactly and only the prices listed below. "
    ) if has_ref_image else ""

    return (
        f"A continuous exterior dashcam shot filmed from a camera mounted on the front of "
        f"a semi-truck driving along a typical US suburban road at moderate speed (around 25–35 mph). "
        f"There is exactly one {brand} gas station on the right side of the road — no other stations. "
        f"As the truck approaches and passes the station, the single {brand} price sign is clearly "
        f"visible and facing the camera directly for several seconds — long enough to read every digit. "
        f"{ref_style_note}"
        f"The sign is a tall pole-mounted roadside price sign with a solid dark background and large "
        f"white replaceable plastic digit panels, exactly as seen at real US gas stations. "
        f"The sign has exactly {n_rows} rows. Each row shows the fuel grade label on the left and "
        f"the price on the right in large, clearly legible digits. "
        f"The rows show exactly: {sign_text}. "
        f"The sign is well-lit, unobstructed, and the digits fill a large portion of the sign face — "
        f"every letter and number must be sharp and perfectly readable in the video. "
        f"The prices are displayed in simple X.XX dollar format with no extra symbols. "
        f"The {brand} logo or brand name is clearly visible on the station canopy or sign. "
        f"Photorealistic dashcam footage, {lighting}, ambient road noise only, "
        f"no narration, no voices, no slow motion, no motion blur on the sign."
    )


# ── Ground truth writer ───────────────────────────────────────────────────────

def write_ground_truth(video_filename: str, brand: str, grades: list[dict]) -> None:
    """Append rows to ground_truth.csv for the generated video."""
    GT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Write header if file doesn't exist
    write_header = not GT_CSV.exists()
    with open(GT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video_file", "brand", "grade", "price", "payment"])
        if write_header:
            writer.writeheader()
        for g in grades:
            writer.writerow({
                "video_file": video_filename,
                "brand":      brand,
                "grade":      g["grade"],
                "price":      g["price"],
                "payment":    g["payment"] if g["payment"] != "NA" else "",
            })
    print(f"  Ground truth written → {GT_CSV}  ({len(grades)} rows)")


# ── Reference image loader ────────────────────────────────────────────────────

def load_reference_image(image_path: Path) -> types.Image:
    """Load an image file and return a google.genai types.Image object."""
    img = PILImage.open(image_path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return types.Image(image_bytes=buf.getvalue(), mime_type="image/jpeg")


# ── Image-to-video animation ──────────────────────────────────────────────────

def _make_approach_frame(ref_path: Path) -> types.Image:
    """
    Create a synthetic 'far away' first frame: the reference image shrunk to ~15%
    and placed in the right-center of a road background, heavily blurred —
    simulating the gas station seen from a distance.
    """
    from PIL import ImageFilter, ImageDraw

    orig = PILImage.open(ref_path).convert("RGB")
    W, H = 1280, 720

    # Road background: grey asphalt + pale sky
    bg = PILImage.new("RGB", (W, H), (120, 120, 120))
    draw = ImageDraw.Draw(bg)
    draw.rectangle([0, 0, W, int(H * 0.45)], fill=(180, 200, 220))   # sky
    draw.rectangle([0, int(H * 0.45), W, H], fill=(90, 90, 90))      # road

    # Paste a tiny, blurred version of the station on the right horizon
    thumb_w = int(W * 0.12)
    thumb_h = int(thumb_w * orig.height / orig.width)
    thumb   = orig.resize((thumb_w, thumb_h), PILImage.LANCZOS)
    thumb   = thumb.filter(ImageFilter.GaussianBlur(radius=3))

    paste_x = int(W * 0.70)
    paste_y = int(H * 0.40) - thumb_h // 2
    bg.paste(thumb, (paste_x, paste_y))

    buf = io.BytesIO()
    bg.save(buf, format="JPEG", quality=90)
    return types.Image(image_bytes=buf.getvalue(), mime_type="image/jpeg")


def animate_from_image(image_path: Path, as_last_frame: bool = False) -> None:
    """
    Expand a static gas station image into a short video using Veo image-to-video.

    as_last_frame=False → image is the first frame (truck passing by).
    as_last_frame=True  → image is the last frame; a synthetic approach shot is
                          used as the first frame so Veo interpolates the truck
                          driving toward and stopping at the station.
    """
    print("=" * 60)
    print("  Veo Image-to-Video (animate)")
    print("=" * 60)
    print(f"  Image      : {image_path}")
    print(f"  Frame mode : {'last frame (interpolation)' if as_last_frame else 'first frame'}")

    if as_last_frame:
        prompt = (
            "A realistic dashcam video filmed from a semi-truck driving along a US suburban road. "
            "The video starts with the gas station visible in the distance on the right side. "
            "The truck approaches at moderate speed (25–35 mph) and the station grows larger. "
            "The video ends with the price sign clearly filling the frame — sharp, fully readable, "
            "exactly as shown in the final reference frame. "
            "Smooth forward motion, slight camera vibration from the road. "
            "No camera cuts, no zooming, no slow motion."
        )
    else:
        prompt = (
            "Animate this gas station scene into a natural, realistic dashcam video. "
            "The image is the opening frame. The semi-truck is driving past the gas station "
            "at moderate speed (25–35 mph) and the scene plays forward from this point. "
            "The price sign is clearly readable at the start and remains visible as long as possible. "
            "Subtle natural motion only: slight camera vibration from the road, "
            "gentle movement of flags or trees if present, natural lighting variation. "
            "Do not blur, obscure, or alter any digits or text on the sign. "
            "No camera cuts, no zooming, no slow motion."
        )
    print(f"\nPrompt:\n{prompt}\n")

    last_image  = load_reference_image(image_path)
    client      = genai.Client()

    if as_last_frame:
        first_image = _make_approach_frame(image_path)
        operation   = client.models.generate_videos(
            model="veo-3.1-generate-preview",
            prompt=prompt,
            image=first_image,
            config=types.GenerateVideosConfig(
                last_frame=last_image,
            ),
        )
    else:
        operation = client.models.generate_videos(
            model="veo-3.1-generate-preview",
            image=last_image,
            prompt=prompt,
        )

    print("Generating video (this takes ~2–3 minutes)...")
    while not operation.done:
        print("  Waiting...")
        time.sleep(10)
        operation = client.operations.get(operation)

    if operation.response is None:
        print(f"\n[ERROR] Operation completed but response is None.")
        print(f"  Operation metadata: {operation.metadata}")
        print(f"  Operation error   : {getattr(operation, 'error', 'N/A')}")
        return

    if not operation.response.generated_videos:
        print(f"\n[ERROR] No videos in response: {operation.response}")
        return

    timestamp      = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem           = image_path.stem.replace(" ", "_").replace("(", "").replace(")", "")
    frame_tag      = "_lastframe" if as_last_frame else "_firstframe"
    video_filename = f"anim_{stem}{frame_tag}_{timestamp}.mp4"
    output_path    = VIDEO_DIR / video_filename

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    generated_video = operation.response.generated_videos[0]
    client.files.download(file=generated_video.video)
    generated_video.video.save(str(output_path))
    print(f"\nVideo saved → {output_path}")
    print("  Note: fill ground_truth.csv manually — prices come from the image.")


# ── Video generation ──────────────────────────────────────────────────────────

def generate_video(brand: str, grades: list[dict], time_of_day: str = "day",
                   ref_image_path: Path | None = None) -> None:
    prompt = build_prompt(brand, grades, time_of_day, has_ref_image=ref_image_path is not None)

    print("=" * 60)
    print("  Veo Video Generator")
    print("=" * 60)
    print(f"  Brand     : {brand}")
    for g in grades:
        print(f"  Grade     : {g['grade']:<12}  ${g['price']:.3f}  {g['payment']}")
    print(f"  Time      : {time_of_day}")
    if ref_image_path:
        print(f"  Reference : {ref_image_path.name}")
    print(f"\nPrompt:\n{prompt}\n")

    client = genai.Client()

    if ref_image_path:
        ref_img = load_reference_image(ref_image_path)
        reference = types.VideoGenerationReferenceImage(
            image=ref_img,
            reference_type="asset",
        )
        operation = client.models.generate_videos(
            model="veo-3.1-generate-preview",
            prompt=prompt,
            config=types.GenerateVideosConfig(
                reference_images=[reference],
            ),
        )
    else:
        operation = client.models.generate_videos(
            model="veo-3.1-generate-preview",
            prompt=prompt,
        )

    print("Generating video (this takes ~2–3 minutes)...")
    while not operation.done:
        print("  Waiting...")
        time.sleep(10)
        operation = client.operations.get(operation)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ref_tag   = f"_ref_{ref_image_path.stem}" if ref_image_path else ""
    # Sanitize stem (remove spaces/parens for clean filenames)
    ref_tag   = ref_tag.replace(" ", "_").replace("(", "").replace(")", "")
    video_filename = f"gen_{brand.lower()}_{time_of_day}{ref_tag}_{timestamp}.mp4"
    output_path    = VIDEO_DIR / video_filename

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    generated_video = operation.response.generated_videos[0]
    client.files.download(file=generated_video.video)
    generated_video.video.save(str(output_path))
    print(f"\nVideo saved → {output_path}")

    write_ground_truth(video_filename, brand, grades)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_grades(grades_str: str) -> list[dict]:
    """
    Parse grade string: "Regular:3.199:Cash,Regular:3.299:Credit,Diesel:4.099:NA"
    → list of dicts with grade, price, payment keys.
    """
    result = []
    for item in grades_str.split(","):
        parts = item.strip().split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid grade format '{item}' — expected Grade:Price:Payment")
        result.append({
            "grade":   parts[0].strip(),
            "price":   float(parts[1].strip()),
            "payment": parts[2].strip(),
        })
    return result


def random_grades() -> list[dict]:
    """
    Generate a randomised but realistic pair of US gas prices (Regular + Diesel).
    Prices are rounded to 2 decimal places (e.g. 3.95, 4.59).
    """
    import random

    def snap(price: float) -> float:
        """Round to nearest cent: e.g. 3.9512 → 3.95"""
        return round(price, 2)

    regular = snap(random.uniform(3.49, 4.49))
    diesel  = snap(random.uniform(4.00, 5.00))

    return [
        {"grade": "Regular", "price": regular, "payment": "NA"},
        {"grade": "Diesel",  "price": diesel,  "payment": "NA"},
    ]

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic gas station video with known GT.")
    parser.add_argument("--brand",   default="Shell",
                        help="Gas station brand (default: Shell)")
    parser.add_argument("--grades",  default=None,
                        help="Grades as 'Grade:Price:Payment,...' (default: randomly generated)")
    parser.add_argument("--time",    default="day", choices=["day", "night"],
                        help="Time of day (default: day)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print prompt and GT rows without generating the video.")
    parser.add_argument("--ref-image", metavar="PATH", default=None,
                        help="Path to a reference image for sign style (uses Veo reference images). "
                             "If set to 'auto', picks a random image from data/AI_reference_images/.")
    parser.add_argument("--list-refs", action="store_true",
                        help="List available reference images and exit.")
    parser.add_argument("--animate", metavar="PATH", default=None,
                        help="Animate a static image into a video (image-to-video). "
                             "Pass a filename from data/AI_reference_images/ or a full path. "
                             "Keeps all sign content from the image unchanged.")
    parser.add_argument("--last-frame", action="store_true",
                        help="Use the --animate image as the last frame (truck approaching). "
                             "Default is first frame (truck driving away).")
    args = parser.parse_args()

    if args.list_refs:
        refs = sorted(REFERENCE_DIR.glob("*"))
        print(f"Reference images in {REFERENCE_DIR}:")
        for r in refs:
            print(f"  {r.name}")
        return

    if args.animate:
        anim_path = Path(args.animate)
        if not anim_path.exists():
            anim_path = REFERENCE_DIR / args.animate
        if not anim_path.exists():
            print(f"Image not found: {args.animate}")
            return
        animate_from_image(anim_path, as_last_frame=args.last_frame)
        return

    grades = parse_grades(args.grades) if args.grades else random_grades()

    # Resolve reference image path
    ref_image_path = None
    if args.ref_image:
        if args.ref_image == "auto":
            import random
            candidates = sorted(REFERENCE_DIR.glob("*.png")) + sorted(REFERENCE_DIR.glob("*.jpg"))
            if not candidates:
                print(f"No reference images found in {REFERENCE_DIR}")
            else:
                ref_image_path = random.choice(candidates)
                print(f"Auto-selected reference image: {ref_image_path.name}")
        else:
            ref_image_path = Path(args.ref_image)
            if not ref_image_path.exists():
                # Try relative to REFERENCE_DIR
                ref_image_path = REFERENCE_DIR / args.ref_image
            if not ref_image_path.exists():
                print(f"Reference image not found: {args.ref_image}")
                return

    if args.dry_run:
        prompt = build_prompt(args.brand, grades, args.time, has_ref_image=ref_image_path is not None)
        print("=== PROMPT ===")
        print(prompt)
        if ref_image_path:
            print(f"\n=== REFERENCE IMAGE ===\n{ref_image_path}")
        print("\n=== GROUND TRUTH ROWS ===")
        print("video_file,brand,grade,price,payment")
        for g in grades:
            print(f"gen_{args.brand.lower()}_{args.time}_<datetime>.mp4,"
                  f"{args.brand},{g['grade']},{g['price']},{g['payment']}")
        return

    generate_video(args.brand, grades, args.time, ref_image_path=ref_image_path)


if __name__ == "__main__":
    main()

"""
extractor.py
────────────
Sends preprocessed images to Gemini and parses the CSV response.

Uses the updated `google-genai` SDK (google.genai.Client) with
model: gemini-3-flash-preview

Image passing strategy (per official docs):
  • Inline via types.Part.from_bytes()  → for batches whose total JPEG
    payload stays under ~18 MB (leaves headroom for prompt tokens).
  • File API via client.files.upload()  → automatic fallback for larger
    batches; uploaded files can also be reused across requests.
"""

from __future__ import annotations
import csv
import io
import logging
import time

from google import genai
from google.genai import types

from .config import GEMINI_API_KEY, GEMINI_MODEL, OCR_PROMPT
from .preprocessor import ProcessedImage

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

BATCH_SIZE          = 15          # images sent per API call
INLINE_SIZE_LIMIT   = 18 * 1024 * 1024   # 18 MB — stay under the 20 MB cap


# ── Gemini client (singleton per process) ─────────────────────────────────────

def _get_client() -> genai.Client:
    return genai.Client(api_key=GEMINI_API_KEY)


# ── CSV response parser ───────────────────────────────────────────────────────

EXPECTED_HEADER = ["Image_Number", "Fuel_Type", "Price",
                   "Gas_Station_Brand", "Payment_Type", "Confidence"]


def _parse_csv_response(raw_text: str) -> list[dict]:
    """
    Parse the model's CSV output into a list of dicts.
    Handles markdown fences (```csv … ```) if the model adds them.
    """
    text = raw_text.strip()
    if text.startswith("```"):
        # Drop opening and closing fence lines
        text = "\n".join(
            line for line in text.splitlines()
            if not line.startswith("```")
        ).strip()

    rows   = []
    reader = csv.DictReader(io.StringIO(text))

    if reader.fieldnames != EXPECTED_HEADER:
        log.warning("Unexpected CSV header from model: %s", reader.fieldnames)

    for row in reader:
        cleaned = {k: (v.strip() if v else "NA") for k, v in row.items()}
        # Normalise Price to exactly 3 decimal places; reject non-numeric
        try:
            cleaned["Price"] = f"{float(cleaned['Price']):.3f}"
        except (ValueError, KeyError):
            cleaned["Price"] = "NA"
        rows.append(cleaned)

    return rows


# ── Image → bytes helper ──────────────────────────────────────────────────────

def _pil_to_jpeg_bytes(proc: ProcessedImage) -> bytes:
    """Encode a ProcessedImage's PIL image to JPEG bytes."""
    buf = io.BytesIO()
    proc.pil_image.convert("RGB").save(buf, format="JPEG", quality=92)
    return buf.getvalue()


# ── Contents builder ──────────────────────────────────────────────────────────

def _build_contents_inline(batch: list[ProcessedImage]) -> list:
    """
    Build contents using types.Part.from_bytes() (inline).
    Per docs: ideal for total request size < 20 MB.

    Layout:
      "---BEGIN IMAGES---"
      "[Image N]"  →  types.Part.from_bytes(jpeg_bytes, mime_type='image/jpeg')
      …
    """
    parts: list = ["---BEGIN IMAGES---\n"]
    for proc in batch:
        jpeg_bytes = _pil_to_jpeg_bytes(proc)
        parts.append(f"\n[Image {proc.image_number} | {proc.source_path.name}]\n")
        parts.append(
            types.Part.from_bytes(
                data      = jpeg_bytes,
                mime_type = "image/jpeg",
            )
        )
    return parts


def _build_contents_file_api(client: genai.Client,
                              batch: list[ProcessedImage]) -> list:
    """
    Upload images via the File API and reference them in contents.
    Use when total JPEG payload exceeds the 20 MB inline limit, or when
    the same images may be reused across multiple requests.
    """
    parts: list = ["---BEGIN IMAGES---\n"]
    for proc in batch:
        jpeg_bytes = _pil_to_jpeg_bytes(proc)
        buf        = io.BytesIO(jpeg_bytes)
        buf.name   = f"image_{proc.image_number}.jpg"   # SDK uses .name as filename

        log.debug("Uploading Image %d via File API …", proc.image_number)
        uploaded = client.files.upload(
            file      = buf,
            config    = types.UploadFileConfig(mime_type="image/jpeg"),
        )
        parts.append(f"\n[Image {proc.image_number}]\n")
        parts.append(uploaded)   # File reference accepted directly in contents

    return parts


def _batch_jpeg_size(batch: list[ProcessedImage]) -> int:
    """Estimate total JPEG payload size for a batch (bytes)."""
    total = 0
    for proc in batch:
        buf = io.BytesIO()
        proc.pil_image.convert("RGB").save(buf, format="JPEG", quality=92)
        total += buf.tell()
    return total


# ── Public API ────────────────────────────────────────────────────────────────

def extract_prices(images: list[ProcessedImage],
                   retries: int = 3,
                   retry_delay: float = 5.0) -> list[dict]:
    """
    Send images to Gemini in batches and return a flat list of price records.

    Automatically chooses inline vs File API upload per batch based on size:
      - total JPEG bytes < INLINE_SIZE_LIMIT  →  types.Part.from_bytes()
      - otherwise                             →  client.files.upload()

    Config:
      - system_instruction = OCR_PROMPT  (cleanly separated from image content)
      - thinking_budget = 0              (disables thinking for structured OCR —
                                          per docs: "better results in object detection")
      - temperature = 1.0                (recommended default for Gemini 3 models)
    """
    client   = _get_client()
    all_rows: list[dict] = []

    config = types.GenerateContentConfig(
        system_instruction = OCR_PROMPT,
        thinking_config    = types.ThinkingConfig(thinking_budget=0),
        temperature        = 1.0,
    )

    batches = [images[i:i + BATCH_SIZE]
               for i in range(0, len(images), BATCH_SIZE)]

    for batch_idx, batch in enumerate(batches):
        estimated_mb = _batch_jpeg_size(batch) / (1024 * 1024)
        use_file_api = estimated_mb * 1024 * 1024 > INLINE_SIZE_LIMIT

        log.info(
            "Batch %d/%d — %d image(s), ~%.1f MB — using %s",
            batch_idx + 1, len(batches), len(batch), estimated_mb,
            "File API" if use_file_api else "inline bytes",
        )

        if use_file_api:
            contents = _build_contents_file_api(client, batch)
        else:
            contents = _build_contents_inline(batch)

        for attempt in range(1, retries + 1):
            try:
                response = client.models.generate_content(
                    model    = GEMINI_MODEL,
                    contents = contents,
                    config   = config,
                )
                rows = _parse_csv_response(response.text)
                all_rows.extend(rows)
                log.info("  → %d price row(s) extracted", len(rows))
                break
            except Exception as exc:
                log.warning("Attempt %d/%d failed: %s", attempt, retries, exc)
                if attempt < retries:
                    time.sleep(retry_delay)
                else:
                    log.error("Batch %d permanently failed after %d attempts.",
                              batch_idx + 1, retries)

    return all_rows
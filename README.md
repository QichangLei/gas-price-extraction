# Gas Price Extraction Pipeline

Automated gas price extraction from images and dashcam video using Google Gemini Vision API.
Extracts structured fuel price records, geo-tags them, and outputs map-ready CSVs.

---

## Project Structure

```
Gas_price/
├── pipeline_csv.py          ← Main CLI entry point (run from project root)
├── draft_goals.txt          ← Project goals outline
│
├── src/                     ← All active source code
│   ├── pipeline/            ← Core pipeline modules
│   │   ├── config.py        ← API key, model, output settings, OCR prompt
│   │   ├── preprocessor.py  ← Image loading, resize, EXIF GPS, video→frames
│   │   ├── extractor.py     ← Gemini API batching (15 imgs/call), CSV parser
│   │   ├── geo_merger.py    ← Join prices + GPS → final CSV
│   │   └── key              ← API key file (local only, not committed)
│   ├── annotation/
│   │   └── annotate_video.py ← Frame-by-frame HUD annotation of gas station video
│   ├── evaluation/
│   │   └── evaluate_gas_prices.py ← Precision/Recall/F1/MAE vs GPA4MME-v2 ground truth
│   └── frame_selection/     ← GPS proximity filter (Task 3 — in development)
│
├── data/
│   ├── images/
│   │   ├── GPA4MME-v2/      ← Public benchmark dataset (ground truth labels)
│   │   ├── data_1/          ← Test images 0–20
│   │   ├── data_2/          ← Test images 21–40
│   │   ├── bright/          ← Bright lighting test set
│   │   ├── dark/            ← Night/dark test set
│   │   └── internet_images/ ← External sourced images
│   └── videos/              ← Raw dashcam / gas station video footage
│
├── prompts/                 ← All Gemini prompt files (consolidated)
│
├── output/
│   ├── pipeline/            ← geo CSV outputs from pipeline_csv.py
│   ├── annotation/          ← Annotated videos, price CSVs, summary PNGs
│   └── evaluation/          ← Evaluation reports (precision/recall/F1/MAE)
│
└── archive/                 ← Legacy code (not active, kept for reference)
    ├── root_scripts/        ← Early prototypes (main.py, ex_*.py, v2_*.py, YOLO)
    ├── ai_model_legacy/     ← Claude-based extractors, old results
    ├── debug_imgs/          ← Debug visualization images
    └── old_results/         ← Legacy extraction results
```

---

## Setup

```bash
conda activate gas_price_cv
export GEMINI_API_KEY="your-key-here"
```

---

## Usage

**Run pipeline on a folder of images (from project root):**
```bash
python pipeline_csv.py --input data/images/data_1
```

**Run pipeline with GPS sidecar CSV:**
```bash
python pipeline_csv.py --input data/images/data_1 --geo-sidecar gps_log.csv
```

**Run pipeline on a video:**
```bash
python pipeline_csv.py --video data/videos/gas_station_1.mp4 --frames-dir frames/
```

**Annotate a video with HUD overlay:**
```bash
python src/annotation/annotate_video.py --input data/videos/gas_station_1.mp4 --output out.mp4 --every 15
```

**Evaluate against ground truth:**
```bash
python src/evaluation/evaluate_gas_prices.py \
    --extracted output/pipeline/gas_prices_geo_YYYYMMDD_HHMMSS.csv \
    --gt_dir data/images/GPA4MME-v2/price-level-ann/labels/ \
    --tolerance 0.01 \
    --output output/evaluation/evaluation_report.csv
```

---

## Pipeline Architecture

```
Input (images or video)
    ↓
[src/pipeline/preprocessor.py]  — EXIF GPS extraction, resize, video→frames
    ↓ list[ProcessedImage]
[src/pipeline/extractor.py]     — Gemini multimodal OCR (15 images/call)
    ↓ list[dict]
[src/pipeline/geo_merger.py]    — Join prices + GPS coordinates
    ↓
output/pipeline/gas_prices_geo_YYYYMMDD_HHMMSS.csv
```

**Key settings** (`src/pipeline/config.py`):
- Model: `gemini-3-flash-preview`
- Batch size: 15 images/API call (auto File API fallback at 18 MB)
- Output: `output/pipeline/`

---

## Module Reference

### Module Responsibilities

#### `pipeline_csv.py` — Entry Point
Runs the full pipeline from the command line. Takes `--input` (image folder) or `--video` (dashcam footage), with optional `--geo-sidecar` for external GPS data and `--min-confidence` to drop low-quality reads. Calls the four pipeline modules in order.

---

#### `src/pipeline/config.py` — Central Configuration
All tunable parameters live here — edit this file to change behaviour without touching pipeline logic:
- `GEMINI_MODEL` — which Gemini model to call
- `OCR_PROMPT` — the system instruction sent to Gemini (see below)
- `MAX_IMAGE_LONG_SIDE = 1920` — images are resized before sending (reduces API cost and payload size)
- `BATCH_SIZE = 15` — images per API call

The `OCR_PROMPT` tells Gemini to read the sign literally — no guessing, no inference. Key rules:
- Prices in `9/10` fraction notation (e.g. `2.99⁹`) must be converted to `2.999`
- Unknown fields → `NA`, never fabricated
- Output must be CSV only, no commentary

---

#### `src/pipeline/preprocessor.py` — Image Loading & GPS Extraction
**`ProcessedImage`** is the core data container: holds the image number, file path, PIL image, and a `GeoPoint` (lat/lon/timestamp).

**`load_images_from_folder()`** scans the folder for supported image types (natural sort order so `2.png` comes before `10.png`), pulls GPS from EXIF metadata (DMS → decimal degrees), and optionally replaces EXIF GPS with a sidecar CSV if one is provided.

**`extract_frames_from_video()`** uses OpenCV to pull 1 frame/second from dashcam footage, saving them as JPEGs for the rest of the pipeline.

---

#### `src/pipeline/extractor.py` — Gemini OCR
**`extract_prices()`** is the core function:
1. Splits images into batches of 15
2. For each batch, estimates total JPEG payload size:
   - Under 18 MB → sends images **inline** as bytes (`types.Part.from_bytes`)
   - Over 18 MB → uploads via **Gemini File API** (avoids the 20 MB hard limit)
3. Calls `client.models.generate_content()` with `thinking_budget=0` (disables chain-of-thought, which gives cleaner structured output) and retries up to 3× on failure
4. Parses the CSV text response back into a list of dicts

**`_parse_csv_response()`** handles the model sometimes wrapping output in markdown code fences (` ```csv ... ``` `), strips them, then normalizes `Price` to exactly 3 decimal places.

---

#### `src/pipeline/geo_merger.py` — Join Prices with GPS & Export
**`merge_and_export()`** joins Gemini's extracted price rows back to their source images to attach GPS data.

The tricky part: Gemini's `Image_Number` field can come back in inconsistent formats (`"3"`, `"21.png"`, `"Image 3"`, `"3 | 21.png"`). `_resolve_image()` handles this with a 5-tier fallback:
1. Direct integer index
2. Exact filename/stem match
3. Filename embedded in a longer string (regex extract)
4. First integer found in any string
5. Give up and log a warning

Output CSVs are timestamped (e.g. `gas_prices_geo_20260312_181901.csv`) so every run produces a new file without overwriting previous results.

---

### Key Design Decisions

| Decision | Reason |
|---|---|
| Gemini multimodal OCR instead of traditional CV | Gas price signs vary wildly in font, layout, and lighting; an LLM handles this more robustly than a fixed detector |
| `thinking_budget=0` | Structured extraction needs direct output, not reasoning chains |
| Batch size 15 / inline vs File API split | Balances API latency vs. the 20 MB per-request size cap |
| Natural sort order for images | Prevents `10.png` from being processed before `2.png` |
| Sidecar CSV for GPS | Cameras often lack GPS; allows data from an external GPS logger to be attached |
| Timestamped output | Prevents accidental overwrites across runs |

---

### How to Extend This Code

- **Add a new pipeline stage** (e.g. frame selection, deduplication): create a new module in `src/pipeline/`, import and call it in `pipeline_csv.py`
- **Change the OCR model or prompt**: edit `config.py` only — `extractor.py` reads from config
- **Add a new output format** (e.g. GeoJSON, database insert): add a function to `geo_merger.py` alongside `merge_and_export()`
- **Attach GPS from a different source** (e.g. NMEA log): implement a new loader like `load_geo_sidecar()` in `preprocessor.py` and pass the result as `geo_override`

---

## Output Schema

| Field | Description |
|---|---|
| Image_Number | Sequential index within batch |
| Fuel_Type | Exact label from sign (e.g. "Regular", "Diesel") |
| Price | 3-decimal format (e.g. 3.459); ⁹⁄₁₀ notation handled |
| Gas_Station_Brand | Brand/logo text if visible |
| Payment_Type | Cash / Credit / NA |
| Confidence | OCR clarity score (integer, >90 = clear) |
| Latitude | Decimal degrees from EXIF or sidecar CSV |
| Longitude | Decimal degrees from EXIF or sidecar CSV |
| Captured_At | ISO-8601 timestamp |
| Source_File | Original image filename |

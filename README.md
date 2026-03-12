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

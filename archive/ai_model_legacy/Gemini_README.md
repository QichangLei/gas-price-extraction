Run:
export GEMINI_API_KEY="AIzaSyAU11CURuR2_5fiQkPa_LqI7AH42QfGQ4k"
(gas_price_cv) qichang@qichang-Legion-Y9000P-IRX9:/media/qichang/Data/research/Gas_price/AI_model/Gemini$ python run_pipeline.py --input ./data_1

evaluation: ( does not function correctly for name of images)

python evaluate_gas_prices.py   --extracted  data_1_correct-_20260224_235309.csv   --gt_dir     /media/qichang/Data/research/Gas_price/data/GPA4MME-v2/test-images/labels   --tolerance  0.01   --output     evaluation_report.csv




# Gas Price Extraction Pipeline

A Python pipeline that extracts fuel prices from dashcam/truck images using
Google Gemini's vision API, merges them with GPS coordinates, and exports a
map-ready CSV.

---

## Architecture

```
Input images / video
        │
        ▼
┌─────────────────────┐
│   preprocessor.py   │  resize, EXIF GPS extraction, video→frames
└────────┬────────────┘
         │  list[ProcessedImage]
         ▼
┌─────────────────────┐
│    extractor.py     │  Gemini 1.5 Pro multimodal OCR → raw CSV rows
└────────┬────────────┘
         │  list[dict]
         ▼
┌─────────────────────┐
│    geo_merger.py    │  join prices + GPS → final CSV
└────────┬────────────┘
         │
         ▼
   gas_prices_geo.csv   ← map-ready output
```

---

## Output CSV Schema

| Column | Description |
|---|---|
| `Image_Number` | Sequential index of the source image |
| `Fuel_Type` | Exact label from the sign (e.g. "Regular", "Diesel", "E15") |
| `Price` | Price to 3 decimal places (e.g. `3.459`) |
| `Gas_Station_Brand` | Brand name if visible, else `NA` |
| `Payment_Type` | `Cash` / `Credit` / `NA` |
| `Confidence` | OCR clarity score (50–99) |
| `Latitude` | Decimal degrees (from EXIF or sidecar) |
| `Longitude` | Decimal degrees (from EXIF or sidecar) |
| `Captured_At` | Timestamp from EXIF or sidecar |
| `Source_File` | Original filename |

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your Gemini API key
```bash
export GEMINI_API_KEY="your-key-here"
```
Or edit `pipeline/config.py`.

### 3. Run on a folder of images
```bash
python run_pipeline.py --input ./images
```

### 4. Run on a video (extracts 1 frame/sec by default)
```bash
python run_pipeline.py --video dashcam.mp4
```

---

## GPS Options

### Option A – EXIF (automatic)
If your images have GPS metadata embedded (most modern dashcams/phones do),
no extra steps needed. The pipeline reads coordinates automatically.

### Option B – Sidecar CSV
If GPS is logged separately (e.g. from a separate GPS unit):

```
filename,latitude,longitude,captured_at
img_001,29.7604,-95.3698,2024-03-15T10:23:00
img_002,29.7521,-95.3802,2024-03-15T10:31:45
```

```bash
python run_pipeline.py --input ./images --geo-sidecar gps_log.csv
```

Copy `gps_log_template.csv` as a starting point.

---

## CLI Options

```
--input FOLDER          Folder of images (mutually exclusive with --video)
--video FILE            Video to extract frames from
--frames-dir FOLDER     Where to save video frames  (default: ./frames)
--geo-sidecar CSV       GPS sidecar CSV file
--output CSV            Output file path  (default: output/gas_prices_geo.csv)
--min-confidence INT    Drop rows below this score  (default: 0 = keep all)
```

---

## Using the Output for Map Display

The final CSV's `Latitude`, `Longitude`, `Gas_Station_Brand`, and `Price`
columns map directly to popular tools:

- **Kepler.gl** – drag-and-drop the CSV
- **Google Maps API** – iterate rows, place markers at (Lat, Lon)
- **Folium (Python)** – `folium.Marker([lat, lon], popup=brand + price)`
- **Mapbox / Deck.gl** – GeoJSON conversion from the CSV

---

## Future: Video Pipeline

For long dashcam footage, the `extract_frames_from_video()` function uses
OpenCV to sample 1 frame/second (configurable via `VIDEO_FPS_EXTRACT` in
`config.py`). After frame extraction, the same image pipeline runs.

Install the optional dependency:
```bash
pip install opencv-python
```

---

## Tuning the Prompt

Edit `OCR_PROMPT` in `pipeline/config.py`.  The current prompt:
- Accepts any number of images (not limited to 15)
- Outputs **CSV only** (no TXT block)
- Uses calibrated confidence scores
- Handles 9/10 fraction prices (2.99⁹ → 2.999)

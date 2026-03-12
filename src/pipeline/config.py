"""
Configuration for the Gas Price Extraction Pipeline.
Edit these values to match your setup.
"""

import os

# ── Gemini API ──────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
GEMINI_MODEL   = "gemini-3-flash-preview"          # supports multi-image input

# ── Image pre-processing ────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
MAX_IMAGE_LONG_SIDE  = 1920                 # resize if larger (preserves AR)
JPEG_QUALITY         = 92                   # for resized saves

# ── Video frame extraction (future use) ─────────────────────────────────────
VIDEO_FPS_EXTRACT    = 1                    # 1 frame per second
VIDEO_EXTENSIONS     = {".mp4", ".mov", ".avi", ".mkv"}

# ── Geocoding / geo-tagging ─────────────────────────────────────────────────
# Images must contain GPS EXIF data  OR  you supply a sidecar CSV:
#   filename,latitude,longitude,captured_at
# Set to None to skip geo-tagging (columns will be NA).
GEO_SIDECAR_CSV      = None                 # e.g. "gps_log.csv"

# ── Output ──────────────────────────────────────────────────────────────────
OUTPUT_DIR           = "output/pipeline"
FINAL_CSV            = "gas_prices_geo.csv" # merged prices + coordinates

# ── Gemini prompt ────────────────────────────────────────────────────────────
OCR_PROMPT = """
ROLE
You are an OCR extraction engine. You read visible text and output structured
records only. You must not infer, interpret, normalize, or guess.

INPUT
You will receive one or more images, each labeled with its Image_Number.
Process each image independently. Each image may contain zero or more fuel
price entries.

WHAT TO EXTRACT (per price entry)
1. Image_Number
2. Fuel_Type  – exact visible label (e.g. "Regular", "Diesel", "E15")
3. Price      – gasoline price in 3-decimal format
4. Gas_Station_Brand – visible logo or brand text only
5. Payment_Type      – only if explicitly written (Cash / Credit / NA)
6. Confidence        – OCR clarity score (integer)

FIELD RULES

Fuel_Type
  Copy exactly what appears beside the price.
  If unreadable or cropped → NA
  Do not standardize wording.

Price
  Must look like a gasoline price. Valid examples: 2.999  3.459  4.109
  If shown as 9/10 fraction → convert to 3 decimals  (e.g. 2.99⁹ → 2.999)
  Ignore unrelated numbers (ads, store prices, street signs).

Gas_Station_Brand
  Only if a logo or brand name is clearly visible. Otherwise → NA
  Never infer from colors or visual style.

Payment_Type
  Only if explicitly written (Cash, Credit, Cash Price, Credit Price).
  Otherwise → NA

Confidence  (integer, no % symbol)
  > 90  = highly clear image
  80–90 = slightly blurred
  70–80 = hard to read
  < 70  = barely recognisable

FORBIDDEN ACTIONS
Do not: guess missing fields / standardize fuel names / merge rows /
explain reasoning / add commentary / change output structure.

OUTPUT FORMAT – CSV only, no other text before or after.

Image_Number,Fuel_Type,Price,Gas_Station_Brand,Payment_Type,Confidence
(one data row per price entry; use NA for any unknown field)

EXAMPLES
Image_Number,Fuel_Type,Price,Gas_Station_Brand,Payment_Type,Confidence
1,Regular (Cash),4.390,Lukoil,Cash,94
1,Regular (Credit),4.470,Lukoil,Credit,95
1,Diesel,5.690,Lukoil,NA,95
2,Unleaded,1.790,Murphy USA,NA,95
2,Diesel,2.040,Murphy USA,NA,95
2,E15,1.760,Murphy USA,NA,85

MISSING DATA RULE
If uncertain → NA   Never fabricate values.
"""

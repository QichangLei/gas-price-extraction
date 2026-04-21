"""
Configuration for the Gas Price Extraction Pipeline.
Edit these values to match your setup.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (two levels up from this file: src/pipeline/ → root)
load_dotenv(Path(__file__).parents[2] / ".env", override=True)

# ── Gemini API ──────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL   = "gemini-3-flash-preview"          # supports multi-image input

# ── Image pre-processing ────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
MAX_IMAGE_LONG_SIDE  = 1920                 # resize if larger (preserves AR)
JPEG_QUALITY         = 92                   # for resized saves

# ── Video frame extraction (future use) ─────────────────────────────────────
VIDEO_FPS_EXTRACT    = 5                    # frames per second to extract from video
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

  DIGIT CAUTION — gas price signs use 7-segment LED displays. These digits
  have very specific shapes; misreading one digit changes the price by $0.60+.
  Examine each digit independently before writing the price.

  Critical confusion pairs on LED signs:
  • 7 vs 1 : THIS IS THE MOST COMMON ERROR.
              "7" has a HORIZONTAL BAR across the top AND a diagonal/angled
              stem going down-right. On many US gas signs "7" also has a
              short middle crossbar (resembles the digit ⌐ or a reverse-L).
              "1" is a plain narrow vertical stroke with NO top bar.
              A digit with ANY horizontal stroke at the top is "7", not "1".
              Example: if you see 3.7XX do NOT write 3.1XX.
  • 9 vs 0 : "9" has a closed loop on top with a tail descending below;
              "0" is a plain oval with no tail.
  • 8 vs 0 : "8" has two stacked loops; "0" has one.
  • 6 vs 0 : "6" has an open top; "0" is fully closed.

  If a digit is ambiguous, lower the Confidence score (< 80) rather than
  substituting a different digit.

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

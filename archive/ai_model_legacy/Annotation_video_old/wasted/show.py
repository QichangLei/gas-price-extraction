
import cv2
import google as genai
import csv, io, re
from PIL import Image
import numpy as np

genai.configure(api_key="YOUR_API_KEY")
model = genai.GenerativeModel("gemini-1.5-flash")
ANNOTATION_PROMPT = """
ROLE
You are a visual detection and OCR engine for gas price signs in video frames.
Your output drives downstream annotation code — it must be machine-parseable CSV only.

INPUT
A single video frame image.

TASK
1. Detect the gas price sign (or display board) in the frame.
2. Extract the bounding box of the entire price sign area.
3. Extract each fuel price entry visible on the sign.

OUTPUT FORMAT — CSV only, no text before or after, no markdown, no explanation.

Two section types, clearly labeled:

## BOUNDING_BOX
label,x_min,y_min,x_max,y_max
gas_price_sign,<int>,<int>,<int>,<int>

## PRICES
Fuel_Type,Price,Gas_Station_Brand,Payment_Type,Confidence
<one row per visible price entry>

BOUNDING BOX RULES
- (x_min, y_min) = top-left corner of the sign, in pixels
- (x_max, y_max) = bottom-right corner of the sign, in pixels
- If no gas price sign is detected → output: gas_price_sign,NA,NA,NA,NA

PRICE EXTRACTION RULES (same as your existing OCR prompt)
- Price format: 3 decimals (e.g. 3.459). Convert 9/10 fractions (e.g. 2.99⁹ → 2.999).
- Fuel_Type: copy exactly. If unreadable → NA
- Gas_Station_Brand: only if clearly visible logo or text. Otherwise → NA
- Payment_Type: only if explicitly written (Cash/Credit). Otherwise → NA
- Confidence: integer (>90 clear, 80-90 slight blur, 70-80 hard, <70 barely readable)
- If no prices visible → output one row with all fields NA

FORBIDDEN
Do not guess. Do not explain. Do not add commentary. CSV only.

EXAMPLE OUTPUT
## BOUNDING_BOX
label,x_min,y_min,x_max,y_max
gas_price_sign,142,87,610,420

## PRICES
Fuel_Type,Price,Gas_Station_Brand,Payment_Type,Confidence
Regular,3.459,Shell,NA,92
Plus,3.659,Shell,NA,91
Premium,3.859,Shell,NA,90
"""
def parse_annotation_response(response_text):
    """Parse Gemini's dual-section CSV response."""
    bbox = None
    prices = []

    # Split on section headers
    bbox_match = re.search(r'## BOUNDING_BOX\n(.*?)(?=## PRICES|\Z)', response_text, re.DOTALL)
    price_match = re.search(r'## PRICES\n(.*?)$', response_text, re.DOTALL)

    if bbox_match:
        reader = csv.DictReader(io.StringIO(bbox_match.group(1).strip()))
        for row in reader:
            if row.get('x_min') != 'NA':
                bbox = (int(row['x_min']), int(row['y_min']),
                        int(row['x_max']), int(row['y_max']))

    if price_match:
        reader = csv.DictReader(io.StringIO(price_match.group(1).strip()))
        prices = [r for r in reader if r.get('Price') != 'NA']

    return bbox, prices

def annotate_frame(frame, bbox, prices):
    """Draw bounding box and price overlay on frame."""
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Draw bounding box
    if bbox:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, "Gas Price Sign", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Draw extracted prices at bottom-left
    if prices:
        overlay_lines = ["Extracted Prices:"]
        for p in prices:
            line = f"  {p.get('Fuel_Type','?')}: ${p.get('Price','?')}"
            if p.get('Payment_Type') not in ('NA', '', None):
                line += f" ({p['Payment_Type']})"
            overlay_lines.append(line)

        y_start = h - 20 - (len(overlay_lines) * 22)
        # Semi-transparent background
        bg_w = 260
        bg_h = len(overlay_lines) * 22 + 10
        sub = annotated[y_start - 5 : y_start + bg_h, 10 : 10 + bg_w]
        black_rect = np.zeros_like(sub)
        cv2.addWeighted(black_rect, 0.5, sub, 0.5, 0, sub)
        annotated[y_start - 5 : y_start + bg_h, 10 : 10 + bg_w] = sub

        for i, line in enumerate(overlay_lines):
            cv2.putText(annotated, line, (12, y_start + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

    return annotated

def process_video(input_path, output_path, sample_every_n=5):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    frame_idx = 0
    last_bbox, last_prices = None, []  # reuse last known annotation

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only call Gemini every N frames to save cost/time
        if frame_idx % sample_every_n == 0:
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            try:
                response = model.generate_content([ANNOTATION_PROMPT, pil_img])
                last_bbox, last_prices = parse_annotation_response(response.text)
            except Exception as e:
                print(f"Frame {frame_idx} error: {e}")

        annotated = annotate_frame(frame, last_bbox, last_prices)
        out.write(annotated)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Annotated video saved to {output_path}")

# Run it
process_video("gas_station_1.mp4", "annotated_output.mp4", sample_every_n=5)
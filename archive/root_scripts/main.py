import re
import cv2
#from matplotlib.pyplot import gray
import numpy as np
import pytesseract
def find_price_roi(img, gray):
    H, W = gray.shape[:2]

    # Otsu binarize (no inversion yet)
    _, bw0 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw1 = cv2.bitwise_not(bw0)

    # Try multiple morphology strengths (yours is often too strong)
    kernels = [
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5)),
    ]

    best = None  # (score, (x,y,w,h), bw_used)

    for bw in (bw0, bw1):
        for k in kernels:
            bw2 = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=1)

            contours, _ = cv2.findContours(bw2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                area = w * h
                if area < 0.002 * (W * H):   # raise this: ignore tiny junk
                    continue

                aspect = w / float(h + 1e-6)

                # For price bands, you usually want moderately wide rectangles
                if aspect < 2.0 or aspect > 12.0:
                    continue

                # Prefer boxes in lower-middle area (often where prices sit)
                cx, cy = x + w/2, y + h/2
                pos_score = 1.0 - abs(cy - (0.65 * H)) / H  # peak near 65% height
                size_score = area / (W * H)
                
                best = find_price_roi(img, gray)
                if best is None:
                    return {"ok": False, "error": "No price block detected."}

                score, (x, y, cw, ch), bw2_used = best

                score = 2.0 * size_score + 0.5 * pos_score
                if (best is None) or (score > best[0]):
                    best = (score, (x, y, w, h), bw2)

    return best  # or None

def extract_price(image_path: str) -> dict:
    img = cv2.imread(image_path)
    if img is None:
        return {"ok": False, "error": "Could not read image."}

    h, w = img.shape[:2]

    # 1) Preprocess
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Contrast boost helps LED digits and weathered signs
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu threshold (invert so digits become white blobs if background is bright)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Force digits to be white
    #bw = cv2.bitwise_not(bw)
    # If digits are dark on bright background, invert; if already good, keep.
    # Heuristic: if most pixels are white, invert.
    # if np.mean(bw) > 127:
    #     bw = 255 - bw
    # bw is 0/255 after threshold
    white = np.sum(bw == 255)
    black = np.sum(bw == 0)
    if white > black:
        print("convertd")
        bw = cv2.bitwise_not(bw)

    # Merge digit strokes into blocks
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 6))
    bw2 = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 2) Find candidate price block by contours
    contours, _ = cv2.findContours(bw2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imwrite("debug_gray.png", gray)
    cv2.imwrite("debug_bw.png", bw)
    cv2.imwrite("debug_bw2.png", bw2)

    dbg = img.copy()
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        cv2.rectangle(dbg, (x, y), (x+cw, y+ch), (0, 255, 0), 2)
    cv2.imwrite("debug_contours.png", dbg)

    candidates = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        if area < 0.0005 * (w * h):   #the actual area is small
            continue

        aspect = cw / float(ch + 1e-6)

        # Price blocks tend to be wider than tall; tune as needed
        if aspect < 2.0 or aspect > 15.0:
            continue

        # Bias toward central region (since you said main position)
        cx, cy = x + cw / 2.0, y + ch / 2.0
        center_dist = ((cx - w/2)**2 + (cy - h/2)**2) ** 0.5
        center_score = 1.0 - (center_dist / ((w**2 + h**2) ** 0.5))

        # Score: prefer large, central-ish boxes
        score = (area / (w * h)) * 2.0 + center_score
        candidates.append((score, (x, y, cw, ch)))

    if not candidates:
        return {"ok": False, "error": "No price block detected. Try adjusting thresholds/kernels."}

    # candidates.sort(reverse=True, key=lambda t: t[0])
    x, y, cw, ch = candidates[0][1]
    price_boxes = [box for _, box in candidates]

    # Expand ROI slightly for safety
    pad_x = int(0.02 * w)
    pad_y = int(0.02 * h)
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(w, x + cw + pad_x)
    y1 = min(h, y + ch + pad_y)

    roi = img[y0:y1, x0:x1]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 3) OCR ROI (digits only)
    # # Binarize ROI again to help OCR
    # roi_gray = cv2.GaussianBlur(roi_gray, (3, 3), 0)
    # _, roi_bw = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # # roi_bw is 0/255 after threshold
    # white = np.sum(roi_bw == 255)
    # black = np.sum(roi_bw == 0)
    # # If background is black-dominant, invert so background becomes white
    # if white > black:
    #     print("Inverted ROI")
    #     roi_bw = cv2.bitwise_not(roi_bw)


    # config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.'
    # text = pytesseract.image_to_string(roi_bw, config=config).strip()
    def ocr_price_try_both(roi_bw):
        config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.'
        candidates = [
            roi_bw,
            cv2.bitwise_not(roi_bw),
        ]

        for img in candidates:
            text = pytesseract.image_to_string(img, config=config).strip()

            # Try decimal format first
            m = re.search(r'(\d+\.\d+)', text)
            if m:
                return float(m.group(1)), text

            # Fallback: digits only
            digits = re.sub(r'[^0-9]', '', text)
            if len(digits) >= 3:
                if len(digits) == 3:
                    return float(digits[0] + "." + digits[1:3]), text
                else:
                    return float(digits[0] + "." + digits[1:4]), text

        return None, None
    # 3) OCR ROI (digits only)
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_gray = cv2.GaussianBlur(roi_gray, (3, 3), 0)
    _, roi_bw = cv2.threshold(
        roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    price, raw_text = ocr_price_try_both(roi_bw)

    if price is None:
        return {
            "ok": False,
            "error": "OCR failed",
            "raw_text": raw_text,
            "bbox": [x0, y0, x1, y1],
        }

    return {
        "ok": True,
        "price": price,
        "raw_text": raw_text,
        "bbox": [x0, y0, x1, y1],
    }

    # Extract a price-like token
    # Accept formats like 3.459, 3.45, 345.9 (rare), etc.
    m = re.search(r'(\d+\.\d+)', text)
    if not m:
        # Sometimes decimal is missed; try fallback: digits only and infer
        digits = re.sub(r'[^0-9]', '', text)
        if len(digits) >= 3:
            # Common in US fuel signage: e.g., 3459 -> 3.459 or 345 -> 3.45
            if len(digits) == 3:
                price = float(digits[0] + "." + digits[1:3])
            else:
                price = float(digits[0] + "." + digits[1:4])
            return {"ok": True, "price": price, "raw_text": text, "bbox": [x0, y0, x1, y1]}
        return {"ok": False, "error": f"OCR did not yield a recognizable price. raw='{text}'", "bbox": [x0, y0, x1, y1]}

    price = float(m.group(1))
    return {"ok": True, "price": price, "raw_text": text, "bbox": [x0, y0, x1, y1]}


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python extract_price.py <image_path>")
        raise SystemExit(2)

    result = extract_price(sys.argv[1])
    print(result)


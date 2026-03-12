#!/usr/bin/env python3
import re
import cv2
import numpy as np
import pytesseract


def ocr_price_try_both_polarities(roi_bw: np.ndarray):
    """
    Try OCR on roi_bw and its inverse. Return (price_float, raw_text, used_inverted_bool)
    or (None, best_text, None) if parsing fails.
    """
    config = r"--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789."

    best_text = ""
    for idx, cand in enumerate((roi_bw, cv2.bitwise_not(roi_bw))):
        text = pytesseract.image_to_string(cand, config=config).strip()
        if len(text) > len(best_text):
            best_text = text

        # Decimal format
        m = re.search(r"(\d+\.\d+)", text)
        if m:
            try:
                return float(m.group(1)), text, (idx == 1)
            except ValueError:
                pass

        # Digits-only fallback (infer decimal)
        digits = re.sub(r"[^0-9]", "", text)
        if len(digits) >= 3:
            try:
                if len(digits) == 3:      # e.g., 345 -> 3.45
                    price = float(digits[0] + "." + digits[1:3])
                else:                     # e.g., 3459 -> 3.459
                    price = float(digits[0] + "." + digits[1:4])
                return price, text, (idx == 1)
            except ValueError:
                pass

    return None, best_text, None


def score_price_box(x, y, w, h, W, H):
    """
    Score a contour box as a price candidate.
    Price regions are usually a moderately wide horizontal band.
    """
    area = w * h
    aspect = w / float(h + 1e-6)

    # Hard filters (tune as needed)
    if area < 0.002 * (W * H):   # ignore tiny stuff
        return None
    if aspect < 2.0 or aspect > 12.0:
        return None

    # Prefer middle-to-lower area (typical signage layout)
    cx, cy = x + w / 2.0, y + h / 2.0
    # peak around 65% height; clamp to [0,1]
    pos_score = 1.0 - min(1.0, abs(cy - 0.65 * H) / H)

    size_score = area / (W * H)

    # Weighted score
    return 2.0 * size_score + 0.6 * pos_score


def find_best_price_roi(img_bgr, gray, debug_prefix=None):
    """
    Robust ROI detection:
    - Run Otsu once, then try BOTH polarities (bw and inverted)
    - Try multiple morphology kernels (avoid over-merging)
    - Score and return best (x, y, w, h) and best processed mask used for contours
    """
    H, W = gray.shape[:2]

    # Otsu binarize (no inversion assumption)
    _, bw0 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw1 = cv2.bitwise_not(bw0)

    # Multiple kernels: smaller first to avoid "fills the world"
    kernels = [
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (19, 6)),  # your original, kept as last try
    ]

    best = None  # (score, (x,y,w,h), bw2_used)

    for polarity_id, bw in enumerate((bw0, bw1)):
        for k_id, k in enumerate(kernels):
            bw2 = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=1)

            contours, _ = cv2.findContours(bw2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                sc = score_price_box(x, y, w, h, W, H)
                if sc is None:
                    continue
                if (best is None) or (sc > best[0]):
                    best = (sc, (x, y, w, h), bw2)

            # Optional debug dumps for each variant (can be a lot)
            if debug_prefix is not None:
                cv2.imwrite(f"{debug_prefix}_bw_p{polarity_id}.png", bw)
                cv2.imwrite(f"{debug_prefix}_bw2_p{polarity_id}_k{k_id}.png", bw2)

    return best  # or None


def extract_price(image_path: str, debug: bool = True) -> dict:
    img = cv2.imread(image_path)
    if img is None:
        return {"ok": False, "error": "Could not read image."}

    H, W = img.shape[:2]

    # 1) Preprocess
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE improves contrast for washed-out/reflective signage
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Mild blur (too much blur breaks thin light digits)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    if debug:
        cv2.imwrite("debug_gray.png", gray)

    # 2) Detect candidate ROI robustly
    best = find_best_price_roi(img, gray, debug_prefix="debug" if debug else None)
    if best is None:
        return {"ok": False, "error": "No price block detected. Try adjusting morphology/filters."}

    best_score, (x, y, cw, ch), bw2_used = best

    # Draw best bbox + all contours bbox for inspection
    if debug:
        dbg = img.copy()
        # Draw best in red
        cv2.rectangle(dbg, (x, y), (x + cw, y + ch), (0, 0, 255), 3)
        cv2.imwrite("debug_best_bbox.png", dbg)

        cv2.imwrite("debug_bw2_used.png", bw2_used)

    # Expand ROI slightly
    pad_x = int(0.02 * W)
    pad_y = int(0.02 * H)
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(W, x + cw + pad_x)
    y1 = min(H, y + ch + pad_y)

    roi = img[y0:y1, x0:x1]
    if roi.size == 0:
        return {"ok": False, "error": "Empty ROI after cropping.", "bbox": [x0, y0, x1, y1]}

    # 3) OCR ROI
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_gray = cv2.GaussianBlur(roi_gray, (3, 3), 0)
    _, roi_bw = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Optional: light morphology to connect broken strokes (keep small!)
    k_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    roi_bw = cv2.morphologyEx(roi_bw, cv2.MORPH_OPEN, k_small, iterations=1)

    if debug:
        cv2.imwrite("debug_roi.png", roi)
        cv2.imwrite("debug_roi_bw.png", roi_bw)

    price, raw_text, used_inverted = ocr_price_try_both_polarities(roi_bw)

    if price is None:
        return {
            "ok": False,
            "error": f"OCR did not yield a recognizable price. raw='{raw_text}'",
            "bbox": [x0, y0, x1, y1],
            "roi_ocr_inverted": used_inverted,
            "roi_detect_score": float(best_score),
        }

    return {
        "ok": True,
        "price": price,
        "raw_text": raw_text,
        "bbox": [x0, y0, x1, y1],
        "roi_ocr_inverted": used_inverted,
        "roi_detect_score": float(best_score),
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python extract_price.py <image_path>")
        raise SystemExit(2)

    result = extract_price(sys.argv[1], debug=True)
    print(result)


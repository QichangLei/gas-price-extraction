import re
import cv2
import numpy as np
import pytesseract

def preprocess_for_ocr(roi):
    """Enhances the ROI specifically for Tesseract."""
    # Resize to at least 300 DPI equivalent (standard for OCR)
    h, w = roi.shape[:2]
    roi = cv2.resize(roi, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Use adaptive thresholding for uneven lighting
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return thresh

def ocr_price_logic(roi_bw):
    """Attempts OCR on normal and inverted images."""
    # PSM 7 is for a single text line; PSM 6 for a block. 
    # Try 7 first as prices are usually one line.
    config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.'
    
    for img in [roi_bw, cv2.bitwise_not(roi_bw)]:
        text = pytesseract.image_to_string(img, config=config).strip()
        
        # Clean up text: find first occurrence of a number pattern
        match = re.search(r'(\d+[\.\,]\d+)', text)
        if match:
            clean_val = match.group(1).replace(',', '.')
            return float(clean_val), text
            
        # Fallback: Just digits
        digits = re.sub(r'[^0-9]', '', text)
        if len(digits) >= 3:
            # Assumes format X.YZ or X.YZA
            val = float(digits[0] + "." + digits[1:])
            return val, text
            
    return None, ""

def extract_price(image_path: str) -> dict:
    img = cv2.imread(image_path)
    if img is None:
        return {"ok": False, "error": "Could not read image."}

    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Edge/Feature detection for finding the price block
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_adj = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray_adj, (5, 5), 0)
    
    # Threshold to find the text blocks
    _, bw = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Close gaps between digits to form one single "price block"
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
    morphed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / float(h + 1e-6)
        area_ratio = (w * h) / (W * H)
        
        # Prices are usually wider than tall and not tiny/huge
        if 1.5 < aspect < 10.0 and 0.001 < area_ratio < 0.2:
            # Score based on being central and moderate size
            dist_center = abs(y + h/2 - H*0.5) / H 
            score = area_ratio - dist_center 
            candidates.append((score, (x, y, w, h)))

    if not candidates:
        return {"ok": False, "error": "No price candidate found"}

    # Sort by score and try OCR on the best candidates
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    for _, (x, y, w, h) in candidates[:3]: # Try top 3
        # Expand ROI slightly
        px, py = int(w*0.05), int(h*0.05)
        roi = img[max(0, y-py):min(H, y+h+py), max(0, x-px):min(W, x+w+px)]
        
        processed_roi = preprocess_for_ocr(roi)
        price, raw = ocr_price_logic(processed_roi)
        
        if price:
            return {
                "ok": True,
                "price": price,
                "raw_text": raw,
                "bbox": [x, y, w, h]
            }

    return {"ok": False, "error": "OCR failed to parse a price"}

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print(extract_price(sys.argv[1]))

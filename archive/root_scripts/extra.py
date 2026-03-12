import re
import cv2
import numpy as np
import pytesseract


def extract_price(image_path: str) -> dict:
    """
    Extract price from an image using OCR with improved bounding box detection.
    """
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

    # Save debug image
    cv2.imwrite("debug_gray.png", gray)

    # Otsu threshold
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Invert if needed (make digits white on black background)
    white = np.sum(bw == 255)
    black = np.sum(bw == 0)
    if white > black:
        print("Inverted binary image to make digits white")
        bw = cv2.bitwise_not(bw)

    cv2.imwrite("debug_bw.png", bw)

    # Try multiple morphology kernels to find the best bounding box
    best_candidates = []
    
    kernel_sizes = [
        (15, 5),   # Medium kernel
        (19, 6),   # Original kernel
        (25, 7),   # Larger kernel
        (11, 4),   # Smaller kernel
    ]
    
    for kw, kh in kernel_sizes:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
        bw2 = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)
        cv2.imwrite("bw2.png", bw2)
        # Find contours
        contours, _ = cv2.findContours(bw2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            area = cw * ch
            
            # Filter by minimum area
            if area < 0.005 * (w * h):
                continue

            aspect = cw / float(ch + 1e-6)

            # Price blocks tend to be wider than tall
            if aspect < 2.0 or aspect > 15.0:
                continue

            # Calculate scores
            # 1. Size score (prefer larger boxes but not too large)
            size_ratio = area / (w * h)
            if size_ratio < 0.001 or size_ratio > 0.99:
                continue
            size_score = min(size_ratio * 100, 1.0)
            
            # 2. Position score (prefer center-lower region, typical for prices)
            cx, cy = x + cw / 2.0, y + ch / 2.0
            # Prefer vertical center around 60-70% of height
            vertical_pref = 1.0 - abs(cy/h - 0.65) * 2
            vertical_pref = max(0, vertical_pref)
            
            # Prefer horizontal center
            horizontal_pref = 1.0 - abs(cx/w - 0.5) * 2
            horizontal_pref = max(0, horizontal_pref)
            
            position_score = (vertical_pref + horizontal_pref) / 2.0
            
            # 3. Aspect ratio score (prefer moderate width)
            ideal_aspect = 6.0
            aspect_score = 1.0 - min(abs(aspect - ideal_aspect) / ideal_aspect, 1.0)
            
            # Combined score
            score = size_score * 3.0 + position_score * 1.5 + aspect_score * 1.0
            
            best_candidates.append((score, (x, y, cw, ch), (kw, kh)))
    
    if not best_candidates:
        return {"ok": False, "error": "No price block detected. Try adjusting thresholds/kernels."}

    # Sort by score and try top candidates
    best_candidates.sort(reverse=True, key=lambda t: t[0])
    
    # Save debug image with all candidates
    dbg = img.copy()
    for i, (score, (x, y, cw, ch), kernel_size) in enumerate(best_candidates[:5]):
        color = (0, 255, 0) if i == 0 else (255, 0, 0)
        cv2.rectangle(dbg, (x, y), (x+cw, y+ch), color, 2)
        cv2.putText(dbg, f"{score:.2f}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.imwrite("debug_candidates.png", dbg)
    
    # Try OCR on top candidates until we get a valid price
    for score, (x, y, cw, ch), kernel_size in best_candidates[:5]:
        print(f"Trying candidate: score={score:.2f}, bbox=({x},{y},{cw},{ch}), kernel={kernel_size}")
        
        # Expand ROI slightly
        pad_x = int(0.05 * cw)
        pad_y = int(0.05 * ch)
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(w, x + cw + pad_x)
        y1 = min(h, y + ch + pad_y)

        roi = img[y0:y1, x0:x1]
        
        if roi.size == 0:
            continue
            
        # Process ROI for OCR
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_gray = cv2.GaussianBlur(roi_gray, (3, 3), 0)
        
        # Try multiple preprocessing approaches
        price, raw_text = ocr_price_multipass(roi_gray)
        
        if price is not None:
            print(f"SUCCESS: Extracted price={price}, raw_text='{raw_text}'")
            return {
                "ok": True,
                "price": price,
                "raw_text": raw_text,
                "bbox": [x0, y0, x1, y1],
                "score": score
            }
        else:
            print(f"Failed on this candidate: raw_text='{raw_text}'")
    
    # If all candidates failed
    return {
        "ok": False,
        "error": "OCR failed on all candidate regions",
        "tried_boxes": len(best_candidates)
    }


def ocr_price_multipass(roi_gray):
    """
    Try OCR with multiple preprocessing methods.
    Returns (price, raw_text) or (None, None).
    """
    config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.'
    
    preprocessing_methods = [
        # Method 1: Otsu threshold
        lambda img: cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        # Method 2: Otsu threshold inverted
        lambda img: cv2.bitwise_not(cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
        # Method 3: Adaptive threshold
        lambda img: cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),
        # Method 4: Adaptive threshold inverted
        lambda img: cv2.bitwise_not(cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)),
        # Method 5: Simple threshold at 127
        lambda img: cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1],
        # Method 6: Simple threshold inverted
        lambda img: cv2.bitwise_not(cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]),
    ]
    
    for i, method in enumerate(preprocessing_methods):
        try:
            processed = method(roi_gray)
            cv2.imwrite(f"debug_roi_method_{i}.png", processed)
            
            text = pytesseract.image_to_string(processed, config=config).strip()
            
            if not text:
                continue
            
            # Try to extract price from text
            price = parse_price_from_text(text)
            if price is not None:
                return price, text
                
        except Exception as e:
            print(f"Method {i} failed: {e}")
            continue
    
    return None, None


def parse_price_from_text(text):
    """
    Parse price from OCR text.
    Handles formats like: 3.459, 3.45, 459 (converts to 4.59), etc.
    """
    # Try decimal format first
    m = re.search(r'(\d+\.\d+)', text)
    if m:
        try:
            price = float(m.group(1))
            # Sanity check for fuel prices (typically 0.50 to 9.999)
            if 0.1 <= price <= 99.99:
                return price
        except ValueError:
            pass
    
    # Fallback: digits only and infer decimal position
    digits = re.sub(r'[^0-9]', '', text)
    if len(digits) >= 2:
        try:
            if len(digits) == 2:
                # 45 -> 0.45
                price = float("0." + digits)
            elif len(digits) == 3:
                # 345 -> 3.45
                price = float(digits[0] + "." + digits[1:3])
            elif len(digits) == 4:
                # 3459 -> 3.459
                price = float(digits[0] + "." + digits[1:4])
            else:
                # For longer sequences, assume first digit is dollars
                price = float(digits[0] + "." + digits[1:4])
            
            # Sanity check
            if 0.1 <= price <= 99.99:
                return price
        except ValueError:
            pass
    
    return None


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python extract_price_fixed.py <image_path>")
        raise SystemExit(2)

    result = extract_price(sys.argv[1])
    print("\n" + "="*50)
    print("FINAL RESULT:")
    print(result)
    print("="*50)

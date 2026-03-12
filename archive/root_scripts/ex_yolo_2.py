"""
Gas Price Extraction with YOLO-based Bounding Box Detection
============================================================
This version uses pretrained YOLOv8 to detect objects/regions instead of cv2.findContours,
then applies OCR on those regions to extract prices.

⚠️ IMPORTANT LIMITATION:
The pretrained YOLOv8n model is trained on the COCO dataset which has 80 classes like:
- person, car, bicycle, truck, bus, train, boat
- traffic light, fire hydrant, stop sign, parking meter
- bench, backpack, umbrella, etc.

It does NOT have a 'gas_sign' or 'price_board' class!

Therefore, this code has two options:

OPTION 1 (Current): Accept ALL detected objects as bounding box proposals
- Casts a wide net to avoid missing the price sign
- Relies on subsequent scoring/filtering to remove irrelevant boxes
- May generate many false candidates

OPTION 2 (Alternative): Filter for sign-like classes (stop sign, parking meter)
- More selective, fewer false positives
- BUT may miss the actual gas sign entirely

For production use, you should:
1. Train a custom YOLO model with 'gas_sign' class (see YOLO_INTEGRATION_GUIDE.md)
2. Or use the contour-based method (--use-contours flag)

The key change: YOLO replaces the contour detection step, providing more robust
bounding box proposals for price regions.
"""

import re
import cv2
import numpy as np
import pytesseract
from typing import Optional, Tuple, List, Dict
import sys


def load_yolo_model():
    """Load pretrained YOLOv8 model."""
    try:
        from ultralytics import YOLO
        print("Loading YOLOv8n model...")
        model = YOLO('yolov8n.pt')  # Will download automatically on first run
        return model
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        print("Falling back to contour detection...")
        return None
    except Exception as e:
        print(f"WARNING: Could not load YOLO model: {e}")
        print("Falling back to contour detection...")
        return None


def get_yolo_bounding_boxes(model, img: np.ndarray, confidence: float = 0.15, filter_mode: str = "all") -> List[Tuple[int, int, int, int]]:
    """
    Use YOLO to get bounding box proposals instead of cv2.findContours.
    
    NOTE: Pretrained YOLOv8n on COCO doesn't have a 'gas_sign' class!
    COCO classes (80 total): person, bicycle, car, motorcycle, airplane, bus, train,
    truck, boat, traffic light, fire hydrant, stop sign, parking meter, etc.
    
    Since there's no 'gas_sign' class, we filter for classes that might contain 
    price displays or signs. For best results, train a custom YOLO model.
    
    Args:
        model: YOLO model
        img: Input image
        confidence: Detection confidence threshold
        filter_mode: How to filter detections:
            - "all": Accept all objects (default)
            - "signs": Only sign-like objects (stop sign, parking meter, traffic light)
            - "custom": Assume custom model with gas_sign/price_board classes
    
    Returns:
        List of bounding boxes as (x, y, w, h)
    """
    if model is None:
        return []
    
    # Run YOLO inference
    results = model(img, conf=confidence, verbose=False)
    
    # Define relevant classes based on filter mode
    if filter_mode == "signs":
        # COCO class IDs: 11=stop sign, 12=parking meter, 9=traffic light
        relevant_classes = [11, 12, 9]
        print(f"  Filtering for sign-like classes: {relevant_classes}")
    elif filter_mode == "custom":
        # If you trained a custom model, it might have class 0=gas_sign
        relevant_classes = [0, 1, 2]  # Adjust based on your custom model
        print(f"  Using custom model classes: {relevant_classes}")
    else:  # "all"
        relevant_classes = None  # Accept everything
        print(f"  Accepting all detected objects (no class filtering)")
    
    # For debugging: show what classes are available
    if results and len(results) > 0:
        class_names = results[0].names
        print(f"  YOLO model has {len(class_names)} classes")
        if relevant_classes and filter_mode != "custom":
            filtered_names = [class_names.get(i, f'class_{i}') for i in relevant_classes]
            print(f"  Targeting: {filtered_names}")
    
    bboxes = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = result.names.get(class_id, f'class_{class_id}')
            
            # Apply class filtering if specified
            if relevant_classes is not None and class_id not in relevant_classes:
                continue
            
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Convert to x, y, w, h format
            x = int(x1)
            y = int(y1)
            w = int(x2 - x1)
            h = int(y2 - y1)
            
            print(f"  Detected: {class_name} (id={class_id}, conf={conf:.2f}, bbox=({x},{y},{w},{h}))")
            
            bboxes.append((x, y, w, h))
    
    return bboxes


def get_contour_bounding_boxes(gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Fallback method: Use original cv2.findContours approach.
    
    Args:
        gray: Grayscale image
    
    Returns:
        List of bounding boxes as (x, y, w, h)
    """
    h, w = gray.shape[:2]
    
    # Otsu threshold
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Invert if needed
    white = np.sum(bw == 255)
    black = np.sum(bw == 0)
    if white > black:
        bw = cv2.bitwise_not(bw)
    
    bboxes = []
    kernel_sizes = [(15, 5), (19, 6), (25, 7), (11, 4)]
    
    for kw, kh in kernel_sizes:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
        bw2 = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        contours, _ = cv2.findContours(bw2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            bboxes.append((x, y, cw, ch))
    
    return bboxes


def extract_price(image_path: str, use_yolo: bool = True, filter_mode: str = "all", debug: bool = True) -> Dict:
    """
    Extract price from image using YOLO for bounding box detection.
    
    Args:
        image_path: Path to input image
        use_yolo: If True, use YOLO for bbox detection; if False, use contours
        filter_mode: YOLO detection filtering mode:
            - "all": Accept all detections (default, casts wide net)
            - "signs": Only sign-like objects (stop sign, parking meter, traffic light)
            - "custom": Use this if you have a custom trained model with gas_sign class
        debug: Save debug images
    
    Returns:
        Dictionary with extraction results
    """
    img = cv2.imread(image_path)
    if img is None:
        return {"ok": False, "error": "Could not read image."}

    h, w = img.shape[:2]

    # 1) Preprocess
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    if debug:
        cv2.imwrite("debug_gray.png", gray)

    # 2) Get bounding box proposals - THIS IS THE KEY CHANGE
    if use_yolo:
        print(f"Using YOLO for bounding box detection (filter_mode='{filter_mode}')...")
        model = load_yolo_model()
        bboxes = get_yolo_bounding_boxes(model, img, confidence=0.15, filter_mode=filter_mode)
        
        if not bboxes:
            print("No YOLO detections, falling back to contour detection...")
            bboxes = get_contour_bounding_boxes(gray)
    else:
        print("Using contour detection for bounding boxes...")
        bboxes = get_contour_bounding_boxes(gray)

    if not bboxes:
        return {"ok": False, "error": "No bounding boxes detected"}

    print(f"Found {len(bboxes)} bounding box candidates")

    # 3) Score and filter bounding boxes
    candidates = []
    for x, y, cw, ch in bboxes:
        area = cw * ch
        
        # Filter by minimum area
        if area < 0.0005 * (w * h):
            continue

        aspect = cw / float(ch + 1e-6)

        # Price blocks tend to be wider than tall
        if aspect < 2.0 or aspect > 15.0:
            continue

        # Size ratio check
        size_ratio = area / (w * h)
        if size_ratio < 0.001 or size_ratio > 0.3:
            continue

        # Calculate scores
        size_score = min(size_ratio * 100, 1.0)
        
        # Position score (prefer center-lower region)
        cx, cy = x + cw / 2.0, y + ch / 2.0
        vertical_pref = 1.0 - abs(cy/h - 0.65) * 2
        vertical_pref = max(0, vertical_pref)
        horizontal_pref = 1.0 - abs(cx/w - 0.5) * 2
        horizontal_pref = max(0, horizontal_pref)
        position_score = (vertical_pref + horizontal_pref) / 2.0
        
        # Aspect ratio score
        ideal_aspect = 6.0
        aspect_score = 1.0 - min(abs(aspect - ideal_aspect) / ideal_aspect, 1.0)
        
        # Combined score
        score = size_score * 3.0 + position_score * 1.5 + aspect_score * 1.0
        
        candidates.append((score, (x, y, cw, ch)))

    if not candidates:
        return {"ok": False, "error": "No valid price candidates after filtering"}

    # Sort by score
    candidates.sort(reverse=True, key=lambda t: t[0])
    
    # Save debug image with candidates
    if debug:
        dbg = img.copy()
        for i, (score, (x, y, cw, ch)) in enumerate(candidates[:10]):
            color = (0, 255, 0) if i == 0 else (255, 0, 0) if i < 5 else (0, 0, 255)
            cv2.rectangle(dbg, (x, y), (x+cw, y+ch), color, 2)
            cv2.putText(dbg, f"{score:.2f}", (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.imwrite("debug_candidates.png", dbg)
        print(f"Saved debug image with {len(candidates)} candidates")

    # 4) Try OCR on top candidates
    print(f"Trying OCR on top {min(5, len(candidates))} candidates...")
    
    for idx, (score, (x, y, cw, ch)) in enumerate(candidates[:5]):
        print(f"  Candidate {idx+1}: score={score:.2f}, bbox=({x},{y},{cw},{ch})")
        
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
        price, raw_text = ocr_price_multipass(roi)
        
        if price is not None:
            print(f"  ✓ SUCCESS: Found price ${price:.3f}")
            return {
                "ok": True,
                "price": price,
                "raw_text": raw_text,
                "bbox": [x0, y0, x1, y1],
                "score": score,
                "method": "yolo" if use_yolo else "contours"
            }
        else:
            print(f"  ✗ Failed: OCR returned '{raw_text}'")

    return {
        "ok": False,
        "error": "OCR failed on all candidates",
        "candidates_tried": min(5, len(candidates))
    }


def ocr_price_multipass(roi: np.ndarray) -> Tuple[Optional[float], Optional[str]]:
    """
    Try OCR with multiple preprocessing methods.
    
    Args:
        roi: Region of interest image
    
    Returns:
        (price, raw_text) or (None, None) if failed
    """
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_gray = cv2.GaussianBlur(roi_gray, (3, 3), 0)
    
    config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.'
    
    # Try multiple preprocessing methods
    preprocessing_methods = [
        lambda img: cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        lambda img: cv2.bitwise_not(cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
        lambda img: cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),
        lambda img: cv2.bitwise_not(cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)),
        lambda img: cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1],
        lambda img: cv2.bitwise_not(cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]),
    ]
    
    for method in preprocessing_methods:
        try:
            processed = method(roi_gray)
            text = pytesseract.image_to_string(processed, config=config).strip()
            
            if not text:
                continue
            
            price = parse_price_from_text(text)
            if price is not None:
                return price, text
        except:
            continue
    
    return None, None


def parse_price_from_text(text: str) -> Optional[float]:
    """
    Parse price from OCR text.
    
    Args:
        text: OCR output text
    
    Returns:
        Price as float or None if parsing failed
    """
    # Try decimal format first
    m = re.search(r'(\d+\.\d+)', text)
    if m:
        try:
            price = float(m.group(1))
            if 0.1 <= price <= 99.99:
                return price
        except ValueError:
            pass
    
    # Fallback: digits only
    digits = re.sub(r'[^0-9]', '', text)
    if len(digits) >= 2:
        try:
            if len(digits) == 2:
                price = float("0." + digits)
            elif len(digits) == 3:
                price = float(digits[0] + "." + digits[1:3])
            elif len(digits) == 4:
                price = float(digits[0] + "." + digits[1:4])
            else:
                price = float(digits[0] + "." + digits[1:4])
            
            if 0.1 <= price <= 99.99:
                return price
        except ValueError:
            pass
    
    return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract gas prices using YOLO or contours for bbox detection')
    parser.add_argument('image_path', help='Path to input image')
    parser.add_argument('--use-contours', action='store_true', 
                       help='Use original contour detection instead of YOLO')
    parser.add_argument('--filter-mode', type=str, default='all',
                       choices=['all', 'signs', 'custom'],
                       help='YOLO detection filter mode: "all"=all objects (default), '
                            '"signs"=only sign-like objects, "custom"=custom trained model')
    parser.add_argument('--no-debug', action='store_true',
                       help='Disable debug image output')
    
    args = parser.parse_args()
    
    # Print warning about pretrained YOLO limitations
    if not args.use_contours:
        print("\n" + "="*60)
        print("⚠️  IMPORTANT: Using pretrained YOLOv8n (COCO dataset)")
        print("="*60)
        print("The pretrained model does NOT have a 'gas_sign' class!")
        print("It will detect general objects (person, car, signs, etc.)")
        print(f"Filter mode: {args.filter_mode}")
        print()
        if args.filter_mode == "all":
            print("→ Accepting ALL detected objects as bbox candidates")
            print("  (Relies on scoring/filtering to find price regions)")
        elif args.filter_mode == "signs":
            print("→ Filtering for sign-like objects only")
            print("  (stop sign, parking meter, traffic light)")
        print()
        print("For better results, consider:")
        print("  1. Training a custom YOLO model (see YOLO_INTEGRATION_GUIDE.md)")
        print("  2. Using --use-contours for the original method")
        print("="*60 + "\n")
    
    result = extract_price(
        args.image_path,
        use_yolo=not args.use_contours,
        filter_mode=args.filter_mode,
        debug=not args.no_debug
    )
    
    # Print result
    print("\n" + "="*60)
    print("FINAL RESULT:")
    print("="*60)
    if result['ok']:
        print(f"✓ Price: ${result['price']:.3f}")
        print(f"  Raw text: '{result['raw_text']}'")
        print(f"  Bounding box: {result['bbox']}")
        print(f"  Detection method: {result['method']}")
        print(f"  Score: {result['score']:.2f}")
    else:
        print(f"✗ Failed: {result['error']}")
    print("="*60)
    
    sys.exit(0 if result['ok'] else 1)

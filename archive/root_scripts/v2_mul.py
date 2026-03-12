#!/usr/bin/env python3
"""
Gas Price Extraction with Multiple OCR Engines - Enhanced for Multiple Prices

Recommended setup:
    pip install "paddleocr>=2.6,<3.0" paddlepaddle easyocr opencv-python numpy

PaddleOCR 3.x has significant API changes and requires model downloads.
PaddleOCR 2.x is recommended for simplicity and offline use.

Usage:
    python extract_price_multi.py <image_path> [--engine paddle|easy|all] [--debug]
"""

import re
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from enum import Enum
import warnings

warnings.filterwarnings("ignore")


class OCREngine(Enum):
    PADDLE = "paddle"
    EASY = "easy"
    TESSERACT = "tesseract"


@dataclass
class PriceResult:
    """Result of price extraction"""
    ok: bool
    price: Optional[float] = None
    prices: Optional[List[Dict[str, Any]]] = None  # List of all prices found
    raw_text: Optional[str] = None
    confidence: Optional[float] = None
    bbox: Optional[List[int]] = None
    engine: Optional[str] = None
    error: Optional[str] = None
    all_detections: Optional[List[Dict]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


class PriceExtractor:
    """
    Multi-engine price extractor for gas station signs.
    
    Supports PaddleOCR 2.x, EasyOCR, and Tesseract with automatic fallback.
    Enhanced to extract multiple prices from different fuel types.
    """
    
    def __init__(self, engines: List[OCREngine] = None, debug: bool = False):
        """
        Initialize the price extractor.
        
        Args:
            engines: List of OCR engines to use (in order of preference)
            debug: Whether to save debug images
        """
        self.debug = debug
        self.engines = engines or [OCREngine.PADDLE, OCREngine.EASY, OCREngine.TESSERACT]
        
        # Lazy-load OCR engines
        self._paddle_ocr = None
        self._paddle_version = None
        self._easy_reader = None
        self._tesseract_available = None
    
    def _get_paddle_version(self) -> Tuple[int, int]:
        """Get PaddleOCR version as (major, minor) tuple"""
        try:
            import paddleocr
            version_str = getattr(paddleocr, '__version__', '2.0.0')
            parts = version_str.split('.')
            major = int(parts[0]) if len(parts) > 0 else 2
            minor = int(parts[1]) if len(parts) > 1 else 0
            return (major, minor)
        except Exception:
            return (2, 0)
    
    @property
    def paddle_ocr(self):
        """Lazy-load PaddleOCR"""
        if self._paddle_ocr is None:
            try:
                from paddleocr import PaddleOCR
                
                self._paddle_version = self._get_paddle_version()
                major, minor = self._paddle_version
                print(f"[INFO] Detected PaddleOCR version: {major}.{minor}")
                
                if major >= 3:
                    # PaddleOCR 3.x - different API
                    print("[WARN] PaddleOCR 3.x detected. This version requires network access")
                    print("       to download models. Consider downgrading to 2.x:")
                    print('       pip install "paddleocr>=2.6,<3.0"')
                    try:
                        self._paddle_ocr = PaddleOCR(
                            ocr_version='PP-OCRv4',
                            lang='en',
                        )
                        print("[INFO] PaddleOCR 3.x initialized")
                    except Exception as e:
                        print(f"[ERROR] PaddleOCR 3.x initialization failed: {e}")
                        self._paddle_ocr = False
                else:
                    # PaddleOCR 2.x - recommended
                    self._paddle_ocr = PaddleOCR(
                        use_angle_cls=True,
                        lang='en',
                        show_log=False,
                        use_gpu=False,
                        det_db_thresh=0.3,
                        det_db_box_thresh=0.5,
                    )
                    print("[INFO] PaddleOCR 2.x initialized successfully")
                    
            except ImportError as e:
                print(f"[WARN] PaddleOCR not installed: {e}")
                print('       Install with: pip install "paddleocr>=2.6,<3.0" paddlepaddle')
                self._paddle_ocr = False
            except Exception as e:
                print(f"[WARN] PaddleOCR initialization failed: {e}")
                self._paddle_ocr = False
                
        return self._paddle_ocr if self._paddle_ocr else None
    
    @property
    def easy_reader(self):
        """Lazy-load EasyOCR"""
        if self._easy_reader is None:
            try:
                import easyocr
                self._easy_reader = easyocr.Reader(
                    ['en'],
                    gpu=False,
                    verbose=False
                )
                print("[INFO] EasyOCR initialized successfully")
            except ImportError:
                print("[WARN] EasyOCR not installed. Install with: pip install easyocr")
                self._easy_reader = False
            except Exception as e:
                print(f"[WARN] EasyOCR initialization failed: {e}")
                self._easy_reader = False
        return self._easy_reader if self._easy_reader else None
    
    @property
    def tesseract_available(self):
        """Check if Tesseract is available"""
        if self._tesseract_available is None:
            try:
                import pytesseract
                pytesseract.get_tesseract_version()
                self._tesseract_available = True
                print("[INFO] Tesseract available")
            except Exception:
                print("[WARN] Tesseract not available")
                self._tesseract_available = False
        return self._tesseract_available

    def preprocess_image(self, img: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        """Generate multiple preprocessed versions of the image."""
        h, w = img.shape[:2]
        results = []
        
        # 1. Original image
        results.append(("original", img.copy()))
        
        # 2. CLAHE enhancement
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        results.append(("clahe", cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)))
        
        # 3. Sharpened
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(img, -1, kernel)
        results.append(("sharpened", sharpened))
        
        # 4. Denoised
        denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        results.append(("denoised", denoised))
        
        # 5. High contrast LAB
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        enhanced_lab = cv2.merge([l, a, b])
        high_contrast = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        results.append(("high_contrast", high_contrast))
        
        # 6. Scale up small images
        if max(h, w) < 800:
            scale = 800 / max(h, w)
            scaled = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            results.append(("scaled_up", scaled))
        
        if self.debug:
            for name, processed in results:
                cv2.imwrite(f"debug_preprocess_{name}.png", processed)
        
        return results

    def extract_with_paddle(self, img: np.ndarray) -> List[Dict]:
        """Extract text using PaddleOCR."""
        if not self.paddle_ocr:
            return []
        
        try:
            major, _ = self._paddle_version
            
            if major >= 3:
                # PaddleOCR 3.x
                if hasattr(self._paddle_ocr, 'predict'):
                    result = self._paddle_ocr.predict(img)
                else:
                    result = self._paddle_ocr.ocr(img)
                return self._parse_paddle_v3_result(result)
            else:
                # PaddleOCR 2.x
                result = self._paddle_ocr.ocr(img, cls=True)
                return self._parse_paddle_v2_result(result)
            
        except Exception as e:
            print(f"[ERROR] PaddleOCR failed: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return []

    def _parse_paddle_v2_result(self, result) -> List[Dict]:
        """Parse PaddleOCR 2.x result: [[[box], (text, conf)], ...]"""
        detections = []
        
        if not result:
            return detections
        
        try:
            # v2 returns list of pages, first element is the page
            data = result[0] if result else []
            
            if not data:
                return detections
            
            for line in data:
                if not line or len(line) < 2:
                    continue
                
                bbox = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                text_info = line[1]  # (text, confidence)
                
                if isinstance(text_info, (tuple, list)) and len(text_info) >= 2:
                    text = str(text_info[0])
                    confidence = float(text_info[1])
                else:
                    continue
                
                # Convert polygon to [x, y, w, h]
                if bbox and len(bbox) >= 4:
                    x_coords = [p[0] for p in bbox]
                    y_coords = [p[1] for p in bbox]
                    x, y = int(min(x_coords)), int(min(y_coords))
                    w, h = int(max(x_coords) - x), int(max(y_coords) - y)
                    
                    detections.append({
                        'text': text,
                        'confidence': confidence,
                        'bbox': [x, y, w, h],
                        'engine': 'paddle'
                    })
                    
        except Exception as e:
            print(f"[WARN] Error parsing PaddleOCR v2 result: {e}")
            if self.debug:
                print(f"       Raw result: {result}")
        
        return detections

    def _parse_paddle_v3_result(self, result) -> List[Dict]:
        """Parse PaddleOCR 3.x result format."""
        detections = []
        
        if result is None:
            return detections
        
        try:
            # v3 can return various formats depending on the method used
            items = result if isinstance(result, list) else [result]
            
            for item in items:
                # Try to extract text and bbox from various possible formats
                text = None
                conf = 0.5
                bbox = None
                
                if isinstance(item, dict):
                    # Dict format: {'text': ..., 'confidence': ..., 'text_box_position': ...}
                    text = item.get('text') or item.get('rec_text')
                    conf = item.get('confidence') or item.get('rec_score', 0.5)
                    bbox = item.get('text_box_position') or item.get('dt_boxes')
                    
                elif hasattr(item, 'text') or hasattr(item, 'rec_text'):
                    # Object with attributes
                    text = getattr(item, 'text', None) or getattr(item, 'rec_text', None)
                    conf = getattr(item, 'confidence', None) or getattr(item, 'rec_score', 0.5)
                    bbox = getattr(item, 'text_box_position', None) or getattr(item, 'dt_boxes', None)
                    
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    # Legacy format: [[box], (text, conf)]
                    bbox = item[0]
                    text_info = item[1]
                    if isinstance(text_info, (tuple, list)) and len(text_info) >= 2:
                        text = text_info[0]
                        conf = text_info[1]
                    elif isinstance(text_info, str):
                        text = text_info
                
                if text and bbox:
                    # Convert bbox to [x, y, w, h]
                    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                        if isinstance(bbox[0], (list, tuple)):
                            # Polygon format
                            x_coords = [p[0] for p in bbox]
                            y_coords = [p[1] for p in bbox]
                            x, y = int(min(x_coords)), int(min(y_coords))
                            w, h = int(max(x_coords) - x), int(max(y_coords) - y)
                        else:
                            # Already [x, y, w, h] or [x1, y1, x2, y2]
                            x, y = int(bbox[0]), int(bbox[1])
                            w, h = int(bbox[2]), int(bbox[3])
                        
                        detections.append({
                            'text': str(text),
                            'confidence': float(conf) if conf else 0.5,
                            'bbox': [x, y, w, h],
                            'engine': 'paddle'
                        })
                        
        except Exception as e:
            print(f"[WARN] Error parsing PaddleOCR v3 result: {e}")
            if self.debug:
                print(f"       Result type: {type(result)}")
                print(f"       Result: {result}")
        
        return detections

    def extract_with_easy(self, img: np.ndarray) -> List[Dict]:
        """Extract text using EasyOCR."""
        if not self.easy_reader:
            return []
        
        try:
            results = self.easy_reader.readtext(img)
            
            detections = []
            for bbox, text, confidence in results:
                x_coords = [p[0] for p in bbox]
                y_coords = [p[1] for p in bbox]
                x, y = int(min(x_coords)), int(min(y_coords))
                w, h = int(max(x_coords) - x), int(max(y_coords) - y)
                
                detections.append({
                    'text': text,
                    'confidence': confidence,
                    'bbox': [x, y, w, h],
                    'engine': 'easy'
                })
            
            return detections
            
        except Exception as e:
            print(f"[ERROR] EasyOCR failed: {e}")
            return []

    def extract_with_tesseract(self, img: np.ndarray) -> List[Dict]:
        """Extract text using Tesseract."""
        if not self.tesseract_available:
            return []
        
        try:
            import pytesseract
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            data = pytesseract.image_to_data(bw, output_type=pytesseract.Output.DICT,
                                             config='--oem 3 --psm 11')
            
            detections = []
            n_boxes = len(data['text'])
            for i in range(n_boxes):
                text = data['text'][i].strip()
                conf = int(data['conf'][i])
                
                if text and conf > 0:
                    detections.append({
                        'text': text,
                        'confidence': conf / 100.0,
                        'bbox': [data['left'][i], data['top'][i], 
                                data['width'][i], data['height'][i]],
                        'engine': 'tesseract'
                    })
            
            return detections
            
        except Exception as e:
            print(f"[ERROR] Tesseract failed: {e}")
            return []

    def parse_price(self, text: str) -> Optional[float]:
        """Parse price from OCR text."""
        if not text:
            return None
        
        # Clean text
        text = text.strip()
        text = re.sub(r'[$€£¥]', '', text)
        text = re.sub(r'[⁹⁰¹²³⁴⁵⁶⁷⁸]', '', text)
        text = re.sub(r'\s+', '', text)
        
        # Common OCR mistakes
        text = text.replace('O', '0').replace('o', '0')
        text = text.replace('l', '1').replace('I', '1')
        text = text.replace(',', '.')
        text = text.replace('S', '5').replace('s', '5')
        text = text.replace('B', '8')
        
        # Try decimal format (e.g., "3.459")
        match = re.search(r'(\d+\.\d+)', text)
        if match:
            try:
                price = float(match.group(1))
                if 0.1 <= price <= 99.99:
                    return round(price, 3)
            except ValueError:
                pass
        
        # Try digits only (e.g., "3459" -> 3.459)
        digits = re.sub(r'[^0-9]', '', text)
        if 2 <= len(digits) <= 5:
            try:
                if len(digits) == 2:
                    price = float("0." + digits)
                elif len(digits) == 3:
                    price = float(digits[0] + "." + digits[1:3])
                elif len(digits) == 4:
                    price = float(digits[0] + "." + digits[1:4])
                elif len(digits) == 5:
                    price = float(digits[0] + "." + digits[1:4])
                else:
                    return None
                
                if 0.1 <= price <= 99.99:
                    return round(price, 3)
            except ValueError:
                pass
        
        return None

    def score_detection(self, detection: Dict, img_shape: Tuple[int, int]) -> float:
        """Score a detection based on likelihood of being a gas price."""
        h, w = img_shape[:2]
        text = detection['text']
        confidence = detection['confidence']
        bbox = detection['bbox']
        
        score = 0.0
        
        # 1. OCR confidence
        score += confidence * 30
        
        # 2. Price pattern match
        price = self.parse_price(text)
        if price:
            score += 25
            if 1.0 <= price <= 7.0:
                score += 15
            elif 0.5 <= price <= 10.0:
                score += 8
        
        # 3. Text characteristics
        if '.' in text:
            score += 10
        
        digits = re.sub(r'[^0-9]', '', text)
        if 3 <= len(digits) <= 4:
            score += 10
        elif 2 <= len(digits) <= 5:
            score += 5
        
        # 4. Position scoring
        x, y, bw, bh = bbox
        cx, cy = x + bw/2, y + bh/2
        
        horizontal_center = 1.0 - abs(cx/w - 0.5) * 1.5
        vertical_pref = 1.0 - abs(cy/h - 0.4) * 1.5
        position_score = max(0, (horizontal_center + vertical_pref) / 2) * 10
        score += position_score
        
        # 5. Size scoring
        area_ratio = (bw * bh) / (w * h)
        if 0.005 <= area_ratio <= 0.15:
            score += 8
        elif 0.002 <= area_ratio <= 0.3:
            score += 4
        
        return score

    def group_prices_by_proximity(self, price_detections: List[Dict], img_shape: Tuple[int, int]) -> List[List[Dict]]:
        """Group price detections that are likely part of the same fuel type row."""
        if not price_detections:
            return []
        
        h, w = img_shape[:2]
        
        # Sort by vertical position (y coordinate)
        sorted_prices = sorted(price_detections, key=lambda d: d['bbox'][1])
        
        groups = []
        current_group = [sorted_prices[0]]
        
        for i in range(1, len(sorted_prices)):
            curr_det = sorted_prices[i]
            prev_det = sorted_prices[i-1]
            
            curr_y = curr_det['bbox'][1]
            prev_y = prev_det['bbox'][1]
            prev_h = prev_det['bbox'][3]
            
            # If y-coordinates are close (within 1.5x height), consider same row
            y_distance = abs(curr_y - prev_y)
            height_threshold = max(prev_h * 1.5, h * 0.05)  # At least 5% of image height
            
            if y_distance < height_threshold:
                current_group.append(curr_det)
            else:
                groups.append(current_group)
                current_group = [curr_det]
        
        # Add the last group
        if current_group:
            groups.append(current_group)
        
        return groups

    def find_fuel_label(self, price_det: Dict, all_detections: List[Dict], img_shape: Tuple[int, int]) -> Optional[str]:
        """Find fuel type label near a price detection."""
        h, w = img_shape[:2]
        px, py, pw, ph = price_det['bbox']
        
        # Fuel type keywords
        fuel_keywords = [
            'regular', 'unleaded', 'plus', 'mid', 'midgrade', 'premium', 
            'super', 'diesel', 'e85', 'ethanol', 'ultra', 'v-power',
            'reg', 'prem', 'sup', 'unlead'
        ]
        
        candidates = []
        
        for det in all_detections:
            if det == price_det:
                continue
            
            text = det['text'].lower()
            
            # Check if text contains fuel keywords
            if any(keyword in text for keyword in fuel_keywords):
                dx, dy, dw, dh = det['bbox']
                
                # Calculate proximity (should be in same row or slightly above/below)
                y_distance = abs(dy - py)
                x_distance = abs(dx - px)
                
                # Prefer labels to the left of the price
                horizontal_score = 0
                if dx < px:  # Label is to the left
                    horizontal_score = 2.0
                elif dx > px + pw:  # Label is to the right
                    horizontal_score = 1.0
                else:  # Overlapping x
                    horizontal_score = 1.5
                
                # Vertical proximity score
                vertical_threshold = max(ph * 2, h * 0.08)
                if y_distance < vertical_threshold:
                    vertical_score = 1.0 - (y_distance / vertical_threshold)
                else:
                    continue
                
                proximity_score = (horizontal_score + vertical_score * 2) / 3
                
                candidates.append({
                    'text': text,
                    'score': proximity_score,
                    'y_distance': y_distance,
                    'x_distance': x_distance
                })
        
        if candidates:
            # Sort by proximity score
            candidates.sort(key=lambda x: x['score'], reverse=True)
            return candidates[0]['text']
        
        return None

    def extract_all_prices(self, image_path: str) -> PriceResult:
        """Extract all prices from an image, grouping by fuel type."""
        img = cv2.imread(image_path)
        if img is None:
            return PriceResult(ok=False, error=f"Could not read image: {image_path}")
        
        print(f"[INFO] Processing image: {image_path} ({img.shape[1]}x{img.shape[0]})")
        
        preprocessed = self.preprocess_image(img)
        all_detections = []
        
        for preprocess_name, processed_img in preprocessed:
            for engine in self.engines:
                print(f"[INFO] Trying {engine.value} with {preprocess_name}...")
                
                if engine == OCREngine.PADDLE:
                    detections = self.extract_with_paddle(processed_img)
                elif engine == OCREngine.EASY:
                    detections = self.extract_with_easy(processed_img)
                elif engine == OCREngine.TESSERACT:
                    detections = self.extract_with_tesseract(processed_img)
                else:
                    continue
                
                for det in detections:
                    det['score'] = self.score_detection(det, img.shape)
                    det['preprocess'] = preprocess_name
                    all_detections.append(det)
                
                if detections:
                    print(f"       Found {len(detections)} text regions")
        
        if not all_detections:
            return PriceResult(
                ok=False,
                error="No text detected by any OCR engine",
                all_detections=[]
            )
        
        all_detections.sort(key=lambda x: x['score'], reverse=True)
        
        # Extract all valid prices
        price_detections = []
        seen_positions = set()
        
        for det in all_detections:
            price = self.parse_price(det['text'])
            if price is not None:
                # Check if we already have a price at roughly this position
                x, y, w, h = det['bbox']
                pos_key = (round(x / 20), round(y / 20))  # Group positions within 20px
                
                if pos_key not in seen_positions:
                    seen_positions.add(pos_key)
                    det['price'] = price
                    price_detections.append(det)
        
        if not price_detections:
            return PriceResult(
                ok=False,
                error="No valid price pattern found in detected text",
                all_detections=all_detections[:20]
            )
        
        # Group prices and find labels
        grouped_prices = self.group_prices_by_proximity(price_detections, img.shape)
        
        result_prices = []
        for group_idx, group in enumerate(grouped_prices):
            for det in group:
                fuel_label = self.find_fuel_label(det, all_detections, img.shape)
                
                price_info = {
                    'price': det['price'],
                    'raw_text': det['text'],
                    'confidence': det['confidence'],
                    'bbox': det['bbox'],
                    'engine': det['engine'],
                    'score': det['score'],
                    'fuel_type': fuel_label if fuel_label else f"Unknown (Group {group_idx + 1})",
                    'group': group_idx
                }
                result_prices.append(price_info)
        
        # Sort by position (top to bottom)
        result_prices.sort(key=lambda x: x['bbox'][1])
        
        if self.debug:
            self._save_debug_visualization_multi(img, result_prices, all_detections)
        
        # Print all found prices
        print(f"\n[SUCCESS] Found {len(result_prices)} price(s):")
        for i, p in enumerate(result_prices, 1):
            print(f"  {i}. ${p['price']:.3f} - {p['fuel_type']}")
            print(f"     (text='{p['raw_text']}', engine={p['engine']}, score={p['score']:.2f})")
        
        return PriceResult(
            ok=True,
            price=result_prices[0]['price'] if result_prices else None,  # First/highest price
            prices=result_prices,
            raw_text=result_prices[0]['raw_text'] if result_prices else None,
            confidence=result_prices[0]['confidence'] if result_prices else None,
            bbox=result_prices[0]['bbox'] if result_prices else None,
            engine=result_prices[0]['engine'] if result_prices else None,
            all_detections=all_detections[:20]
        )

    def extract_price(self, image_path: str) -> PriceResult:
        """Main method to extract price(s) from an image - now returns all prices."""
        return self.extract_all_prices(image_path)

    def _save_debug_visualization_multi(self, img: np.ndarray, prices: List[Dict], all_detections: List[Dict]):
        """Save debug image with all detected prices highlighted."""
        debug_img = img.copy()
        
        # Draw all detections in light gray
        for det in all_detections[:30]:
            x, y, w, h = det['bbox']
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (200, 200, 200), 1)
        
        # Draw detected prices in color
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]
        
        for i, price_info in enumerate(prices):
            x, y, w, h = price_info['bbox']
            color = colors[i % len(colors)]
            
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), color, 3)
            
            label = f"${price_info['price']:.3f}"
            if price_info.get('fuel_type') and 'Unknown' not in price_info['fuel_type']:
                label = f"{price_info['fuel_type']}: {label}"
            
            # Draw label with background
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(debug_img, (x, y-label_h-10), (x+label_w+10, y), color, -1)
            cv2.putText(debug_img, label, (x+5, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imwrite("debug_all_prices.png", debug_img)
        print(f"[DEBUG] Saved visualization to debug_all_prices.png")

    def _save_debug_visualization(self, img: np.ndarray, detections: List[Dict]):
        """Save debug image with all detections."""
        debug_img = img.copy()
        
        for i, det in enumerate(detections[:15]):
            x, y, w, h = det['bbox']
            green = max(0, 255 - i * 30)
            red = min(255, i * 30)
            color = (0, green, red)
            
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), color, 2)
            label = f"{det['text'][:10]} ({det['score']:.1f})"
            cv2.putText(debug_img, label, (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        cv2.imwrite("debug_all_detections.png", debug_img)
        print(f"[DEBUG] Saved visualization to debug_all_detections.png")


def extract_price(image_path: str, engine: str = "all", debug: bool = False) -> dict:
    """Convenience function for backward compatibility."""
    if engine == "all":
        engines = [OCREngine.PADDLE, OCREngine.EASY, OCREngine.TESSERACT]
    elif engine == "paddle":
        engines = [OCREngine.PADDLE]
    elif engine == "easy":
        engines = [OCREngine.EASY]
    elif engine == "tesseract":
        engines = [OCREngine.TESSERACT]
    else:
        engines = [OCREngine.PADDLE, OCREngine.EASY]
    
    extractor = PriceExtractor(engines=engines, debug=debug)
    result = extractor.extract_price(image_path)
    return result.to_dict()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract gas prices from images (multiple prices)")
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("--engine", choices=["paddle", "easy", "tesseract", "all"],
                       default="all", help="OCR engine to use (default: all)")
    parser.add_argument("--debug", action="store_true", help="Save debug images")
    
    args = parser.parse_args()
    
    result = extract_price(args.image_path, engine=args.engine, debug=args.debug)
    
    print("\n" + "=" * 60)
    print("EXTRACTION RESULT:")
    print("=" * 60)
    
    if result.get('ok'):
        if result.get('prices'):
            print(f"\n  Found {len(result['prices'])} price(s):\n")
            for i, price_info in enumerate(result['prices'], 1):
                print(f"  {i}. ${price_info['price']:.3f} - {price_info.get('fuel_type', 'Unknown')}")
                print(f"     Raw text: '{price_info['raw_text']}'")
                print(f"     Confidence: {price_info['confidence']:.2%}")
                print(f"     Engine: {price_info['engine']}")
                print()
        else:
            print(f"  Price: ${result['price']}")
            print(f"  Raw Text: '{result.get('raw_text', 'N/A')}'")
            print(f"  Confidence: {result.get('confidence', 0):.2%}")
            print(f"  Engine: {result.get('engine', 'N/A')}")
    else:
        print(f"  Error: {result.get('error', 'Unknown error')}")
    
    print("=" * 60)

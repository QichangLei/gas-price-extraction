import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
import os
os.environ["FLAGS_use_pir_api"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"
os.environ["FLAGS_use_new_executor"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"          
# disables oneDNN/MKLDNN

#!/usr/bin/env python3
"""
Gas Price Extraction with Multiple OCR Engines

Compatible with PaddleOCR 3.x (tested with 3.4)

Usage:
    python extract_price_v3.py <image_path> [--engine paddle|easy|all]

Requirements:
    pip install paddleocr>=3.0.0 paddlepaddle easyocr opencv-python numpy
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
    
    Supports PaddleOCR 3.x, EasyOCR, and Tesseract with automatic fallback.
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
    
    def _detect_paddle_version(self) -> int:
        """Detect PaddleOCR major version"""
        try:
            import paddleocr
            version = getattr(paddleocr, '__version__', '2.0.0')
            major = int(version.split('.')[0])
            print(f"[INFO] Detected PaddleOCR version: {version} (major: {major})")
            return major
        except Exception:
            return 2  # Default to v2 API
    
    @property
    def paddle_ocr(self):
        """Lazy-load PaddleOCR with version detection"""
        if self._paddle_ocr is None:
            try:
                from paddleocr import PaddleOCR
                
                self._paddle_version = self._detect_paddle_version()
                
                if self._paddle_version >= 3:
                    # PaddleOCR 3.x API
                    self._paddle_ocr = PaddleOCR(
                        ocr_version='PP-OCRv4',  # Best model in v3
                        lang='en',
                        device = "cpu"
                        #show_log=False,
                        #use_gpu=False,  # Set True if GPU available
                    )
                    print("[INFO] PaddleOCR 3.x initialized with PP-OCRv4")
                else:
                    # PaddleOCR 2.x API (legacy)
                    self._paddle_ocr = PaddleOCR(
                        use_angle_cls=True,
                        lang='en',
                        #show_log=False,
                        use_gpu=False,
                        det_db_thresh=0.3,
                        det_db_box_thresh=0.5,
                    )
                    print("[INFO] PaddleOCR 2.x initialized")
                    
            except ImportError as e:
                print(f"[WARN] PaddleOCR not installed: {e}")
                print("       Install with: pip install paddleocr paddlepaddle")
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
                    gpu=False,  # Set True if GPU available
                    verbose=False
                )
                print("[INFO] EasyOCR initialized successfully")
            except ImportError:
                print("[WARN] EasyOCR not installed. Install with: pip install easyocr")
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
        """
        Generate multiple preprocessed versions of the image.
        
        Returns list of (name, preprocessed_image) tuples.
        """
        h, w = img.shape[:2]
        results = []
        
        # 1. Original image (good for high quality inputs)
        results.append(("original", img.copy()))
        
        # 2. Grayscale with CLAHE enhancement
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        results.append(("clahe", cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)))
        
        # 3. Sharpened image
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(img, -1, kernel)
        results.append(("sharpened", sharpened))
        
        # 4. Denoised image (good for noisy night shots)
        denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        results.append(("denoised", denoised))
        
        # 5. High contrast version
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        enhanced_lab = cv2.merge([l, a, b])
        high_contrast = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        results.append(("high_contrast", high_contrast))
        
        # 6. Scale up small images (helps with distant shots)
        if max(h, w) < 800:
            scale = 800 / max(h, w)
            scaled = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            results.append(("scaled_up", scaled))
        
        if self.debug:
            for name, processed in results:
                cv2.imwrite(f"debug_preprocess_{name}.png", processed)
        
        return results

    def _parse_paddle_v3_result(self, result) -> List[Dict]:
        """
        Parse PaddleOCR 3.x result format.
        
        v3.x returns a list of TextResult objects or similar structure.
        """
        detections = []
        
        if result is None:
            return detections
        
        try:
            # PaddleOCR 3.x returns different structures depending on input
            # Try to handle various formats
            
            # Format 1: List of dictionaries with 'text', 'confidence', 'text_box_position'
            if isinstance(result, list):
                for item in result:
                    if isinstance(item, dict):
                        # Direct dict format
                        text = item.get('text', item.get('rec_text', ''))
                        conf = item.get('confidence', item.get('rec_score', 0.0))
                        bbox = item.get('text_box_position', item.get('dt_boxes', []))
                        
                        if text and bbox:
                            # Convert bbox to [x, y, w, h]
                            if len(bbox) >= 4:
                                x_coords = [p[0] if isinstance(p, (list, tuple)) else bbox[0] for p in bbox[:4]]
                                y_coords = [p[1] if isinstance(p, (list, tuple)) else bbox[1] for p in bbox[:4]]
                                
                                if isinstance(bbox[0], (list, tuple)):
                                    x_coords = [p[0] for p in bbox]
                                    y_coords = [p[1] for p in bbox]
                                    x, y = int(min(x_coords)), int(min(y_coords))
                                    w, h = int(max(x_coords) - x), int(max(y_coords) - y)
                                else:
                                    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                                
                                detections.append({
                                    'text': str(text),
                                    'confidence': float(conf) if conf else 0.5,
                                    'bbox': [x, y, w, h],
                                    'engine': 'paddle'
                                })
                    
                    elif hasattr(item, 'text') or hasattr(item, 'rec_text'):
                        # Object with attributes (TextResult-like)
                        text = getattr(item, 'text', getattr(item, 'rec_text', ''))
                        conf = getattr(item, 'confidence', getattr(item, 'rec_score', 0.5))
                        bbox = getattr(item, 'text_box_position', getattr(item, 'dt_boxes', []))
                        
                        if text:
                            if hasattr(bbox, '__iter__') and len(bbox) >= 4:
                                if isinstance(bbox[0], (list, tuple)):
                                    x_coords = [p[0] for p in bbox]
                                    y_coords = [p[1] for p in bbox]
                                    x, y = int(min(x_coords)), int(min(y_coords))
                                    w, h = int(max(x_coords) - x), int(max(y_coords) - y)
                                else:
                                    x, y, w, h = 0, 0, 100, 30  # Default
                            else:
                                x, y, w, h = 0, 0, 100, 30
                            
                            detections.append({
                                'text': str(text),
                                'confidence': float(conf) if conf else 0.5,
                                'bbox': [x, y, w, h],
                                'engine': 'paddle'
                            })
                    
                    elif isinstance(item, (list, tuple)) and len(item) >= 2:
                        # Format 2: Legacy format [[box], (text, conf)]
                        # This might still be returned in some cases
                        bbox = item[0]
                        text_conf = item[1]
                        
                        if isinstance(text_conf, (list, tuple)) and len(text_conf) >= 2:
                            text = text_conf[0]
                            conf = text_conf[1]
                        elif isinstance(text_conf, str):
                            text = text_conf
                            conf = 0.5
                        else:
                            continue
                        
                        if bbox and len(bbox) >= 4:
                            x_coords = [p[0] for p in bbox]
                            y_coords = [p[1] for p in bbox]
                            x, y = int(min(x_coords)), int(min(y_coords))
                            w, h = int(max(x_coords) - x), int(max(y_coords) - y)
                            
                            detections.append({
                                'text': str(text),
                                'confidence': float(conf),
                                'bbox': [x, y, w, h],
                                'engine': 'paddle'
                            })
            
            # If result has a 'data' attribute (some v3 formats)
            elif hasattr(result, 'data'):
                return self._parse_paddle_v3_result(result.data)
            
        except Exception as e:
            print(f"[WARN] Error parsing PaddleOCR v3 result: {e}")
            print(f"       Result type: {type(result)}")
            if self.debug:
                print(f"       Result content: {result}")
        
        return detections

    def _parse_paddle_v2_result(self, result) -> List[Dict]:
        """
        Parse PaddleOCR 2.x result format.
        
        v2.x returns: [[[box], (text, conf)], ...]
        """
        detections = []
        
        if result is None or not result:
            return detections
        
        try:
            # v2 format: result is a list, first element contains the detections
            data = result[0] if result else []
            
            for line in data:
                if not line or len(line) < 2:
                    continue
                    
                bbox = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                text = line[1][0]
                confidence = line[1][1]
                
                # Convert polygon bbox to [x, y, w, h]
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
        
        return detections

    def extract_with_paddle(self, img: np.ndarray) -> List[Dict]:
        """
        Extract text using PaddleOCR.
        
        Automatically handles v2.x and v3.x result formats.
        """
        if not self.paddle_ocr:
            return []
        
        try:
            # Call OCR
            if self._paddle_version >= 3:
                # v3.x API - predict method or direct call
                if hasattr(self._paddle_ocr, 'predict'):
                    result = self._paddle_ocr.predict(img)
                else:
                    result = self._paddle_ocr.ocr(img)
            else:
                # v2.x API
                result = self._paddle_ocr.ocr(img, cls=True)
            
            if self.debug:
                print(f"[DEBUG] PaddleOCR raw result type: {type(result)}")
                print(f"[DEBUG] PaddleOCR raw result: {result[:2] if isinstance(result, list) and len(result) > 2 else result}")
            
            # Parse based on version
            if self._paddle_version >= 3:
                detections = self._parse_paddle_v3_result(result)
            else:
                detections = self._parse_paddle_v2_result(result)
            
            # If v3 parsing failed, try v2 format (sometimes v3 returns v2-style)
            if not detections and self._paddle_version >= 3:
                detections = self._parse_paddle_v2_result(result)
            
            return detections
            
        except Exception as e:
            print(f"[ERROR] PaddleOCR failed: {e}")
            import traceback
            if self.debug:
                traceback.print_exc()
            return []

    def extract_with_easy(self, img: np.ndarray) -> List[Dict]:
        """
        Extract text using EasyOCR.
        
        Returns list of detections with text, confidence, and bbox.
        """
        if not self.easy_reader:
            return []
        
        try:
            results = self.easy_reader.readtext(img)
            
            detections = []
            for bbox, text, confidence in results:
                # bbox is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
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
        """
        Extract text using Tesseract (legacy fallback).
        
        Returns list of detections with text, confidence, and bbox.
        """
        if not self.tesseract_available:
            return []
        
        try:
            import pytesseract
            
            # Preprocess for Tesseract
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Get detailed data
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
        """
        Parse price from OCR text.
        
        Handles various formats:
        - 3.459, 3.45 (standard decimal)
        - 3459, 345 (no decimal point - infer position)
        - $3.45 (with currency symbol)
        - 3.45⁹ (with superscript 9/10)
        """
        if not text:
            return None
        
        # Clean text
        text = text.strip()
        text = re.sub(r'[$€£¥]', '', text)  # Remove currency symbols
        text = re.sub(r'[⁹⁰¹²³⁴⁵⁶⁷⁸]', '', text)  # Remove superscripts
        text = re.sub(r'\s+', '', text)  # Remove whitespace
        
        # Replace common OCR mistakes
        text = text.replace('O', '0').replace('o', '0')
        text = text.replace('l', '1').replace('I', '1')
        text = text.replace(',', '.')
        text = text.replace('S', '5').replace('s', '5')
        text = text.replace('B', '8')
        
        # Try decimal format first (e.g., "3.459", "3.45")
        match = re.search(r'(\d+\.\d+)', text)
        if match:
            try:
                price = float(match.group(1))
                # Sanity check for fuel prices (typically 0.50 to 9.999)
                if 0.1 <= price <= 99.99:
                    return round(price, 3)
            except ValueError:
                pass
        
        # Try digits only and infer decimal (e.g., "3459" -> 3.459)
        digits = re.sub(r'[^0-9]', '', text)
        if 2 <= len(digits) <= 5:
            try:
                if len(digits) == 2:
                    # 45 -> 0.45 (unlikely for gas prices)
                    price = float("0." + digits)
                elif len(digits) == 3:
                    # 345 -> 3.45
                    price = float(digits[0] + "." + digits[1:3])
                elif len(digits) == 4:
                    # 3459 -> 3.459
                    price = float(digits[0] + "." + digits[1:4])
                elif len(digits) == 5:
                    # 34599 -> 3.459 (take first 4 meaningful)
                    price = float(digits[0] + "." + digits[1:4])
                else:
                    return None
                
                # Sanity check
                if 0.1 <= price <= 99.99:
                    return round(price, 3)
            except ValueError:
                pass
        
        return None

    def score_detection(self, detection: Dict, img_shape: Tuple[int, int]) -> float:
        """
        Score a detection based on likelihood of being a gas price.
        
        Higher score = more likely to be a valid gas price.
        """
        h, w = img_shape[:2]
        text = detection['text']
        confidence = detection['confidence']
        bbox = detection['bbox']
        
        score = 0.0
        
        # 1. Base confidence from OCR engine
        score += confidence * 30
        
        # 2. Check if it looks like a price
        price = self.parse_price(text)
        if price:
            score += 25
            # Typical gas price range bonus (1.50 - 6.00)
            if 1.0 <= price <= 7.0:
                score += 15
            elif 0.5 <= price <= 10.0:
                score += 8
        
        # 3. Text characteristics
        # Contains decimal point
        if '.' in text:
            score += 10
        
        # Reasonable length for a price
        digits = re.sub(r'[^0-9]', '', text)
        if 3 <= len(digits) <= 4:
            score += 10
        elif 2 <= len(digits) <= 5:
            score += 5
        
        # 4. Position scoring (prices often in specific regions)
        x, y, bw, bh = bbox
        cx, cy = x + bw/2, y + bh/2
        
        # Prefer center and upper-center regions
        horizontal_center = 1.0 - abs(cx/w - 0.5) * 1.5
        vertical_pref = 1.0 - abs(cy/h - 0.4) * 1.5  # Slightly favor upper half
        position_score = max(0, (horizontal_center + vertical_pref) / 2) * 10
        score += position_score
        
        # 5. Size scoring
        area_ratio = (bw * bh) / (w * h)
        if 0.005 <= area_ratio <= 0.15:  # Not too small, not too large
            score += 8
        elif 0.002 <= area_ratio <= 0.3:
            score += 4
        
        return score

    def extract_price(self, image_path: str) -> PriceResult:
        """
        Main method to extract price from an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            PriceResult with extraction results
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return PriceResult(ok=False, error=f"Could not read image: {image_path}")
        
        print(f"[INFO] Processing image: {image_path} ({img.shape[1]}x{img.shape[0]})")
        
        # Generate preprocessed versions
        preprocessed = self.preprocess_image(img)
        
        all_detections = []
        
        # Try each preprocessing variant with each engine
        for preprocess_name, processed_img in preprocessed:
            for engine in self.engines:
                print(f"[INFO] Trying {engine.value} with {preprocess_name} preprocessing...")
                
                if engine == OCREngine.PADDLE:
                    detections = self.extract_with_paddle(processed_img)
                elif engine == OCREngine.EASY:
                    detections = self.extract_with_easy(processed_img)
                elif engine == OCREngine.TESSERACT:
                    detections = self.extract_with_tesseract(processed_img)
                else:
                    continue
                
                # Score each detection
                for det in detections:
                    det['score'] = self.score_detection(det, img.shape)
                    det['preprocess'] = preprocess_name
                    all_detections.append(det)
                
                if detections:
                    print(f"  Found {len(detections)} text regions")
        
        if not all_detections:
            return PriceResult(
                ok=False,
                error="No text detected by any OCR engine",
                all_detections=[]
            )
        
        # Sort by score
        all_detections.sort(key=lambda x: x['score'], reverse=True)
        
        # Debug output
        if self.debug:
            self._save_debug_visualization(img, all_detections)
        
        # Try to parse price from top candidates
        for det in all_detections[:10]:
            price = self.parse_price(det['text'])
            if price is not None:
                print(f"[SUCCESS] Found price: {price} (text='{det['text']}', "
                      f"engine={det['engine']}, score={det['score']:.2f})")
                return PriceResult(
                    ok=True,
                    price=price,
                    raw_text=det['text'],
                    confidence=det['confidence'],
                    bbox=det['bbox'],
                    engine=det['engine'],
                    all_detections=all_detections[:20]
                )
        
        # No valid price found
        return PriceResult(
            ok=False,
            error="No valid price pattern found in detected text",
            all_detections=all_detections[:20]
        )

    def _save_debug_visualization(self, img: np.ndarray, detections: List[Dict]):
        """Save debug image with all detections visualized"""
        debug_img = img.copy()
        
        for i, det in enumerate(detections[:15]):
            x, y, w, h = det['bbox']
            
            # Color based on rank (green for top, fading to red)
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
    """
    Convenience function for backward compatibility.
    
    Args:
        image_path: Path to image
        engine: "paddle", "easy", "tesseract", or "all"
        debug: Whether to save debug images
        
    Returns:
        Dictionary with extraction results
    """
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
    
    parser = argparse.ArgumentParser(description="Extract gas prices from images")
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("--engine", choices=["paddle", "easy", "tesseract", "all"],
                       default="all", help="OCR engine to use (default: all)")
    parser.add_argument("--debug", action="store_true", 
                       help="Save debug images")
    
    args = parser.parse_args()
    
    result = extract_price(args.image_path, engine=args.engine, debug=args.debug)
    
    print("\n" + "=" * 60)
    print("EXTRACTION RESULT:")
    print("=" * 60)
    
    if result.get('ok'):
        print(f"  Price: ${result['price']}")
        print(f"  Raw Text: '{result.get('raw_text', 'N/A')}'")
        print(f"  Confidence: {result.get('confidence', 0):.2%}")
        print(f"  Engine: {result.get('engine', 'N/A')}")
        print(f"  Bounding Box: {result.get('bbox', 'N/A')}")
    else:
        print(f"  Error: {result.get('error', 'Unknown error')}")
    
    if result.get('all_detections'):
        print(f"\n  Top detections ({len(result['all_detections'])} total):")
        for i, det in enumerate(result['all_detections'][:5]):
            print(f"    {i+1}. '{det['text']}' (score={det['score']:.1f}, "
                  f"engine={det['engine']}, conf={det['confidence']:.2f})")
    
    print("=" * 60)
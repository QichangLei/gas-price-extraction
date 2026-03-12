# Annotation Video — Improvement Points

## 1. Validation Threshold Too Strict (presence filter)
**Problem:** The current validated prices filter requires ≥5% frame presence AND ≥60% agreement. If a gas price sign is only clearly visible for a brief moment (few frames) but with high confidence, it gets dropped.

**Example:** IMG_0977 — TA/TA Express brand clearly detected, prices extracted with high confidence on some frames, but failed the presence threshold → output shows "No prices passed validation".

**Proposed fix:** If a price is detected with high confidence (e.g. ≥85%) even on a small number of frames, it should still be reported. Consider separating the validation into:
- High-confidence detections (≥85%): always keep, regardless of presence %
- Low-confidence detections: apply the current presence + agreement filter

---

## 2. Cash vs Credit Price Separation
**Problem:** The same gas station sign may display both Cash and Credit prices in different physical locations on the sign. These could appear in different frames or different regions of the same frame, and currently may be conflated or missed.

**Example:** A sign shows "Cash 3.199 / Credit 3.299" — if the camera captures them in separate frames or if only one is in view, the pairing is lost.

**Proposed fix:**
- Track `Payment_Type` (Cash/Credit) per detection and keep them as separate rows
- If both are detected across frames at the same station (same location/timestamp window), output both rows explicitly rather than merging or dropping one
- Consider spatial reasoning: if bounding boxes are at different Y positions on the sign, treat them as distinct price entries


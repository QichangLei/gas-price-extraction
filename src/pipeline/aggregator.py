"""
aggregator.py
─────────────
Pluggable strategies for aggregating per-frame price detections into a
clean station-level summary.

The prompt and extractor work on individual frames — this module decides
how to combine those per-frame results into a final price list.

Available strategies
--------------------
  modal             Most common price across all frames per (grade, payment).
                    Robust to a few bad frames. Good general-purpose default.

  highest_confidence  Price from the single highest-confidence frame per
                    (grade, payment). Trusts Gemini's self-reported clarity.

  consensus         Only output a price if ≥ threshold % of frames agree.
                    Conservative — reduces hallucinations at the cost of
                    possibly missing grades seen in few frames.
                    Set threshold with --consensus-threshold (default 0.5).

  best_frame        Find the one frame with the highest average confidence,
                    then use only that frame's prices. Fast but brittle if
                    the best frame is partially occluded.

Adding a new strategy
---------------------
1. Subclass AggregationStrategy and implement aggregate().
2. Register it in STRATEGIES at the bottom of this file.
3. That's it — it will automatically appear in --strategy choices.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import Counter, defaultdict

log = logging.getLogger(__name__)

# ── Grade / payment ordering for output sorting ───────────────────────────────

_GRADE_ORDER   = {"Regular": 0, "Mid-Grade": 1, "Premium": 2,
                  "Diesel": 3, "E85": 4, "Unknown": 5}
_PAYMENT_ORDER = {"Cash": 0, "Credit": 1, "Both": 2, "NA": 3}


def _sort_key(row: dict) -> tuple:
    return (
        _GRADE_ORDER.get(row["Fuel_Type"], 99),
        _PAYMENT_ORDER.get(row["Payment_Type"], 99),
    )


# ── Base class ────────────────────────────────────────────────────────────────

class AggregationStrategy(ABC):
    """
    Base class for frame aggregation strategies.

    Input  (to aggregate): list of dicts, one per valid extracted row:
        grade       str   normalised grade label  (e.g. "Regular")
        price       float
        payment     str   "Cash" / "Credit" / "NA"
        brand       str
        confidence  int
        frame_idx   int   image_number from the raw CSV

    Output (from aggregate): list of dicts, one per unique price entry:
        Gas_Station_Brand  str
        Fuel_Type          str
        Price              float
        Payment_Type       str
        Max_Confidence     int
        Frames_Detected    int
        Consistency_Pct    float
    """

    name:        str = ""
    description: str = ""

    @abstractmethod
    def aggregate(self, rows: list[dict]) -> list[dict]:
        ...

    # ── Shared helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _majority_brand(rows: list[dict]) -> str:
        brands = [r["brand"] for r in rows if r["brand"] not in ("NA", "")]
        return Counter(brands).most_common(1)[0][0] if brands else "NA"

    @staticmethod
    def _make_row(brand, grade, price, payment, best_conf,
                  n_frames, n_modal) -> dict:
        return {
            "Gas_Station_Brand": brand,
            "Fuel_Type":         grade,
            "Price":             price,
            "Payment_Type":      payment,
            "Max_Confidence":    best_conf,
            "Frames_Detected":   n_frames,
            "Consistency_Pct":   round(n_modal / n_frames * 100, 1),
        }

    @staticmethod
    def _group_by_grade_payment(rows: list[dict]) -> dict[tuple, list[dict]]:
        """
        Group rows by (grade, payment).
        Exception: Unknown-grade rows are grouped by (grade, price, payment)
        so that each distinct price is preserved as a separate entry rather
        than being collapsed into one modal value.
        """
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for r in rows:
            if r["grade"] == "Unknown":
                groups[("Unknown", round(r["price"], 3), r["payment"])].append(r)
            else:
                groups[(r["grade"], r["payment"])].append(r)
        return groups


# ── Strategy: modal ───────────────────────────────────────────────────────────

class ModalStrategy(AggregationStrategy):
    """
    Per (grade, payment): pick the most common price across all frames.
    Tie-break: highest confidence.

    Best for: general use. Robust to occasional misread frames.
    """
    name        = "modal"
    description = "Most common price per grade/payment across all frames (default)."

    def aggregate(self, rows: list[dict]) -> list[dict]:
        brand   = self._majority_brand(rows)
        groups  = self._group_by_grade_payment(rows)
        results = []

        for key, entries in groups.items():
            grade, payment        = key[0], key[-1]
            price_counts          = Counter(e["price"] for e in entries)
            modal_price, n_modal  = price_counts.most_common(1)[0]
            modal_entries         = [e for e in entries if e["price"] == modal_price]
            best_conf             = max(e["confidence"] for e in modal_entries)
            results.append(self._make_row(
                brand, grade, modal_price, payment,
                best_conf, len(entries), n_modal,
            ))

        return sorted(results, key=_sort_key)


# ── Strategy: highest_confidence ─────────────────────────────────────────────

class HighestConfidenceStrategy(AggregationStrategy):
    """
    Per (grade, payment): use the price from the single highest-confidence frame.

    Best for: clean, well-lit footage where Gemini's confidence scores are
    reliable. May pick an outlier if one frame has inflated confidence.
    """
    name        = "highest_confidence"
    description = "Price from the highest-confidence frame per grade/payment."

    def aggregate(self, rows: list[dict]) -> list[dict]:
        brand   = self._majority_brand(rows)
        groups  = self._group_by_grade_payment(rows)
        results = []

        for key, entries in groups.items():
            grade, payment = key[0], key[-1]
            best       = max(entries, key=lambda e: e["confidence"])
            chosen_p   = best["price"]
            n_agree    = sum(1 for e in entries if e["price"] == chosen_p)
            results.append(self._make_row(
                brand, grade, chosen_p, payment,
                best["confidence"], len(entries), n_agree,
            ))

        return sorted(results, key=_sort_key)


# ── Strategy: consensus ───────────────────────────────────────────────────────

class ConsensusStrategy(AggregationStrategy):
    """
    Per (grade, payment): only output a price if it appears in ≥ threshold
    fraction of frames. Grades that don't reach consensus are omitted.

    Best for: noisy video with many bad frames. Reduces hallucinations.
    Set threshold (0.0–1.0) at construction time.
    """
    name        = "consensus"
    description = "Only output prices seen in ≥ threshold% of frames (default 50%)."

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def aggregate(self, rows: list[dict]) -> list[dict]:
        brand   = self._majority_brand(rows)
        groups  = self._group_by_grade_payment(rows)
        results = []

        for key, entries in groups.items():
            grade, payment       = key[0], key[-1]
            price_counts         = Counter(e["price"] for e in entries)
            modal_price, n_modal = price_counts.most_common(1)[0]
            ratio = n_modal / len(entries)

            if ratio < self.threshold:
                log.info(
                    "  [consensus] Skipping %s/%s — best price %.3f seen in "
                    "%.0f%% of frames (threshold %.0f%%)",
                    grade, payment, modal_price,
                    ratio * 100, self.threshold * 100,
                )
                continue

            modal_entries = [e for e in entries if e["price"] == modal_price]
            best_conf     = max(e["confidence"] for e in modal_entries)
            results.append(self._make_row(
                brand, grade, modal_price, payment,
                best_conf, len(entries), n_modal,
            ))

        return sorted(results, key=_sort_key)


# ── Strategy: best_frame ──────────────────────────────────────────────────────

class BestFrameStrategy(AggregationStrategy):
    """
    Find the single frame with the highest average confidence across all its
    detected prices, then use only that frame's prices.

    Best for: short clips where you trust one clear frame more than averaging.
    Brittle if the clearest frame happens to be partially occluded.
    """
    name        = "best_frame"
    description = "Use prices from the single highest-average-confidence frame only."

    def aggregate(self, rows: list[dict]) -> list[dict]:
        brand = self._majority_brand(rows)

        # Group by frame
        frame_groups: dict[int, list[dict]] = defaultdict(list)
        for r in rows:
            frame_groups[r["frame_idx"]].append(r)

        # Pick frame with highest mean confidence
        best_frame_idx = max(
            frame_groups,
            key=lambda idx: sum(e["confidence"] for e in frame_groups[idx])
                            / len(frame_groups[idx])
        )
        best_entries = frame_groups[best_frame_idx]
        log.info("  [best_frame] Using frame %d (%d entries, avg conf %.1f)",
                 best_frame_idx,
                 len(best_entries),
                 sum(e["confidence"] for e in best_entries) / len(best_entries))

        results = []
        for entry in best_entries:
            results.append(self._make_row(
                brand,
                entry["grade"],
                entry["price"],
                entry["payment"],
                entry["confidence"],
                1,   # only one frame
                1,
            ))

        return sorted(results, key=_sort_key)


# ── Registry ──────────────────────────────────────────────────────────────────

STRATEGIES: dict[str, AggregationStrategy] = {
    ModalStrategy.name:             ModalStrategy(),
    HighestConfidenceStrategy.name: HighestConfidenceStrategy(),
    ConsensusStrategy.name:         ConsensusStrategy(),
    BestFrameStrategy.name:         BestFrameStrategy(),
}


def get_strategy(name: str, **kwargs) -> AggregationStrategy:
    """
    Return a strategy instance by name.
    Pass extra kwargs for strategies that have parameters
    (e.g. threshold for 'consensus').
    """
    if name not in STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{name}'. "
            f"Available: {list(STRATEGIES.keys())}"
        )
    # Reinstantiate with kwargs so parameters (e.g. threshold) take effect
    cls = type(STRATEGIES[name])
    return cls(**kwargs) if kwargs else STRATEGIES[name]


def list_strategies() -> str:
    lines = ["Available aggregation strategies:", ""]
    for s in STRATEGIES.values():
        lines.append(f"  {s.name:<22}  {s.description}")
    return "\n".join(lines)

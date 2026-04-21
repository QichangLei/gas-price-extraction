"""
Label Frame Extractor
======================
Helps manually label video ground truth by finding and enhancing the
clearest frames from each video so prices on gas station signs can be read.

For each video:
  1. Scores every frame by sharpness (Laplacian variance — blurry = low score)
  2. Picks the top-N sharpest frames (spread evenly across the video)
  3. Saves each frame as:
       original.jpg  — unmodified
       enhanced.jpg  — contrast + sharpening applied
  4. Saves a contact sheet (grid) showing all candidates at a glance

Output goes to:  output/label_frames/<video_stem>/

Usage
-----
    # One video:
    python src/utils/extract_label_frames.py \\
        --video "data/videos/Gas Videos/IMG_0966.MOV"

    # All videos in a folder:
    python src/utils/extract_label_frames.py \\
        --folder "data/videos/Gas Videos"

    # Keep more candidate frames:
    python src/utils/extract_label_frames.py \\
        --video "data/videos/Gas Videos/IMG_0966.MOV" --top 10
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


# ---------------------------------------------------------------------------
# Sharpness scoring
# ---------------------------------------------------------------------------

def sharpness_score(gray_frame: np.ndarray) -> float:
    """Laplacian variance — higher = sharper."""
    return float(cv2.Laplacian(gray_frame, cv2.CV_64F).var())


# ---------------------------------------------------------------------------
# Image enhancement
# ---------------------------------------------------------------------------

def enhance_frame(bgr: np.ndarray) -> np.ndarray:
    """
    Apply a stack of enhancements designed to make gas price text legible:
      1. CLAHE on the L channel  → improves local contrast / handles glare
      2. Unsharp mask             → sharpens edges
      3. Mild saturation boost    → helps colour-coded grade labels stand out
    """
    # 1. CLAHE in LAB colour space
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 2. Unsharp mask  (blend of original and high-pass)
    blurred = cv2.GaussianBlur(bgr, (0, 0), sigmaX=3)
    bgr = cv2.addWeighted(bgr, 1.5, blurred, -0.5, 0)

    # 3. Mild saturation boost via PIL
    pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    pil = ImageEnhance.Color(pil).enhance(1.3)
    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    return bgr


# ---------------------------------------------------------------------------
# Contact sheet
# ---------------------------------------------------------------------------

def make_contact_sheet(images: list[np.ndarray],
                       labels: list[str],
                       cols: int = 3) -> np.ndarray:
    """
    Arrange a list of BGR images into a grid.
    Each image gets its label drawn at the top.
    All images are resized to a common height.
    """
    target_h = 600
    cells = []
    for img, label in zip(images, labels):
        h, w = img.shape[:2]
        new_w = int(w * target_h / h)
        cell = cv2.resize(img, (new_w, target_h))
        # Draw label bar
        bar = np.zeros((40, new_w, 3), dtype=np.uint8)
        cv2.putText(bar, label, (5, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cell = np.vstack([bar, cell])
        cells.append(cell)

    # Pad to a common width within each row
    rows = [cells[i:i + cols] for i in range(0, len(cells), cols)]
    row_imgs = []
    for row in rows:
        max_w = max(c.shape[1] for c in row)
        padded = []
        for c in row:
            if c.shape[1] < max_w:
                pad = np.zeros((c.shape[0], max_w - c.shape[1], 3), dtype=np.uint8)
                c = np.hstack([c, pad])
            padded.append(c)
        # Pad row to full cols width
        while len(padded) < cols:
            h = padded[0].shape[0]
            padded.append(np.zeros((h, padded[0].shape[1], 3), dtype=np.uint8))
        row_imgs.append(np.hstack(padded))

    return np.vstack(row_imgs)


# ---------------------------------------------------------------------------
# Core: process one video
# ---------------------------------------------------------------------------

def process_video(video_path: Path, top_n: int, out_root: Path) -> None:
    stem = video_path.stem
    out_dir = out_root / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open {video_path}")
        return

    vid_fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    n_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w         = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h         = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration  = n_frames / vid_fps

    print(f"\n  {video_path.name}  {w}x{h}  {duration:.1f}s  ({n_frames} frames @ {vid_fps:.0f}fps)")

    # --- Score every frame ---
    scores = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scores.append((idx, sharpness_score(gray)))
        idx += 1
    cap.release()

    if not scores:
        print("  [WARN] No frames read.")
        return

    # --- Pick top-N, spread across video to avoid picking the same moment ---
    # Divide video into top_n equal segments; pick the sharpest frame in each
    segment_size = max(1, len(scores) // top_n)
    candidates = []
    for seg in range(top_n):
        start = seg * segment_size
        end   = min(start + segment_size, len(scores))
        seg_scores = scores[start:end]
        best = max(seg_scores, key=lambda x: x[1])
        candidates.append(best)  # (frame_idx, score)

    # Also grab the global sharpest frame if not already included
    global_best = max(scores, key=lambda x: x[1])
    if global_best[0] not in {c[0] for c in candidates}:
        candidates.append(global_best)

    candidates.sort(key=lambda x: x[0])  # chronological order

    print(f"  Selected {len(candidates)} candidate frames  "
          f"(sharpness range: {min(s for _, s in candidates):.0f} – "
          f"{max(s for _, s in candidates):.0f})")

    # --- Re-read selected frames and save ---
    cap = cv2.VideoCapture(str(video_path))
    frame_buffer: dict[int, np.ndarray] = {}
    idx = 0
    needed = {c[0] for c in candidates}
    while needed:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in needed:
            frame_buffer[idx] = frame
            needed.discard(idx)
        idx += 1
    cap.release()

    sheet_originals = []
    sheet_enhanced  = []
    sheet_labels    = []

    for rank, (frame_idx, score) in enumerate(candidates, 1):
        frame = frame_buffer.get(frame_idx)
        if frame is None:
            continue

        ts_sec = frame_idx / vid_fps
        label  = f"#{rank}  t={ts_sec:.1f}s  sharpness={score:.0f}"

        enhanced = enhance_frame(frame)

        orig_path = out_dir / f"frame_{rank:02d}_t{ts_sec:.1f}s_orig.jpg"
        enh_path  = out_dir / f"frame_{rank:02d}_t{ts_sec:.1f}s_enhanced.jpg"

        cv2.imwrite(str(orig_path),  frame,    [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(enh_path),   enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])

        sheet_originals.append(frame)
        sheet_enhanced.append(enhanced)
        sheet_labels.append(label)
        print(f"    Frame {rank:2d}: t={ts_sec:5.1f}s  sharpness={score:7.1f}  → {enh_path.name}")

    # --- Contact sheets ---
    if sheet_originals:
        cols = min(3, len(sheet_originals))
        orig_sheet = make_contact_sheet(sheet_originals, sheet_labels, cols=cols)
        enh_sheet  = make_contact_sheet(sheet_enhanced,  sheet_labels, cols=cols)

        cv2.imwrite(str(out_dir / "contact_sheet_original.jpg"),  orig_sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
        cv2.imwrite(str(out_dir / "contact_sheet_enhanced.jpg"),  enh_sheet,  [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"  Contact sheets saved → {out_dir}/contact_sheet_*.jpg")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract and enhance the sharpest frames from gas station videos for labeling.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video",  help="Path to a single video file.")
    group.add_argument("--folder", help="Folder containing video files (processes all).")
    parser.add_argument("--top",   type=int, default=6,
                        help="Number of candidate frames to extract per video (default 6).")
    parser.add_argument("--output", default="output/label_frames",
                        help="Root output directory (default: output/label_frames).")
    args = parser.parse_args()

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".mts"}

    if args.video:
        videos = [Path(args.video)]
    else:
        folder = Path(args.folder)
        videos = sorted(p for p in folder.iterdir() if p.suffix.lower() in VIDEO_EXTS)

    if not videos:
        print("[ERROR] No video files found.")
        sys.exit(1)

    print("=" * 60)
    print("  Label Frame Extractor")
    print("=" * 60)
    print(f"  Videos    : {len(videos)}")
    print(f"  Top-N     : {args.top} frames per video")
    print(f"  Output    : {out_root}/")

    for vp in videos:
        process_video(vp, top_n=args.top, out_root=out_root)

    print(f"\nDone. Open output/label_frames/ to browse frames.")
    print("Fill prices into:  data/videos/labels/ground_truth.csv")


if __name__ == "__main__":
    main()

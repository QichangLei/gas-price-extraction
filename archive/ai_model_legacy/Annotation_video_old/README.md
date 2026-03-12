# Annotation_video

Annotates a dashcam/gas station video using Gemini OCR. For each sampled frame, it detects the gas price sign, draws a bounding box, and overlays the extracted prices. Outputs an annotated video, a prices CSV, and a summary image.

## Folder structure

```
Annotation_video/
├── annotate_video.py   # main script
├── results/            # output from each run (videos, CSVs, summary images)
└── wasted/             # old prototypes (ignore)
```

## Usage

```bash
conda activate gas_price_cv
cd /media/qichang/Data/research/Gas_price/AI_model/Gemini/Annotation_video
```

**Basic — auto-named output:**
```bash
python3 annotate_video.py --input /path/to/video.mp4
```

**Custom output name:**
```bash
python3 annotate_video.py --input /path/to/video.mp4 --output results/my_run.mp4
```

**Control sampling rate (call Gemini every N frames):**
```bash
python3 annotate_video.py --input /path/to/video.mp4 --every 10
```

**Limit number of frames processed (for quick tests):**
```bash
python3 annotate_video.py --input /path/to/video.mp4 --frames 100
```

## Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | *(required)* | Path to input video file |
| `--output` | auto-generated | Path to annotated output video (`.mp4`) |
| `--every` | 5 | Call Gemini every N frames (higher = faster & cheaper) |
| `--frames` | all | Stop after processing this many frames |

## Outputs (saved alongside the output video)

| File | Description |
|------|-------------|
| `<name>.mp4` | Annotated video with bounding boxes and price overlays |
| `<name>_prices.csv` | Extracted prices per frame (Fuel_Type, Price, Brand, Payment_Type, Confidence) |
| `<name>_summary.png` | Summary chart of prices detected across the video |

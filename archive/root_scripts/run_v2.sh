#!/bin/bash

# Batch Gas Price Extraction Script
# This script processes all images in a directory using the Python OCR script

# Check if directory path is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <directory_path> [output_directory]"
    echo ""
    echo "Examples:"
    echo "  $0 /data/price_1/"
    echo "  $0 /data/price_1/ ./results"
    exit 1
fi

# Get directory path from first argument
IMAGE_DIR="$1"

# Check if directory exists
if [ ! -d "$IMAGE_DIR" ]; then
    echo "Error: Directory does not exist: $IMAGE_DIR"
    exit 1
fi

# Get output directory (optional second argument)
if [ $# -ge 2 ]; then
    OUTPUT_DIR="$2"
    echo "Processing images in: $IMAGE_DIR"
    echo "Saving results to: $OUTPUT_DIR"
    python3 v2_batch.py "$IMAGE_DIR" --output-dir "$OUTPUT_DIR" --format txt
else
    echo "Processing images in: $IMAGE_DIR"
    python3 v2_batch.py "$IMAGE_DIR"
fi

echo ""
echo "Done!"

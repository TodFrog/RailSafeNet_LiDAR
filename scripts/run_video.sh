#!/bin/bash
# RailSafeNet Final - Video Mode Runner
# Usage: ./scripts/run_video.sh [options]
#
# Options:
#   --fullscreen    Run in fullscreen mode
#   --video PATH    Specific video file path

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Change to project directory
cd "$PROJECT_DIR"

# Parse arguments
FULLSCREEN=""
VIDEO=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --fullscreen)
            FULLSCREEN="--fullscreen"
            shift
            ;;
        --video)
            VIDEO="--video $2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "RailSafeNet Final - Video Mode"
echo "========================================"
echo "Project: $PROJECT_DIR"
echo ""

# Check if running in Docker or native
if [[ -f /.dockerenv ]]; then
    echo "Running in Docker container..."
    python3 videoAssessor_final.py --mode video $FULLSCREEN $VIDEO
else
    echo "Running natively..."

    # Activate conda environment if available
    if command -v conda &> /dev/null; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate RailSafeNet 2>/dev/null || conda activate rail 2>/dev/null || true
    fi

    python3 videoAssessor_final.py --mode video $FULLSCREEN $VIDEO
fi

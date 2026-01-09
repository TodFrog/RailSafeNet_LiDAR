#!/bin/bash
# RailSafeNet Final - Camera Mode Runner
# Usage: ./scripts/run_camera.sh [options]
#
# Options:
#   --camera ID     Camera device ID (default: 0)
#   --no-fullscreen Disable fullscreen mode

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Change to project directory
cd "$PROJECT_DIR"

# Parse arguments
CAMERA="0"
FULLSCREEN="--fullscreen"

while [[ $# -gt 0 ]]; do
    case $1 in
        --camera)
            CAMERA="$2"
            shift 2
            ;;
        --no-fullscreen)
            FULLSCREEN=""
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "RailSafeNet Final - Camera Mode"
echo "========================================"
echo "Project: $PROJECT_DIR"
echo "Camera: /dev/video$CAMERA"
echo ""

# Check if camera exists
if [[ ! -e "/dev/video$CAMERA" ]]; then
    echo "Warning: /dev/video$CAMERA not found"
    echo "Available cameras:"
    ls -la /dev/video* 2>/dev/null || echo "  No cameras found"
fi

# Check if running in Docker or native
if [[ -f /.dockerenv ]]; then
    echo "Running in Docker container..."
    python3 videoAssessor_final.py --mode camera --camera "$CAMERA" $FULLSCREEN
else
    echo "Running natively..."

    # Activate conda environment if available
    if command -v conda &> /dev/null; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate RailSafeNet 2>/dev/null || conda activate rail 2>/dev/null || true
    fi

    python3 videoAssessor_final.py --mode camera --camera "$CAMERA" $FULLSCREEN
fi

#!/bin/bash
# RailSafeNet Final - Docker Build Script
# Usage: ./scripts/build_docker.sh [options]
#
# Options:
#   --no-cache    Build without cache
#   --push        Push to registry after build

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Change to project directory
cd "$PROJECT_DIR"

# Image name
IMAGE_NAME="railsafenet-final"
IMAGE_TAG="latest"

# Parse arguments
NO_CACHE=""
PUSH=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        --push)
            PUSH=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "RailSafeNet Final - Docker Build"
echo "========================================"
echo "Image: $IMAGE_NAME:$IMAGE_TAG"
echo "Context: $PROJECT_DIR"
echo ""

# Build the image
echo "Building Docker image..."
docker build \
    -t "$IMAGE_NAME:$IMAGE_TAG" \
    -f docker/Dockerfile \
    $NO_CACHE \
    .

echo ""
echo "Build complete!"
echo ""

# Show image info
docker images "$IMAGE_NAME:$IMAGE_TAG"

# Push if requested
if [[ "$PUSH" == true ]]; then
    echo ""
    echo "Pushing to registry..."
    docker push "$IMAGE_NAME:$IMAGE_TAG"
    echo "Push complete!"
fi

echo ""
echo "========================================"
echo "Quick Start Commands:"
echo "========================================"
echo ""
echo "=== Native (Development) ==="
echo "  Video mode:    python3 videoAssessor_final.py --mode video"
echo "  Camera mode:   python3 videoAssessor_final.py --mode camera --fullscreen"
echo "  VP Calibrate:  python3 videoAssessor_final.py --calibrate-vp"
echo "  Process video: python3 videoAssessor_final.py --video input.mp4 --output out.mp4"
echo ""
echo "=== Docker (Production) ==="
echo "  # Video mode (display)"
echo "  cd docker && docker-compose up railsafenet-video"
echo ""
echo "  # Camera mode (fullscreen)"
echo "  cd docker && docker-compose up railsafenet-camera"
echo ""
echo "  # VP Calibration"
echo "  cd docker && docker-compose up railsafenet-calibrate-vp"
echo ""
echo "  # Process & save video (headless)"
echo "  cd docker && docker-compose run railsafenet-process"
echo ""
echo "  # Custom video processing"
echo "  cd docker && docker-compose run railsafenet-process \\"
echo "      --video /app/assets/crop/input.mp4 \\"
echo "      --output /app/output/result.mp4 \\"
echo "      --start-time 30"
echo ""
echo "  # Development shell"
echo "  cd docker && docker-compose run railsafenet-dev bash"
echo ""

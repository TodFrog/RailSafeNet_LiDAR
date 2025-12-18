#!/bin/bash

# Build the Docker image
echo "Building Docker image..."
docker build -t railsafenet-jetson .

# Allow X11 forwarding
xhost +local:docker

# Run the container
# -v /tmp/.X11-unix:/tmp/.X11-unix: Maps X11 socket for GUI
# -e DISPLAY=$DISPLAY: Passes display environment variable
# --runtime nvidia: Enables GPU access
# -v /home/mmc-server4/Server/Users/minkyu/RailSafeNet_LiDAR/assets:/app/assets: Mounts assets (adjust source path as needed)
# --network host: Useful for some camera streams or ROS communication

echo "Running RailSafeNet App..."
docker run -it --rm \
    --runtime nvidia \
    --network host \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(pwd)/../assets:/app/assets \
    railsafenet-jetson

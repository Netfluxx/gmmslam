#!/usr/bin/env bash
# Launch gmmslam inside the ROS2 Humble container.
# Usage: ./run_slam.sh [ros2_launch_args...]
#   e.g. ./run_slam.sh lidar_topic:=/my/lidar rviz:=true
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

IMAGE="gmmslam:humble"
CONTAINER="gmmslam_humble"
COLCON_WS="/workspaces/gmmslam"

# ---------------------------------------------------------------------------
# Build image if it doesn't exist yet
# ---------------------------------------------------------------------------
if ! docker image inspect "$IMAGE" &>/dev/null; then
    echo "[run_slam] Image $IMAGE not found — building..."
    docker build -t "$IMAGE" -f "$SCRIPT_DIR/Dockerfile.humble" "$REPO_ROOT"
fi

# ---------------------------------------------------------------------------
# Start container if not already running
# ---------------------------------------------------------------------------
if ! docker ps --filter "name=^${CONTAINER}$" --filter "status=running" -q | grep -q .; then
    # Remove a leftover stopped container with the same name, if any
    docker rm -f "$CONTAINER" &>/dev/null || true

    echo "[run_slam] Starting container $CONTAINER..."
    docker run -d \
        --name "$CONTAINER" \
        --network host \
        -e "DISPLAY=${DISPLAY:-}" \
        -e "DEBIAN_FRONTEND=noninteractive" \
        -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
        -v "$REPO_ROOT:$COLCON_WS:rw" \
        -v "gmmslam_humble_home:/root" \
        "$IMAGE" \
        tail -f /dev/null
fi

# ---------------------------------------------------------------------------
# Build gmmslam with colcon (skipped if the node binary already exists)
# ---------------------------------------------------------------------------
docker exec "$CONTAINER" bash -lc "
    set -e
    source /opt/ros/humble/setup.bash
    cd $COLCON_WS
    if [ ! -f install/gmmslam/lib/gmmslam/gmmslam_node ]; then
        echo '[run_slam] Building gmmslam...'
        colcon build --packages-select gmmslam \
            --cmake-args -DCMAKE_BUILD_TYPE=Release
    fi
"

# ---------------------------------------------------------------------------
# Launch SLAM (pass any extra args through to ros2 launch)
# ---------------------------------------------------------------------------
echo "[run_slam] Launching gmmslam..."
docker exec -it "$CONTAINER" bash -lc "
    source /opt/ros/humble/setup.bash
    source $COLCON_WS/install/setup.bash
    exec ros2 launch gmmslam gmmslam.launch.py $(printf '%q ' "$@")
"

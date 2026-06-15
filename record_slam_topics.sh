#!/usr/bin/env bash
# Record the topics needed for a SLAM loop closure GIF.
# Runs inside the disal_slam Docker container (where ROS lives).
# Usage: ./record_slam_topics.sh [output_bag_name]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER="disal_slam"
ROS_DISTRO="${ROS_DISTRO:-noetic}"
CATKIN_WS="${CATKIN_WS:-/root/catkin_ws}"

BAG_NAME="${1:-slam_demo_$(date +%Y%m%d_%H%M%S).bag}"

# If not inside the container, re-exec inside it
if [[ ! -f /.dockerenv ]]; then
    echo "[record] Entering Docker container $CONTAINER ..."
    exec docker exec -it "${CONTAINER}" bash -lc "
        source /opt/ros/${ROS_DISTRO}/setup.bash
        if [[ -f ${CATKIN_WS}/devel/setup.bash ]]; then
            source ${CATKIN_WS}/devel/setup.bash
        fi
        cd /root/catkin_ws/src/gmmslam
        echo 'Recording SLAM demo topics → ${BAG_NAME}'
        echo 'Press Ctrl+C to stop.'
        rosbag record \\
          /gmmslam_node/global_graph_path \\
          /gmmslam_node/path \\
          /gmmslam_node/gt_path \\
          /gmmslam_node/loop_closure_markers \\
          /gmmslam_node/graph_nodes \\
          /gmmslam_node/map_cloud \\
          -O logs/${BAG_NAME}
    "
fi

# Already inside container
source "/opt/ros/${ROS_DISTRO}/setup.bash"
[[ -f "${CATKIN_WS}/devel/setup.bash" ]] && source "${CATKIN_WS}/devel/setup.bash"

echo "Recording SLAM demo topics → logs/$BAG_NAME"
echo "Press Ctrl+C to stop."

rosbag record \
  /gmmslam_node/global_graph_path \
  /gmmslam_node/path \
  /gmmslam_node/gt_path \
  /gmmslam_node/loop_closure_markers \
  /gmmslam_node/graph_nodes \
  /gmmslam_node/map_cloud \
  -O "logs/$BAG_NAME"

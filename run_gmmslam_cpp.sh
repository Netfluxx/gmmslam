#!/usr/bin/env bash
set -euo pipefail

# ─── Paths ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CATKIN_WS="${CATKIN_WS:-/root/catkin_ws}"
GIRA_WS="${GIRA_WS:-/root/gira_ws}"
ROS_DISTRO="${ROS_DISTRO:-noetic}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
GMMSLAM_USE_GMMAP="${GMMSLAM_USE_GMMAP:-ON}"

# If launched from the host, enter the project Docker container first.  The
# container bind-mounts this repo at /root/catkin_ws/src/gmmslam; running on the
# host would otherwise try to write into /root/catkin_ws and fail.
if [[ "${GMMSLAM_CPP_IN_CONTAINER:-0}" != "1" && ! -f /.dockerenv ]]; then
    DOCKER_RUN="${SCRIPT_DIR}/docker/run.sh"
    CONTAINER_SCRIPT="/root/catkin_ws/src/gmmslam/run_gmmslam_cpp.sh"
    if [[ -x "${DOCKER_RUN}" ]]; then
        echo "[run_gmmslam_cpp] Entering Docker container ..."
        exec env GMMSLAM_CPP_IN_CONTAINER=1 \
            ROS_DISTRO="${ROS_DISTRO}" \
            BUILD_TYPE="${BUILD_TYPE}" \
            GMMSLAM_USE_GMMAP="${GMMSLAM_USE_GMMAP}" \
            "${DOCKER_RUN}" -- "${CONTAINER_SCRIPT}" "$@"
    fi
fi

# ─── Source ROS ─────────────────────────────────────────────────────
set +u
source "/opt/ros/${ROS_DISTRO}/setup.bash"
set -u

# ─── GIRA3D runtime library paths (needed by gmmslam_lib at runtime) ─
GIRA_REG="${GIRA_WS}/gira3d-registration"
GIRA_REC="${GIRA_WS}/gira3d-reconstruction"
for lib_dir in \
    "${GIRA_REG}/wet/install/lib" \
    "${GIRA_REG}/dry/install/lib" \
    "${GIRA_REC}/wet/install/lib" \
    "${GIRA_REC}/dry/install/lib"; do
    if [ -d "${lib_dir}" ]; then
        export LD_LIBRARY_PATH="${lib_dir}:${LD_LIBRARY_PATH:-}"
    fi
done

# ─── Symlink gmmslam into catkin workspace if not already there ─────
LINK_TARGET="${CATKIN_WS}/src/gmmslam"
if [ ! -e "${LINK_TARGET}" ]; then
    echo "[run_gmmslam_cpp] Symlinking ${SCRIPT_DIR} → ${LINK_TARGET}"
    ln -sfn "${SCRIPT_DIR}" "${LINK_TARGET}"
fi

# ─── Build (skip with --no-build) ──────────────────────────────────
SKIP_BUILD=false
LAUNCH_RVIZ=false
LAUNCH_ARGS=()

for arg in "$@"; do
    case "${arg}" in
        --no-build) SKIP_BUILD=true ;;
        --rviz)     LAUNCH_RVIZ=true ;;
        *)          LAUNCH_ARGS+=("${arg}") ;;
    esac
done

if [ "${SKIP_BUILD}" = false ]; then
    echo "[run_gmmslam_cpp] Building catkin workspace (${BUILD_TYPE}) ..."
    cd "${CATKIN_WS}"
    catkin_make -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
        -DGMMSLAM_USE_GMMAP="${GMMSLAM_USE_GMMAP}" \
        -j"$(nproc)" 2>&1
    echo "[run_gmmslam_cpp] Build complete."
fi

# ─── Source the workspace overlay ───────────────────────────────────
set +u
source "${CATKIN_WS}/devel/setup.bash"
set -u

# ─── Optionally launch RViz in the background ──────────────────────
if [ "${LAUNCH_RVIZ}" = true ]; then
    RVIZ_CFG="${SCRIPT_DIR}/config/gmmslam.rviz"
    if [ -f "${RVIZ_CFG}" ]; then
        echo "[run_gmmslam_cpp] Starting RViz ..."
        rosrun rviz rviz -d "${RVIZ_CFG}" >/tmp/gmmslam_rviz.log 2>&1 &
    else
        rosrun rviz rviz >/tmp/gmmslam_rviz.log 2>&1 &
    fi
fi

# ─── Launch ─────────────────────────────────────────────────────────
echo "[run_gmmslam_cpp] Launching gmmslam_cpp.launch ..."
exec roslaunch gmmslam gmmslam_cpp.launch "${LAUNCH_ARGS[@]}"

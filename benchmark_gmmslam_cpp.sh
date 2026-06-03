#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
LOG_FILE=""
BENCHMARK_LOG_DIR=""
ROS_DISTRO_TO_USE="noetic"
BUILD_TYPE="${BUILD_TYPE:-Release}"
SAMPLE_INTERVAL="1"
ENABLE_SAMPLING=true
DOCKER_MODE="auto"
PROCESS_PATTERN="gmmslam_node|d2d_registration_node|noisy_gt_publisher_node|run_gmmslam_cpp|benchmark_gmmslam_cpp|roslaunch|rosmaster|roscore|rosout|rosbag|rviz|webots|catkin_make|make|cmake|g\\+\\+|cc1plus"
ORIGINAL_ARGS=("$@")
RUN_ARGS=()

usage() {
    cat <<'EOF'
Usage:
  ./benchmark_gmmslam_cpp.sh [options] [roslaunch/run_gmmslam_cpp args...]

Options:
  --log-dir DIR           Directory for benchmark logs (default: ./logs)
  --log-file FILE         Exact txt log file path to write
  --benchmark-log-dir DIR Directory for structured CSV/TUM metrics
  --ros-distro DISTRO     ROS distribution for run_gmmslam_cpp.sh (default: noetic)
  --build-type TYPE       CMake build type for run_gmmslam_cpp.sh (default: Release)
  --docker                Run inside the project's Docker container
  --no-docker             Do not auto-run inside Docker
  --sample-interval SEC   CPU/RAM/GPU sampling interval (default: 1)
  --process-pattern REGEX Process regex sampled by ps
  --no-sampling           Disable time-series CPU/RAM/GPU sampling
  -h, --help              Show this help

Examples:
  ./benchmark_gmmslam_cpp.sh --no-build
  ./benchmark_gmmslam_cpp.sh --sample-interval 0.5 --no-build
  ./benchmark_gmmslam_cpp.sh --log-dir /tmp/gmmslam_logs --no-build config_file:=/path/to/params_bench.yaml

The script launches run_gmmslam_cpp.sh and writes stdout, stderr, and
/usr/bin/time -v resource usage into console.txt while also printing to screen.
It also writes time-series process/GPU metrics and structured ROS benchmark
metrics next to the main log.
EOF
}

sampler_loop() {
    local stop_file="$1"
    local process_file="$2"
    local gpu_file="$3"
    local gpu_process_file="$4"
    local start_epoch
    start_epoch="$(date +%s)"

    printf "timestamp_iso\telapsed_s\tpid\tppid\tcpu_percent\tmem_percent\trss_kb\tvsz_kb\tcomm\targs\n" > "${process_file}"
    printf "timestamp,index,name,gpu_util_percent,mem_util_percent,mem_used_mib,mem_total_mib,power_w,temp_c\n" > "${gpu_file}"
    printf "timestamp,gpu_uuid,pid,process_name,used_gpu_memory_mib\n" > "${gpu_process_file}"

    while [[ ! -f "${stop_file}" ]]; do
        local now_iso now_epoch elapsed
        now_iso="$(date --iso-8601=seconds)"
        now_epoch="$(date +%s)"
        elapsed="$((now_epoch - start_epoch))"

        ps -eo pid=,ppid=,pcpu=,pmem=,rss=,vsz=,comm=,args= \
            | awk -v ts="${now_iso}" -v elapsed="${elapsed}" -v pat="${PROCESS_PATTERN}" '
                $0 ~ pat {
                    args = $8
                    for (i = 9; i <= NF; ++i) {
                        args = args " " $i
                    }
                    gsub(/\t/, " ", args)
                    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", ts, elapsed, $1, $2, $3, $4, $5, $6, $7, args
                }
            ' >> "${process_file}"

        if command -v nvidia-smi >/dev/null 2>&1; then
            nvidia-smi \
                --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
                --format=csv,noheader,nounits >> "${gpu_file}" 2>/dev/null || true

            nvidia-smi \
                --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
                --format=csv,noheader,nounits 2>/dev/null \
                | awk -v ts="${now_iso}" '{ print ts ", " $0 }' >> "${gpu_process_file}" || true
        fi

        sleep "${SAMPLE_INTERVAL}"
    done
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --log-dir)
            if [[ $# -lt 2 ]]; then
                echo "error: --log-dir requires a directory" >&2
                exit 2
            fi
            LOG_DIR="$2"
            shift 2
            ;;
        --log-file)
            if [[ $# -lt 2 ]]; then
                echo "error: --log-file requires a file path" >&2
                exit 2
            fi
            LOG_FILE="$2"
            shift 2
            ;;
        --benchmark-log-dir)
            if [[ $# -lt 2 ]]; then
                echo "error: --benchmark-log-dir requires a directory" >&2
                exit 2
            fi
            BENCHMARK_LOG_DIR="$2"
            shift 2
            ;;
        --ros-distro)
            if [[ $# -lt 2 ]]; then
                echo "error: --ros-distro requires a ROS distribution name" >&2
                exit 2
            fi
            ROS_DISTRO_TO_USE="$2"
            shift 2
            ;;
        --build-type)
            if [[ $# -lt 2 ]]; then
                echo "error: --build-type requires a CMake build type" >&2
                exit 2
            fi
            BUILD_TYPE="$2"
            shift 2
            ;;
        --docker)
            DOCKER_MODE="always"
            shift
            ;;
        --no-docker)
            DOCKER_MODE="never"
            shift
            ;;
        --sample-interval)
            if [[ $# -lt 2 ]]; then
                echo "error: --sample-interval requires a value in seconds" >&2
                exit 2
            fi
            SAMPLE_INTERVAL="$2"
            shift 2
            ;;
        --process-pattern)
            if [[ $# -lt 2 ]]; then
                echo "error: --process-pattern requires a regex" >&2
                exit 2
            fi
            PROCESS_PATTERN="$2"
            shift 2
            ;;
        --no-sampling)
            ENABLE_SAMPLING=false
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            RUN_ARGS+=("$1")
            shift
            ;;
    esac
done

ROS_SETUP="/opt/ros/${ROS_DISTRO_TO_USE}/setup.bash"
if [[ "${GMMSLAM_BENCH_IN_CONTAINER:-0}" != "1" && "${DOCKER_MODE}" != "never" ]]; then
    if [[ "${DOCKER_MODE}" = "always" || ! -f "${ROS_SETUP}" ]]; then
        DOCKER_RUN="${SCRIPT_DIR}/docker/run.sh"
        CONTAINER_SCRIPT="/root/catkin_ws/src/gmmslam/benchmark_gmmslam_cpp.sh"
        if [[ ! -x "${DOCKER_RUN}" ]]; then
            echo "error: ${ROS_SETUP} does not exist and ${DOCKER_RUN} is not executable" >&2
            exit 1
        fi

        echo "[benchmark] ${ROS_SETUP} not found on host; running benchmark inside Docker container."
        exec "${DOCKER_RUN}" -- env GMMSLAM_BENCH_IN_CONTAINER=1 BUILD_TYPE="${BUILD_TYPE}" "${CONTAINER_SCRIPT}" --no-docker "${ORIGINAL_ARGS[@]}"
    fi
fi

if [[ ! -f "${ROS_SETUP}" ]]; then
    echo "error: ${ROS_SETUP} does not exist. Use --docker, install ROS ${ROS_DISTRO_TO_USE}, or pass --ros-distro <installed-distro>." >&2
    exit 1
fi

if [[ -z "${LOG_FILE}" ]]; then
    mkdir -p "${LOG_DIR}"
    TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
    BENCHMARK_LOG_DIR="${BENCHMARK_LOG_DIR:-${LOG_DIR}/gmmslam_cpp_${TIMESTAMP}}"
    mkdir -p "${BENCHMARK_LOG_DIR}"
    LOG_FILE="${BENCHMARK_LOG_DIR}/console.txt"
else
    mkdir -p "$(dirname "${LOG_FILE}")"
    if [[ -z "${BENCHMARK_LOG_DIR}" ]]; then
        LOG_BASE_FOR_DIR="${LOG_FILE%.txt}"
        BENCHMARK_LOG_DIR="${LOG_BASE_FOR_DIR}_artifacts"
    fi
    mkdir -p "${BENCHMARK_LOG_DIR}"
fi

LOG_BASE="${LOG_FILE%.txt}"
PROCESS_METRICS_FILE="${BENCHMARK_LOG_DIR}/processes.tsv"
GPU_METRICS_FILE="${BENCHMARK_LOG_DIR}/gpu.csv"
GPU_PROCESS_METRICS_FILE="${BENCHMARK_LOG_DIR}/gpu_processes.csv"
RUN_ARGS_FOR_LAUNCH=("${RUN_ARGS[@]}")
HAS_BENCHMARK_LOG_DIR_ARG=false
for arg in "${RUN_ARGS[@]}"; do
    if [[ "${arg}" == benchmark_log_dir:=* ]]; then
        HAS_BENCHMARK_LOG_DIR_ARG=true
        break
    fi
done
if [[ "${HAS_BENCHMARK_LOG_DIR_ARG}" = false ]]; then
    RUN_ARGS_FOR_LAUNCH+=("benchmark_log_dir:=${BENCHMARK_LOG_DIR}")
fi
SAMPLER_STOP_FILE="$(mktemp)"
rm -f "${SAMPLER_STOP_FILE}"
SAMPLER_PID=""

cleanup() {
    if [[ -n "${SAMPLER_PID}" ]]; then
        touch "${SAMPLER_STOP_FILE}" 2>/dev/null || true
        wait "${SAMPLER_PID}" 2>/dev/null || true
    fi
    rm -f "${SAMPLER_STOP_FILE}"
}
trap cleanup EXIT INT TERM

{
    echo "============================================================"
    echo "GMM-SLAM C++ benchmark log"
    echo "Started: $(date --iso-8601=seconds)"
    echo "Host: $(hostname)"
    echo "Workspace: ${SCRIPT_DIR}"
    echo "Command: ROS_DISTRO=${ROS_DISTRO_TO_USE} BUILD_TYPE=${BUILD_TYPE} ${SCRIPT_DIR}/run_gmmslam_cpp.sh ${RUN_ARGS_FOR_LAUNCH[*]}"
    echo "ROS distro: ${ROS_DISTRO_TO_USE}"
    echo "Build type: ${BUILD_TYPE}"
    echo "Log file: ${LOG_FILE}"
    echo "Benchmark metrics directory: ${BENCHMARK_LOG_DIR}"
    if [[ "${ENABLE_SAMPLING}" = true ]]; then
        echo "Process metrics: ${PROCESS_METRICS_FILE}"
        echo "GPU metrics: ${GPU_METRICS_FILE}"
        echo "GPU process metrics: ${GPU_PROCESS_METRICS_FILE}"
        echo "Sample interval: ${SAMPLE_INTERVAL}s"
        echo "Process pattern: ${PROCESS_PATTERN}"
    else
        echo "Time-series sampling: disabled"
    fi
    echo "============================================================"
} | tee "${LOG_FILE}"

if [[ "${ENABLE_SAMPLING}" = true ]]; then
    sampler_loop "${SAMPLER_STOP_FILE}" "${PROCESS_METRICS_FILE}" "${GPU_METRICS_FILE}" "${GPU_PROCESS_METRICS_FILE}" &
    SAMPLER_PID=$!
fi

set +e
if [[ -x /usr/bin/time ]]; then
    echo "Timer: /usr/bin/time -v" | tee -a "${LOG_FILE}"
    stdbuf -oL -eL /usr/bin/time -v env ROS_DISTRO="${ROS_DISTRO_TO_USE}" BUILD_TYPE="${BUILD_TYPE}" "${SCRIPT_DIR}/run_gmmslam_cpp.sh" "${RUN_ARGS_FOR_LAUNCH[@]}" > >(tee -a "${LOG_FILE}") 2>&1
    EXIT_CODE=$?
else
    echo "Timer: bash time fallback" | tee -a "${LOG_FILE}"
    TIMEFORMAT=$'real\t%3R\nuser\t%3U\nsys\t%3S'
    { time stdbuf -oL -eL env ROS_DISTRO="${ROS_DISTRO_TO_USE}" BUILD_TYPE="${BUILD_TYPE}" "${SCRIPT_DIR}/run_gmmslam_cpp.sh" "${RUN_ARGS_FOR_LAUNCH[@]}"; } > >(tee -a "${LOG_FILE}") 2>&1
    EXIT_CODE=$?
fi
set -e

cleanup
trap - EXIT INT TERM

{
    echo "============================================================"
    echo "Finished: $(date --iso-8601=seconds)"
    echo "Exit code: ${EXIT_CODE}"
    echo "============================================================"
} | tee -a "${LOG_FILE}"

echo "Benchmark log written to: ${LOG_FILE}"
echo "Structured benchmark metrics written to: ${BENCHMARK_LOG_DIR}"
if [[ "${ENABLE_SAMPLING}" = true ]]; then
    echo "Process metrics written to: ${PROCESS_METRICS_FILE}"
    echo "GPU metrics written to: ${GPU_METRICS_FILE}"
    echo "GPU process metrics written to: ${GPU_PROCESS_METRICS_FILE}"
fi
exit "${EXIT_CODE}"

#!/usr/bin/env bash
# Bring up the disal_slam container with X11 + GPU access and exec into it.
#
# Usage:
#   ./run.sh                # build (if needed), start, and exec into bash
#   ./run.sh --build        # force rebuild before starting
#   ./run.sh --no-cache     # rebuild from scratch
#   ./run.sh --down         # stop and remove the container
#   ./run.sh -- <cmd...>    # exec <cmd> in the container instead of bash
#
# Requires: docker, docker compose plugin, and (for GPU) nvidia-container-toolkit.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

SERVICE="slam"
CONTAINER="disal_slam"

BUILD=0
NO_CACHE=0
DOWN=0
EXEC_CMD=(bash)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build)    BUILD=1; shift ;;
    --no-cache) BUILD=1; NO_CACHE=1; shift ;;
    --down)     DOWN=1; shift ;;
    --)         shift; EXEC_CMD=("$@"); break ;;
    -h|--help)  sed -n '2,11p' "$0"; exit 0 ;;
    *)          echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

if [[ "${DOWN}" -eq 1 ]]; then
  docker compose down
  exit 0
fi

# --- X11: make sure the host's X server accepts local connections from the container ---
if [[ -z "${DISPLAY:-}" ]]; then
  echo "WARNING: DISPLAY is not set on the host; GUI apps (rviz, webots) won't work." >&2
else
  if command -v xhost >/dev/null 2>&1; then
    # Allow local non-network clients (Unix-socket); safest scope for desktop use.
    xhost +local:root >/dev/null
  else
    echo "WARNING: xhost not found on host. Install x11-xserver-utils for GUI passthrough." >&2
  fi
fi

export DISPLAY

# --- Build if requested or if the image doesn't exist yet ---
if [[ "${BUILD}" -eq 1 ]]; then
  if [[ "${NO_CACHE}" -eq 1 ]]; then
    docker compose build --no-cache "${SERVICE}"
  else
    docker compose build "${SERVICE}"
  fi
fi

# --- Start (or reuse) the container in the background ---
if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER}"; then
  docker compose up -d "${SERVICE}"
fi

# --- Sanity check: is the X socket actually visible inside the container? ---
docker exec "${CONTAINER}" bash -lc '
  echo "[run.sh] DISPLAY=${DISPLAY:-<unset>}"
  ls /tmp/.X11-unix/ 2>/dev/null || echo "[run.sh] /tmp/.X11-unix/ missing inside container"
' || true

# --- Drop into the container ---
exec docker exec -it "${CONTAINER}" "${EXEC_CMD[@]}"

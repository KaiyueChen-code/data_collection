#!/usr/bin/env bash
set -Eeuo pipefail

# Data selector + live camera visualization environment.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ENV_DIR="${VISUAL_ENV_DIR:-${PROJECT_ROOT}/.venv-visual}"
PYTHON_BIN="${PYTHON_BIN:-}"
PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PIP_DOWNLOAD_TIMEOUT="${PIP_DOWNLOAD_TIMEOUT:-120}"
PIP_DOWNLOAD_RETRIES="${PIP_DOWNLOAD_RETRIES:-10}"
INSTALL_SYSTEM_DEPS=1

usage() {
    cat <<'EOF'
Usage: bash scripts/setup_visual_env.sh [options]

Options:
  --env-dir PATH          Virtual environment path (default: .venv-visual)
  --python PATH           Python executable (default: python3.11, then python3)
  --pip-index-url URL     PyPI mirror (default: Tsinghua TUNA)
  --skip-system-deps      Do not install Ubuntu/Debian system packages
  -h, --help              Show this help

Environment variables:
  VISUAL_ENV_DIR          Same as --env-dir
  PYTHON_BIN              Same as --python
  PYPI_INDEX_URL          Same as --pip-index-url
  PIP_DOWNLOAD_TIMEOUT    Download timeout in seconds (default: 120)
  PIP_DOWNLOAD_RETRIES    Download retry count (default: 10)
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-dir)
            [[ $# -ge 2 ]] || { echo "Missing value for --env-dir" >&2; exit 2; }
            ENV_DIR="$2"
            shift 2
            ;;
        --python)
            [[ $# -ge 2 ]] || { echo "Missing value for --python" >&2; exit 2; }
            PYTHON_BIN="$2"
            shift 2
            ;;
        --pip-index-url)
            [[ $# -ge 2 ]] || { echo "Missing value for --pip-index-url" >&2; exit 2; }
            PYPI_INDEX_URL="$2"
            shift 2
            ;;
        --skip-system-deps)
            INSTALL_SYSTEM_DEPS=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "This environment targets Linux/V4L2 and must be set up on Linux." >&2
    exit 1
fi

if [[ -z "${PYTHON_BIN}" ]]; then
    if command -v python3.11 >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python3.11)"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python3)"
    else
        echo "Python 3.11+ was not found." >&2
        exit 1
    fi
fi

"${PYTHON_BIN}" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else "Python 3.11+ is required")'

install_apt_packages() {
    local -a privilege=()
    if [[ "${EUID}" -ne 0 ]]; then
        command -v sudo >/dev/null 2>&1 || {
            echo "sudo is required for system dependencies; rerun with --skip-system-deps if they are already installed." >&2
            exit 1
        }
        privilege=(sudo)
    fi

    "${privilege[@]}" apt-get update
    "${privilege[@]}" apt-get install -y \
        build-essential \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libv4l-dev \
        libxext6 \
        libxrender1 \
        pkg-config \
        python3-dev \
        python3-venv \
        udev \
        v4l-utils
}

if [[ "${INSTALL_SYSTEM_DEPS}" -eq 1 ]]; then
    if command -v apt-get >/dev/null 2>&1; then
        install_apt_packages
    else
        echo "Warning: apt-get was not found; skipping system packages." >&2
    fi
fi

if [[ -x "${ENV_DIR}/bin/python" ]]; then
    echo "Updating existing visual environment: ${ENV_DIR}"
else
    echo "Creating visual environment: ${ENV_DIR}"
    "${PYTHON_BIN}" -m venv "${ENV_DIR}"
fi
echo "Using PyPI mirror: ${PYPI_INDEX_URL}"
"${ENV_DIR}/bin/python" -m pip config --site set global.index-url "${PYPI_INDEX_URL}"
"${ENV_DIR}/bin/python" -m pip config --site set global.timeout "${PIP_DOWNLOAD_TIMEOUT}"
"${ENV_DIR}/bin/python" -m pip config --site set global.retries "${PIP_DOWNLOAD_RETRIES}"
PIP_INSTALL_ARGS=(
    --index-url "${PYPI_INDEX_URL}"
    --timeout "${PIP_DOWNLOAD_TIMEOUT}"
    --retries "${PIP_DOWNLOAD_RETRIES}"
)
"${ENV_DIR}/bin/python" -m pip install "${PIP_INSTALL_ARGS[@]}" --upgrade pip setuptools wheel
"${ENV_DIR}/bin/python" -m pip install "${PIP_INSTALL_ARGS[@]}" -r "${PROJECT_ROOT}/data_selector/requirements.txt"
"${ENV_DIR}/bin/python" -m pip install "${PIP_INSTALL_ARGS[@]}" pyudev v4l2-python3

"${ENV_DIR}/bin/python" - <<'PY'
import cv2
import matplotlib
import numpy
import pandas
import pyarrow
import pyudev
import pygame
import v4l2

print("Visual environment import check: OK")
PY

cat <<EOF

Visual environment is ready.
Activate it with:
  source "${ENV_DIR}/bin/activate"

Data selection:
  python3 data_selector/src/run_selector_workflow.py DATASET_NAME --hf-home DATASET_ROOT --only select

Camera preview:
  python3 data_collection/preview_cameras.py
EOF

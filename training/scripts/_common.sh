#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATASETS_DIR="${TRAINING_DIR}/datasets"
DOWNLOADS_DIR="${DATASETS_DIR}/downloads"
TFRECORDS_DIR="${DATASETS_DIR}/tfrecords"
CHECKPOINTS_DIR="${TRAINING_DIR}/checkpoints"
OUTPUTS_DIR="${TRAINING_DIR}/outputs"
LOGS_DIR="${TRAINING_DIR}/logs"

info() {
  printf '[INFO] %s\n' "$*"
}

warn() {
  printf '[WARN] %s\n' "$*" >&2
}

die() {
  printf '[ERROR] %s\n' "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

require_download_tool() {
  if command -v wget >/dev/null 2>&1; then
    DOWNLOAD_TOOL="wget"
  elif command -v curl >/dev/null 2>&1; then
    DOWNLOAD_TOOL="curl"
  else
    die "Install wget or curl before downloading datasets."
  fi
}

check_base_tools() {
  require_command python
  require_command unzip
  require_download_tool
  info "Tools found: python, unzip and ${DOWNLOAD_TOOL}."
}

create_layout() {
  mkdir -p \
    "${DOWNLOADS_DIR}" \
    "${DATASETS_DIR}/content/coco" \
    "${DATASETS_DIR}/style/dtd" \
    "${DATASETS_DIR}/style/custom_styles" \
    "${TFRECORDS_DIR}/content" \
    "${TFRECORDS_DIR}/style" \
    "${CHECKPOINTS_DIR}/pretrained" \
    "${CHECKPOINTS_DIR}/mobile_model" \
    "${OUTPUTS_DIR}/tflite" \
    "${LOGS_DIR}"
}

download_once() {
  local url="$1"
  local destination="$2"
  if [[ -s "${destination}" ]]; then
    info "Archive already exists, skipping download: ${destination}"
    return
  fi
  info "Downloading ${url}"
  if [[ "${DOWNLOAD_TOOL}" == "wget" ]]; then
    wget --continue --output-document="${destination}" "${url}"
  else
    curl --fail --location --continue-at - --output "${destination}" "${url}"
  fi
}

activate_venv_if_available() {
  if [[ -f "${TRAINING_DIR}/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${TRAINING_DIR}/.venv/bin/activate"
    info "Activated virtual environment: ${TRAINING_DIR}/.venv"
  else
    warn "No .venv found. Run scripts/00_setup_env.sh or activate your Magenta environment."
  fi
}

require_python_import() {
  python -c "import $1" >/dev/null 2>&1 || die "Python module '$1' is unavailable in the active environment."
}

use_slim_models_library() {
  local slim_dir="${TRAINING_DIR}/third_party/tensorflow-models/research/slim"
  [[ -d "${slim_dir}/nets" ]] ||
    die "TF-Slim models library not found at ${slim_dir}. Run scripts/00_setup_env.sh first."
  export PYTHONPATH="${slim_dir}:${PYTHONPATH:-}"
  info "Added TF-Slim image models to PYTHONPATH: ${slim_dir}"
}

checkpoint_prefix_exists() {
  local prefix="$1"
  [[ -e "${prefix}" || -e "${prefix}.index" || -e "${prefix}.meta" ]]
}

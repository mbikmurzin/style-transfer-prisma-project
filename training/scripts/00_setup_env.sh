#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

check_base_tools
require_command git
create_layout

PYTHON_VERSION="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
case "${PYTHON_VERSION}" in
  3.7|3.8|3.9|3.10) ;;
  *)
    die "Magenta's pinned tensorflow==2.9.1 environment requires Python 3.7-3.10. Found ${PYTHON_VERSION}."
    ;;
esac

VENV_DIR="${TRAINING_DIR}/.venv"
MAGENTA_DIR="${TRAINING_DIR}/third_party/magenta"
TF_MODELS_DIR="${TRAINING_DIR}/third_party/tensorflow-models"
if [[ ! -d "${VENV_DIR}" ]]; then
  info "Creating virtual environment at ${VENV_DIR}"
  python -m venv "${VENV_DIR}"
else
  info "Virtual environment already exists: ${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "${TRAINING_DIR}/requirements_training.txt"

if [[ ! -d "${MAGENTA_DIR}/.git" ]]; then
  mkdir -p "$(dirname "${MAGENTA_DIR}")"
  info "Cloning Magenta source needed for mobile training modules."
  git clone --depth 1 https://github.com/magenta/magenta.git "${MAGENTA_DIR}"
else
  info "Magenta source already exists: ${MAGENTA_DIR}"
fi

python -m pip install --no-deps --editable "${MAGENTA_DIR}"
if [[ ! -d "${TF_MODELS_DIR}/.git" ]]; then
  info "Cloning TensorFlow models repository for the TF-Slim MobileNet source."
  git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/tensorflow/models.git "${TF_MODELS_DIR}"
  git -C "${TF_MODELS_DIR}" sparse-checkout set research/slim
else
  info "TF-Slim models source already exists: ${TF_MODELS_DIR}/research/slim"
fi

info "Environment ready. Activate with: source ${VENV_DIR}/bin/activate"

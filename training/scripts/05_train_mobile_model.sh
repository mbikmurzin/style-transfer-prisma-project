#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

check_base_tools
create_layout
activate_venv_if_available
require_python_import tensorflow

DATA_DIR="${DATA_DIR:-${DATASETS_DIR}}"
STYLE_TFRECORD="${STYLE_TFRECORD:-${DATA_DIR}/tfrecords/style/style_train.tfrecord}"
# Magenta's legacy image_utils.imagenet_inputs() accepts a directory and globs train-*.
# 04_create_tfrecords.sh writes content_train.tfrecord and this required train-* alias.
CONTENT_DIR="${CONTENT_DIR:-${DATA_DIR}/tfrecords/content}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${CHECKPOINTS_DIR}/pretrained}"
TRAIN_DIR="${TRAIN_DIR:-${CHECKPOINTS_DIR}/mobile_model}"
VGG_CHECKPOINT="${VGG_CHECKPOINT:-${CHECKPOINT_DIR}/vgg_16.ckpt}"
MOBILENET_CHECKPOINT="${MOBILENET_CHECKPOINT:-${CHECKPOINT_DIR}/mobilenet_v2_1.0_224.ckpt}"
TRAIN_MODULE="magenta.models.arbitrary_image_stylization.arbitrary_image_stylization_train_mobile"

if ! python -c "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('${TRAIN_MODULE}') else 1)" >/dev/null 2>&1; then
  cat >&2 <<EOF
[ERROR] Magenta mobile training module was not found:
        ${TRAIN_MODULE}

Install the legacy-compatible environment first:
  bash scripts/00_setup_env.sh
  source .venv/bin/activate

This Magenta model is based on TensorFlow 1-style code and is not guaranteed
to run under current TensorFlow/Python releases. If local legacy setup is not
practical, run this pipeline in a pinned Colab/WSL/Linux environment.
EOF
  exit 1
fi

TF_VERSION="$(python -c 'import tensorflow as tf; print(tf.__version__)' 2>/dev/null || true)"
if [[ "${TF_VERSION}" != 2.9.* ]]; then
  warn "Detected TensorFlow ${TF_VERSION:-unknown}; Magenta's pinned setup uses TensorFlow 2.9.1."
  warn "Training may fail. Prefer scripts/00_setup_env.sh or a pinned Colab runtime."
fi

use_slim_models_library
python -c "from nets.mobilenet import mobilenet_v2" >/dev/null 2>&1 ||
  die "Cannot import nets.mobilenet from the checked out TF-Slim source. Run scripts/00_setup_env.sh."

find "${CONTENT_DIR}" -type f -name 'train-*' -print -quit | grep -q . ||
  die "Content TFRecords not found. Run scripts/04_create_tfrecords.sh first."
[[ -s "${STYLE_TFRECORD}" ]] || die "Style TFRecord not found: ${STYLE_TFRECORD}"
checkpoint_prefix_exists "${VGG_CHECKPOINT}" ||
  die "Missing VGG checkpoint prefix: ${VGG_CHECKPOINT}. See README_training.md."
checkpoint_prefix_exists "${MOBILENET_CHECKPOINT}" ||
  die "Missing MobileNetV2 checkpoint prefix: ${MOBILENET_CHECKPOINT}. See README_training.md."

info "Starting Magenta mobile arbitrary stylization training."
info "DATA_DIR: ${DATA_DIR}"
info "STYLE_TFRECORD: ${STYLE_TFRECORD}"
info "CONTENT_DIR: ${CONTENT_DIR}"
info "CHECKPOINT_DIR: ${CHECKPOINT_DIR}"
info "Training output: ${TRAIN_DIR}"

# Template based on Magenta's checked-in mobile trainer. If its API changes,
# replace the module path or flag names below according to the installed source.
python -m "${TRAIN_MODULE}" \
  --imagenet_data_dir="${CONTENT_DIR}" \
  --style_dataset_file="${STYLE_TFRECORD}" \
  --vgg_checkpoint="${VGG_CHECKPOINT}" \
  --mobilenet_checkpoint="${MOBILENET_CHECKPOINT}" \
  --train_dir="${TRAIN_DIR}" \
  --batch_size="${BATCH_SIZE:-4}" \
  --image_size="${IMAGE_SIZE:-256}" \
  --alpha="${ALPHA:-0.25}" \
  --train_steps="${TRAIN_STEPS:-1000}" \
  --random_style_image_size=True \
  --augment_style_images=True \
  --center_crop=False \
  --save_summaries_secs="${SAVE_SUMMARIES_SECS:-30}" \
  --save_interval_secs="${SAVE_INTERVAL_SECS:-120}" \
  --logtostderr \
  2>&1 | tee "${LOGS_DIR}/train_mobile.log"

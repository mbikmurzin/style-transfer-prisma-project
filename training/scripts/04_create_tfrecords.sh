#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

check_base_tools
create_layout
activate_venv_if_available
require_python_import tensorflow

COCO_IMAGES="${DATASETS_DIR}/content/coco/val2017"
DTD_IMAGES="${DATASETS_DIR}/style/dtd/images"
CUSTOM_IMAGES="${DATASETS_DIR}/style/custom_styles"
STYLE_RECORD="${TFRECORDS_DIR}/style/style_train.tfrecord"
CUSTOM_STYLE_RECORD="${TFRECORDS_DIR}/style/style_custom.tfrecord"
CONTENT_RECORD="${TFRECORDS_DIR}/content/content_train.tfrecord"
# The legacy Magenta content loader globs train-* from --imagenet_data_dir.
CONTENT_MAGENTA_RECORD="${TFRECORDS_DIR}/content/train-00000-of-00001"

[[ -d "${COCO_IMAGES}" ]] || die "COCO images not found. Run scripts/01_download_content_dataset.sh first."
[[ -d "${DTD_IMAGES}" ]] || die "DTD images not found. Run scripts/02_download_style_dataset.sh first."

if [[ -s "${STYLE_RECORD}" ]]; then
  info "Style TFRecord already exists, skipping: ${STYLE_RECORD}"
else
  info "Creating style_train.tfrecord from DTD RGB images."
  python "${SCRIPT_DIR}/create_image_tfrecord.py" \
    --input_dir "${DTD_IMAGES}" \
    --output_file "${STYLE_RECORD}"
fi

if [[ -s "${CONTENT_RECORD}" ]]; then
  info "Content TFRecord already exists, skipping: ${CONTENT_RECORD}"
else
  info "Creating content_train.tfrecord from COCO val2017 RGB images."
  python "${SCRIPT_DIR}/create_image_tfrecord.py" \
    --input_dir "${COCO_IMAGES}" \
    --output_file "${CONTENT_RECORD}"
fi

if [[ ! -s "${CONTENT_MAGENTA_RECORD}" || "${CONTENT_RECORD}" -nt "${CONTENT_MAGENTA_RECORD}" ]]; then
  info "Writing Magenta-compatible train-* content alias: ${CONTENT_MAGENTA_RECORD}"
  cp "${CONTENT_RECORD}" "${CONTENT_MAGENTA_RECORD}"
else
  info "Magenta-compatible content alias already exists: ${CONTENT_MAGENTA_RECORD}"
fi

if find "${CUSTOM_IMAGES}" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) -print -quit | grep -q .; then
  if [[ -s "${CUSTOM_STYLE_RECORD}" ]]; then
    info "Custom style TFRecord already exists, skipping: ${CUSTOM_STYLE_RECORD}"
  else
    info "Creating optional style_custom.tfrecord from custom_styles."
    python "${SCRIPT_DIR}/create_image_tfrecord.py" \
      --input_dir "${CUSTOM_IMAGES}" \
      --output_file "${CUSTOM_STYLE_RECORD}"
  fi
else
  info "No custom style images found; using DTD only."
fi

info "TFRecord preparation complete:"
printf '  style:   %s\n' "${STYLE_RECORD}"
printf '  content: %s\n' "${CONTENT_RECORD}"
printf '  Magenta content reader path: %s\n' "${TFRECORDS_DIR}/content"

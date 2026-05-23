#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

check_base_tools
create_layout

COCO_URL="https://images.cocodataset.org/zips/val2017.zip"
COCO_ZIP="${DOWNLOADS_DIR}/val2017.zip"
COCO_ROOT="${DATASETS_DIR}/content/coco"
COCO_IMAGES="${COCO_ROOT}/val2017"

if [[ -d "${COCO_IMAGES}" ]] && find "${COCO_IMAGES}" -type f -name '*.jpg' -print -quit | grep -q .; then
  info "COCO val2017 is already extracted: ${COCO_IMAGES}"
  exit 0
fi

download_once "${COCO_URL}" "${COCO_ZIP}"
info "Extracting COCO val2017 into ${COCO_ROOT}"
unzip -q "${COCO_ZIP}" -d "${COCO_ROOT}"
info "Content images ready: ${COCO_IMAGES}"

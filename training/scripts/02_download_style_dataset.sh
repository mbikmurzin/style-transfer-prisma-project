#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

check_base_tools
require_command tar
create_layout

DTD_URL="https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"
DTD_ARCHIVE="${DOWNLOADS_DIR}/dtd-r1.0.1.tar.gz"
STYLE_ROOT="${DATASETS_DIR}/style"
DTD_IMAGES="${STYLE_ROOT}/dtd/images"

if [[ -d "${DTD_IMAGES}" ]] && find "${DTD_IMAGES}" -type f -name '*.jpg' -print -quit | grep -q .; then
  info "DTD is already extracted: ${DTD_IMAGES}"
  exit 0
fi

download_once "${DTD_URL}" "${DTD_ARCHIVE}"
info "Extracting DTD into ${STYLE_ROOT}"
tar -xzf "${DTD_ARCHIVE}" -C "${STYLE_ROOT}"
info "Style images ready: ${DTD_IMAGES}"
info "Optional personal images may be added to ${STYLE_ROOT}/custom_styles"

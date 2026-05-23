#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

check_base_tools
create_layout

info "Training folder layout:"
printf '  content raw:      %s\n' "${DATASETS_DIR}/content/coco/val2017"
printf '  style DTD:        %s\n' "${DATASETS_DIR}/style/dtd/images"
printf '  custom styles:    %s\n' "${DATASETS_DIR}/style/custom_styles"
printf '  TFRecords:        %s\n' "${TFRECORDS_DIR}"
printf '  checkpoints:      %s\n' "${CHECKPOINTS_DIR}"
printf '  model outputs:    %s\n' "${OUTPUTS_DIR}"
printf '  logs:             %s\n' "${LOGS_DIR}"

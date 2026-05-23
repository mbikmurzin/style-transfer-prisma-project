#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

require_command python
create_layout
activate_venv_if_available
require_python_import tensorflow

EXPORT_DIR="${OUTPUTS_DIR}/tflite"
ANDROID_MODELS_DIR="$(cd "${TRAINING_DIR}/../android-app/app/src/main/assets/models" && pwd)"
MAGENTA_CHECKPOINT="${MAGENTA_CHECKPOINT:-${CHECKPOINTS_DIR}/mobile_model}"
EXPORT_MODULE="magenta.models.arbitrary_image_stylization.arbitrary_image_stylization_convert_tflite"

fallback_instructions() {
  cat >&2 <<EOF
[INFO] A trained Magenta checkpoint or a predict/transform SavedModel pair was not available.

Fast fallback for the Android demo:
  Download the official pretrained TensorFlow Lite style transfer model pair:
    style_predict.tflite
    style_transform.tflite
  from the official TensorFlow Lite Artistic Style Transfer example / TF Hub
  and place both files in:
    ${ANDROID_MODELS_DIR}

Official overview:
  https://www.tensorflow.org/lite/examples/style_transfer/overview

Example download commands:
  curl -L -o "${ANDROID_MODELS_DIR}/style_predict.tflite" \
    "https://tfhub.dev/google/lite-model/magenta/arbitrary-image-stylization-v1-256/int8/prediction/1?lite-format=tflite"
  curl -L -o "${ANDROID_MODELS_DIR}/style_transform.tflite" \
    "https://tfhub.dev/google/lite-model/magenta/arbitrary-image-stylization-v1-256/int8/transfer/1?lite-format=tflite"

The official pretrained architecture runs style_predict first and then
style_transform. The Android executor must support that two-stage contract.
EOF
}

copy_to_android_assets() {
  local predict="${EXPORT_DIR}/style_predict.tflite"
  local transform="${EXPORT_DIR}/style_transform.tflite"
  [[ -s "${predict}" && -s "${transform}" ]] ||
    die "Export did not create both required files in ${EXPORT_DIR}."
  mkdir -p "${ANDROID_MODELS_DIR}"
  cp "${predict}" "${ANDROID_MODELS_DIR}/style_predict.tflite"
  cp "${transform}" "${ANDROID_MODELS_DIR}/style_transform.tflite"
  info "Copied exported TFLite models to Android assets: ${ANDROID_MODELS_DIR}"
}

has_savedmodel_pair() {
  local root="$1"
  local predict=""
  local transform=""
  for name in predict style_predict style_prediction; do
    [[ -f "${root}/${name}/saved_model.pb" ]] && predict="yes"
  done
  for name in transform style_transform style_transfer; do
    [[ -f "${root}/${name}/saved_model.pb" ]] && transform="yes"
  done
  [[ -n "${predict}" && -n "${transform}" ]]
}

find_savedmodel_root() {
  local candidate
  if [[ -n "${SAVED_MODEL_DIR:-}" ]] && has_savedmodel_pair "${SAVED_MODEL_DIR}"; then
    printf '%s\n' "${SAVED_MODEL_DIR}"
    return 0
  fi
  for candidate in \
    "${OUTPUTS_DIR}/saved_model" \
    "${OUTPUTS_DIR}/saved_models" \
    "${CHECKPOINTS_DIR}/mobile_model/saved_model" \
    "${CHECKPOINTS_DIR}/mobile_model/export"; do
    if has_savedmodel_pair "${candidate}"; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  while IFS= read -r saved_model_file; do
    candidate="$(dirname "$(dirname "${saved_model_file}")")"
    if has_savedmodel_pair "${candidate}"; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done < <(find "${OUTPUTS_DIR}" "${CHECKPOINTS_DIR}" -type f -name 'saved_model.pb' 2>/dev/null)
  return 1
}

SAVED_MODEL_ROOT=""
if SAVED_MODEL_ROOT="$(find_savedmodel_root)"; then
  info "Found exported SavedModel pair: ${SAVED_MODEL_ROOT}"
  python "${SCRIPT_DIR}/export_to_tflite.py" \
    --saved_model_dir "${SAVED_MODEL_ROOT}" \
    --output_dir "${EXPORT_DIR}" \
    2>&1 | tee "${LOGS_DIR}/export_tflite.log"
  copy_to_android_assets
  exit 0
fi

if [[ ! -s "${MAGENTA_CHECKPOINT}/checkpoint" ]]; then
  while IFS= read -r checkpoint_marker; do
    MAGENTA_CHECKPOINT="$(dirname "${checkpoint_marker}")"
    break
  done < <(find "${CHECKPOINTS_DIR}" -type f -name 'checkpoint' 2>/dev/null)
fi

if [[ -s "${MAGENTA_CHECKPOINT}/checkpoint" ]]; then
  info "Found trained checkpoint directory: ${MAGENTA_CHECKPOINT}"
  if ! python -c "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('${EXPORT_MODULE}') else 1)" >/dev/null 2>&1; then
    warn "Magenta checkpoint exists, but the legacy Magenta export module is not installed."
    warn "Use scripts/00_setup_env.sh in Python 3.7-3.10, or run export in a pinned Colab runtime."
    fallback_instructions
    exit 1
  fi
  use_slim_models_library
  python -c "from nets.mobilenet import mobilenet_v2" >/dev/null 2>&1 ||
    die "Cannot import nets.mobilenet. Run scripts/00_setup_env.sh in a legacy environment."
  # The Magenta checkpoint exporter builds its own predict/transform SavedModels
  # internally, then writes the float and quantized TFLite model pairs.
  python -m "${EXPORT_MODULE}" \
    --checkpoint="${MAGENTA_CHECKPOINT}" \
    --output_dir="${EXPORT_DIR}" \
    --image_size="${IMAGE_SIZE:-256}" \
    --alpha="${ALPHA:-0.25}" \
    --logtostderr \
    2>&1 | tee "${LOGS_DIR}/export_tflite.log"
  copy_to_android_assets
  exit 0
fi

fallback_instructions
exit 1

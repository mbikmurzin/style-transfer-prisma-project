"""Convert style prediction and transform SavedModels to TensorFlow Lite."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import tensorflow as tf

LOGGER = logging.getLogger("export_to_tflite")
MODEL_NAMES = {
    "style_predict": ("predict", "style_predict", "style_prediction"),
    "style_transform": ("transform", "style_transform", "style_transfer"),
}


def find_saved_model(root: Path, aliases: tuple[str, ...]) -> Path | None:
    for name in aliases:
        candidate = root / name
        if (candidate / "saved_model.pb").is_file():
            return candidate
    if (root / "saved_model.pb").is_file() and root.name.lower() in aliases:
        return root
    return None


def convert(saved_model: Path, output_file: Path) -> None:
    LOGGER.info("Converting SavedModel: %s", saved_model)
    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model))
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        model_bytes = converter.convert()
        LOGGER.info("Enabled TensorFlow Lite default optimizations.")
    except Exception as optimized_error:  # noqa: BLE001 - report and retry export.
        LOGGER.warning("Optimized conversion failed: %s", optimized_error)
        LOGGER.warning("Retrying float TensorFlow Lite conversion without optimizations.")
        try:
            converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model))
            model_bytes = converter.convert()
        except Exception as plain_error:  # noqa: BLE001 - command line error surface.
            raise RuntimeError(
                f"Failed to convert {saved_model} with and without optimizations: {plain_error}"
            ) from plain_error
    output_file.write_bytes(model_bytes)
    LOGGER.info("Saved %s (%.2f MB)", output_file, len(model_bytes) / (1024 * 1024))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--saved_model_dir",
        type=Path,
        required=True,
        help="Directory containing predict/ and transform/ SavedModel folders.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    if not args.saved_model_dir.is_dir():
        raise SystemExit(
            f"SavedModel directory not found: {args.saved_model_dir}. "
            "Train/export the model first or use official pretrained TFLite models."
        )

    models: dict[str, Path] = {}
    for output_name, aliases in MODEL_NAMES.items():
        saved_model = find_saved_model(args.saved_model_dir, aliases)
        if saved_model is None:
            names = ", ".join(str(args.saved_model_dir / name / "saved_model.pb") for name in aliases)
            raise SystemExit(
                f"Cannot find SavedModel for {output_name}. Expected one of: {names}."
            )
        models[output_name] = saved_model

    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        for output_name, saved_model in models.items():
            convert(saved_model, args.output_dir / f"{output_name}.tflite")
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    LOGGER.info("Exported prediction and transform TFLite models to %s", args.output_dir)


if __name__ == "__main__":
    main()

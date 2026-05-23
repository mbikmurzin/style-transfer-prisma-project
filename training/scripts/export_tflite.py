"""Convert the trained two-input Keras model to TensorFlow Lite."""

from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("artifacts/style_transfer.keras"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/style_transfer.tflite"))
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model, compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(tflite_model)

    interpreter = tf.lite.Interpreter(model_path=str(args.out))
    interpreter.allocate_tensors()
    print(f"Created {args.out} ({args.out.stat().st_size / 1024 / 1024:.2f} MB)")
    for tensor in interpreter.get_input_details():
        print(f"input: {tensor['name']} {tensor['shape'].tolist()} {tensor['dtype']}")
    print(f"output: {interpreter.get_output_details()[0]['shape'].tolist()}")


if __name__ == "__main__":
    main()

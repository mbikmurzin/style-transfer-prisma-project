"""Run desktop inference with the exact TensorFlow Lite model used by Android."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf


def load_image(path: Path, size: int) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    return np.expand_dims(np.asarray(image, dtype=np.float32) / 255.0, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--content", type=Path, required=True)
    parser.add_argument("--style", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    interpreter = tf.lite.Interpreter(model_path=str(args.model))
    interpreter.allocate_tensors()
    inputs = interpreter.get_input_details()
    image_size = int(inputs[0]["shape"][1])
    content = load_image(args.content, image_size)
    style = load_image(args.style, image_size)
    for tensor in inputs:
        source = style if "style" in tensor["name"].lower() else content
        interpreter.set_tensor(tensor["index"], source)
    interpreter.invoke()
    output = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])[0]
    result = Image.fromarray((np.clip(output, 0, 1) * 255).astype(np.uint8))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.out)
    print(f"Saved inference output to {args.out}")


if __name__ == "__main__":
    main()

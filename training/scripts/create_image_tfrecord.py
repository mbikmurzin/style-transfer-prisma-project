"""Create image TFRecords consumable by Magenta arbitrary stylization readers.

Each example intentionally contains both feature naming conventions:
`image_raw`/`label` for style input and `image/encoded`/`image/class/label`
for the ImageNet-shaped content reader used by the legacy mobile trainer.
"""

from __future__ import annotations

import argparse
import io
import logging
from pathlib import Path

from PIL import Image, UnidentifiedImageError
import tensorflow as tf

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
LOGGER = logging.getLogger("create_image_tfrecord")


def bytes_feature(value: bytes) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def int64_feature(value: int) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def encode_rgb_jpeg(path: Path) -> bytes:
    with Image.open(path) as source:
        rgb_image = source.convert("RGB")
        buffer = io.BytesIO()
        rgb_image.save(buffer, format="JPEG", quality=95)
        return buffer.getvalue()


def make_example(path: Path, encoded: bytes, label: int) -> tf.train.Example:
    return tf.train.Example(
        features=tf.train.Features(
            feature={
                "image_raw": bytes_feature(encoded),
                "label": int64_feature(label),
                "image/encoded": bytes_feature(encoded),
                "image/format": bytes_feature(b"JPEG"),
                "image/filename": bytes_feature(path.name.encode("utf-8")),
                "image/class/label": int64_feature(label),
                "image/class/text": bytes_feature(b"training-image"),
            }
        )
    )


def list_images(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    if not args.input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {args.input_dir}")

    candidates = list_images(args.input_dir)
    if not candidates:
        raise SystemExit(f"No jpg/jpeg/png images found in {args.input_dir}")

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    skipped = 0
    with tf.io.TFRecordWriter(str(args.output_file)) as writer:
        for path in candidates:
            try:
                encoded = encode_rgb_jpeg(path)
            except (OSError, ValueError, UnidentifiedImageError) as error:
                skipped += 1
                LOGGER.warning("Skipping unreadable image %s: %s", path, error)
                continue
            writer.write(make_example(path, encoded, processed).SerializeToString())
            processed += 1

    if processed == 0:
        raise SystemExit("All discovered images were unreadable; generated TFRecord is empty.")

    LOGGER.info("Created TFRecord: %s", args.output_file)
    LOGGER.info("Processed images: %d; skipped broken images: %d.", processed, skipped)


if __name__ == "__main__":
    main()

"""Prepare content/style image folders for arbitrary style transfer training."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from PIL import Image, ImageOps
from tqdm import tqdm

SUPPORTED = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--content-src", type=Path, required=True)
    parser.add_argument("--style-src", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def list_images(folder: Path) -> list[Path]:
    return sorted(path for path in folder.rglob("*") if path.suffix.lower() in SUPPORTED)


def split(paths: list[Path], ratio: float, seed: int) -> tuple[list[Path], list[Path]]:
    shuffled = paths.copy()
    random.Random(seed).shuffle(shuffled)
    validation_count = max(1, int(len(shuffled) * ratio)) if len(shuffled) > 1 else 0
    return shuffled[validation_count:], shuffled[:validation_count]


def write_resized(images: list[Path], target: Path, size: int) -> list[str]:
    target.mkdir(parents=True, exist_ok=True)
    written = []
    for index, source in enumerate(tqdm(images, desc=f"Preparing {target}")):
        destination = target / f"{index:06d}.jpg"
        with Image.open(source) as image:
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            rgb = ImageOps.fit(rgb, (size, size), method=Image.Resampling.LANCZOS)
            rgb.save(destination, "JPEG", quality=95)
        written.append(str(destination))
    return written


def prepare_domain(name: str, source: Path, output: Path, size: int, ratio: float, seed: int) -> dict:
    paths = list_images(source)
    if not paths:
        raise ValueError(f"No supported images found in {source}.")
    train, validation = split(paths, ratio, seed)
    return {
        "source": str(source),
        "train": write_resized(train, output / name / "train", size),
        "validation": write_resized(validation, output / name / "validation", size),
    }


def main() -> None:
    args = parse_args()
    if args.out.exists():
        shutil.rmtree(args.out)
    manifest = {
        "image_size": args.size,
        "content": prepare_domain(
            "content", args.content_src, args.out, args.size, args.validation_ratio, args.seed
        ),
        "style": prepare_domain(
            "style", args.style_src, args.out, args.size, args.validation_ratio, args.seed + 1
        ),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared dataset manifest: {args.out / 'manifest.json'}")


if __name__ == "__main__":
    main()

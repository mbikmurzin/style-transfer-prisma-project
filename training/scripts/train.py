"""Train a compact arbitrary image style transfer model with perceptual losses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf

from models import build_feature_extractor, build_stylizer, gram_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/train_config.json"))
    return parser.parse_args()


def image_dataset(folder: Path, image_size: int, batch_size: int) -> tf.data.Dataset:
    paths = [str(path) for path in sorted(folder.glob("*.jpg"))]
    if not paths:
        raise ValueError(f"No prepared JPEG files found in {folder}. Run prepare_dataset.py first.")
    dataset = tf.data.Dataset.from_tensor_slices(paths).shuffle(len(paths))

    def load(path: tf.Tensor) -> tf.Tensor:
        image = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
        image = tf.image.resize(image, (image_size, image_size))
        return tf.cast(image, tf.float32) / 255.0

    return dataset.map(load, num_parallel_calls=tf.data.AUTOTUNE).repeat().batch(batch_size).prefetch(2)


def vgg_preprocess(images: tf.Tensor) -> tf.Tensor:
    return tf.keras.applications.vgg19.preprocess_input(images * 255.0)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    image_size = config["image_size"]
    batch_size = config["batch_size"]
    content_data = image_dataset(Path(config["content_dir"]), image_size, batch_size)
    style_data = image_dataset(Path(config["style_dir"]), image_size, batch_size)
    train_data = tf.data.Dataset.zip((content_data, style_data))

    model = build_stylizer(image_size)
    features = build_feature_extractor()
    optimizer = tf.keras.optimizers.Adam(config["learning_rate"])
    checkpoint_dir = Path(config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    manager = tf.train.CheckpointManager(checkpoint, str(checkpoint_dir), max_to_keep=3)

    @tf.function
    def train_step(content: tf.Tensor, style: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        with tf.GradientTape() as tape:
            stylized = model([content, style], training=True)
            generated_features = features(vgg_preprocess(stylized), training=False)
            content_features = features(vgg_preprocess(content), training=False)
            style_features = features(vgg_preprocess(style), training=False)
            content_loss = tf.reduce_mean(tf.square(generated_features[-1] - content_features[-1]))
            style_loss = tf.add_n(
                [
                    tf.reduce_mean(tf.square(gram_matrix(output) - gram_matrix(reference)))
                    for output, reference in zip(generated_features, style_features)
                ]
            )
            tv_loss = tf.reduce_mean(tf.image.total_variation(stylized))
            total = (
                config["content_weight"] * content_loss
                + config["style_weight"] * style_loss
                + config["tv_weight"] * tv_loss
            )
        gradients = tape.gradient(total, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return total, content_loss, style_loss

    for epoch in range(config["epochs"]):
        totals = []
        for step, (content, style) in enumerate(train_data.take(config["steps_per_epoch"]), start=1):
            total, content_loss, style_loss = train_step(content, style)
            totals.append(total)
            if step % 50 == 0:
                print(
                    f"epoch={epoch + 1} step={step} total={total.numpy():.4f} "
                    f"content={content_loss.numpy():.4f} style={style_loss.numpy():.4f}"
                )
        print(f"epoch={epoch + 1} mean_loss={tf.reduce_mean(totals).numpy():.4f}")
        manager.save()

    output = Path(config["keras_model_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    print(f"Saved Keras inference model to {output}")


if __name__ == "__main__":
    main()

"""Mobile-sized arbitrary style transfer model and training feature extractor."""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def conv_block(x: tf.Tensor, filters: int, stride: int = 1) -> tf.Tensor:
    x = layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False)(x)
    x = layers.LayerNormalization(axis=-1)(x)
    return layers.Activation("relu")(x)


def build_stylizer(image_size: int = 256) -> keras.Model:
    content_image = keras.Input((image_size, image_size, 3), name="content_image")
    style_image = keras.Input((image_size, image_size, 3), name="style_image")

    style = conv_block(style_image, 32, 2)
    style = conv_block(style, 64, 2)
    style = conv_block(style, 128, 2)
    style = layers.GlobalAveragePooling2D()(style)
    gamma = layers.Dense(128, name="style_gamma")(style)
    beta = layers.Dense(128, name="style_beta")(style)
    gamma = layers.Reshape((1, 1, 128))(gamma)
    beta = layers.Reshape((1, 1, 128))(beta)

    content = conv_block(content_image, 32)
    content = conv_block(content, 64, 2)
    content = conv_block(content, 128, 2)
    normalized = layers.LayerNormalization(axis=-1)(content)
    mixed = normalized * (1.0 + gamma) + beta
    mixed = conv_block(mixed, 128)
    mixed = conv_block(mixed, 128)

    output = layers.UpSampling2D(interpolation="bilinear")(mixed)
    output = conv_block(output, 64)
    output = layers.UpSampling2D(interpolation="bilinear")(output)
    output = conv_block(output, 32)
    output = layers.Conv2D(3, 3, padding="same", activation="sigmoid", name="stylized_image")(output)
    return keras.Model([content_image, style_image], output, name="mobile_arbitrary_stylizer")


def build_feature_extractor() -> keras.Model:
    vgg = keras.applications.VGG19(include_top=False, weights="imagenet")
    vgg.trainable = False
    layer_names = ["block2_conv2", "block3_conv4", "block4_conv4"]
    outputs = [vgg.get_layer(name).output for name in layer_names]
    return keras.Model(vgg.input, outputs, name="perceptual_vgg19")


def gram_matrix(feature: tf.Tensor) -> tf.Tensor:
    return tf.linalg.einsum("bijc,bijd->bcd", feature, feature) / tf.cast(
        tf.shape(feature)[1] * tf.shape(feature)[2], tf.float32
    )

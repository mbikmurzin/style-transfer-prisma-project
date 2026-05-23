# Model Assets

The Android app runs the two-stage arbitrary style transfer graph. Put both
TensorFlow Lite files in this directory with these exact filenames:

```text
style_predict.tflite
style_transform.tflite
```

Model flow:

```text
style image -> style_predict -> style bottleneck
content image + style bottleneck -> style_transform -> stylized image
```

`StyleTransferModelRunner.kt` reads the input tensor metadata at runtime and
handles float or quantized RGB image tensors. The two model files must be a
matching exported or pretrained pair.

When either file is missing, the app reports:

```text
Models not found. Put style_predict.tflite and style_transform.tflite into assets/models.
```

Use `training/scripts/06_export_tflite.sh` after export, or follow the
pretrained-model fast path in `training/README_training.md`.

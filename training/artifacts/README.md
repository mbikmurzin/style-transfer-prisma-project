# Generated Artifacts

This folder is intentionally excluded from version control except for this note.

Expected final files after real training:

```text
style_transfer.keras
style_transfer.tflite   <- legacy одностадийный учебный эксперимент
checkpoints/
```

During project validation, `smoke_untrained.tflite` and
`smoke_inference_output.png` may be created to verify conversion and execution.
They use randomly initialized weights and are **not** valid style-transfer
results for the report or Android demo.

Текущий Android-клиент использует не этот файл, а Magenta-compatible пару
`../outputs/tflite/style_predict.tflite` и
`../outputs/tflite/style_transform.tflite`.

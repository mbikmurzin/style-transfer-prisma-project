# Training Pipeline

Основная инструкция для обучения Magenta arbitrary image stylization и
экспорта мобильной пары моделей находится в
[`README_training.md`](README_training.md).

Текущий Android-клиент ожидает:

```text
outputs/tflite/style_predict.tflite
outputs/tflite/style_transform.tflite
```

После экспорта `scripts/06_export_tflite.sh` копирует эту пару в
`../android-app/app/src/main/assets/models/`.

Каталоги `data/`, `config/`, `artifacts/` и ранние Python-скрипты оставлены
как отдельный учебный эксперимент с одностадийной Keras-моделью. Его
`style_transfer.tflite` не является входным контрактом текущего
Android-приложения.

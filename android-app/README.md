# Android Application

`Neural Canvas` - Kotlin-приложение по двухстадийной схеме официального
TensorFlow Lite Style Transfer example.

## Возможности

1. Выбор content image из галереи или съемка через системную камеру.
2. Выбор style image из галереи или из `app/src/main/assets/styles/`.
3. Локальный inference: `style_predict` вычисляет style bottleneck,
   `style_transform` создает изображение результата.
4. Переключатель GPU Delegate с автоматическим возвратом на CPU, если GPU
   недоступен или не поддерживает модель.
5. Выбор CPU threads: `1`, `2` или `4`.
6. Вывод времени preprocessing, prediction, transfer, postprocessing и полного
   выполнения.
7. Сохранение PNG в `Pictures/NeuralCanvas`.

## Подключение моделей

Поместите два файла:

```text
app/src/main/assets/models/style_predict.tflite
app/src/main/assets/models/style_transform.tflite
```

Если они отсутствуют, приложение показывает:

```text
Models not found. Put style_predict.tflite and style_transform.tflite into assets/models.
```

Модели можно получить через `training/scripts/06_export_tflite.sh` или
скачать как официальную pretrained пару, следуя
`training/README_training.md`.

## Сборка

Откройте этот каталог в Android Studio, выберите JDK 17 и выполните запуск
конфигурации `app` на Android 10 (API 29) или новее. Камера удобнее всего
проверяется на физическом устройстве.

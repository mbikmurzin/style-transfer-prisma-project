# Architecture

## Product Flow

```text
Gallery content image ----\
                            -> preprocessing -> TensorFlow Lite -> result Bitmap -> gallery PNG
Gallery style image ------/
```

На устройстве изображения масштабируются под tensor shapes загруженных
TFLite-моделей и кодируются как `float32` либо quantized RGB в зависимости от
model metadata. `style_predict.tflite` получает style image и формирует
bottleneck, после чего `style_transform.tflite` принимает content image вместе
с bottleneck и возвращает готовый output.

## Training Model

Inference-модель состоит из двух ветвей:

| Block | Purpose |
| --- | --- |
| Style encoder | Извлекает компактное style embedding из произвольного style image. |
| Content encoder | Извлекает spatial feature map, сохраняющий расположение объектов. |
| Feature modulation | Style embedding формирует `gamma` и `beta` для изменения content features. |
| Decoder | Upsampling и convolution blocks создают stylized RGB image. |

На этапе обучения дополнительно используется frozen `VGG19`:

- content loss сравнивает высокоуровневые признаки результата и content image;
- style loss сравнивает Gram matrices признаков результата и style image;
- total variation loss сглаживает пиксельный шум.

VGG19 нужен только при обучении. В `.tflite` экспортируется компактный stylizer, что делает запуск на телефоне возможным.

## TensorFlow Lite Boundary

```text
style_predict.tflite:
  style_image -> style_bottleneck

style_transform.tflite:
  content_image + style_bottleneck -> stylized_image
```

`StyleTransferModelRunner.kt` определяет image tensor shape из модели, замеряет
времена этапов, поддерживает float/quantized buffers и при недоступном GPU
Delegate автоматически выполняет inference на CPU.

## Production Extensions

- GPU delegate или NNAPI benchmarking на целевом телефоне.
- Float16/INT8 quantization с representative dataset.
- Tile-based high-resolution output вместо фиксированного `256x256`.
- Camera input и история сохраненных фильтров.

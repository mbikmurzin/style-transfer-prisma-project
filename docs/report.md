# Отчет по проекту Neural Canvas

## 1. Описание задачи

Цель проекта - создать учебный прототип arbitrary image style transfer:
пользователь выбирает исходную фотографию (`content image`) и изображение
визуального стиля (`style image`), а модель строит новое изображение, которое
сохраняет композицию исходника и переносит свойства выбранного стиля.

Проект рассчитан на мобильную демонстрацию в приложении Android. На случай,
если мобильную сборку невозможно показать на защите, подготовлен веб-интерфейс
на Flask с тем же двухстадийным TensorFlow Lite inference.

## 2. Почему TensorFlow Lite

TensorFlow Lite выбран как формат выполнения модели на Android:

- инференс может выполняться локально на устройстве без отправки фотографий
  на сервер;
- мобильное приложение может загружать компактные `.tflite` модели из
  `assets/models/`;
- можно сравнивать CPU и GPU Delegate и выводить время каждого этапа;
- тот же формат моделей запускается в fallback Flask-прототипе для
  демонстрации результата.

Используемый runtime-контракт состоит из двух моделей:

```text
style image -> style_predict.tflite -> style bottleneck
content image + style bottleneck -> style_transform.tflite -> stylized image
```

## 3. Используемые датасеты

Для воспроизводимого учебного pipeline выбраны:

| Domain | Dataset | Назначение | Путь после подготовки |
| --- | --- | --- | --- |
| Content | COCO val2017 | Фотографии с объектами и сценами | `training/datasets/content/coco/val2017/` |
| Style | DTD (Describable Textures Dataset) | Текстуры и визуальные признаки стиля | `training/datasets/style/dtd/images/` |
| Optional style | Пользовательские JPG/PNG | Собственные референсы стиля | `training/datasets/style/custom_styles/` |

DTD выбран как легкий основной набор стилей для учебного запуска. Для
расширения style domain документация допускает WikiArt или Painter by Number.
В текущем состоянии репозитория каталоги и download-скрипты подготовлены, но
полные датасеты в репозиторий не включены. Для фактического демонстрационного
inference использована совместимая pretrained пара TensorFlow Lite моделей.

## 4. Формирование датасета

Pipeline расположен в `training/scripts/`:

| Скрипт | Действие |
| --- | --- |
| `01_download_content_dataset.sh` | Скачивает и распаковывает COCO val2017. |
| `02_download_style_dataset.sh` | Скачивает и распаковывает DTD. |
| `03_prepare_folders.sh` | Создает рабочую структуру каталогов. |
| `04_create_tfrecords.sh` | Запускает формирование TFRecord-файлов. |
| `create_image_tfrecord.py` | Читает JPG/JPEG/PNG, преобразует в RGB, пропускает битые изображения и пишет encoded bytes. |

Результаты подготовки должны появиться в:

```text
training/datasets/tfrecords/style_train.tfrecord
training/datasets/tfrecords/content_train.tfrecord
```

Для совместимости с legacy reader Magenta content records также могут
использоваться под именем `train-00000-of-00001`. В проекте проведена
smoke-проверка формирования TFRecord, но фактический полный training dataset
перед сдачей нужно скачать или обозначить использование pretrained моделей.

## 5. Обучение модели

`training/scripts/05_train_mobile_model.sh` содержит заготовку запуска mobile
модели Magenta arbitrary image stylization с переменными:

```text
DATA_DIR, STYLE_TFRECORD, CONTENT_DIR, CHECKPOINT_DIR, TRAIN_DIR
```

Обучение использует content image и style image, а целевой stylized результат
оптимизируется через perceptual losses. Для мобильной конфигурации необходимы
pretrained checkpoints VGG16 и MobileNetV2.

Важное ограничение: исходный Magenta pipeline опирается на legacy TensorFlow
API (`tensorflow.compat.v1`, `tf_slim`) и требует отдельного старого
окружения. Поэтому проект не утверждает, что обучение гарантированно
запустится на современном TensorFlow без адаптации. Для быстрой и честной
демонстрации допускается использовать совместимую pretrained TFLite пару.

## 6. Конвертация в TensorFlow Lite

Экспорт реализован в:

```text
training/scripts/06_export_tflite.sh
training/scripts/export_to_tflite.py
```

Экспортный pipeline ищет SavedModel для prediction и transformation, запускает
`TFLiteConverter`, пытается включить optimizations и сохраняет:

```text
training/outputs/tflite/style_predict.tflite
training/outputs/tflite/style_transform.tflite
```

После успешного экспорта модели копируются в Android assets:

```text
android-app/app/src/main/assets/models/
```

Если обучение не запускалось, README training содержит быстрый путь с
pretrained моделями. Для демонстрации рабочая pretrained пара добавлена в:

```text
android-app/app/src/main/assets/models/
demo/flask-prototype/models/
```

## 7. Как работает приложение

Android-приложение написано на Kotlin и содержит:

| Файл | Назначение |
| --- | --- |
| `MainActivity.kt` | Выбор content/style, запуск, отображение результата, сохранение в галерею. |
| `StyleTransferModelRunner.kt` | Загрузка двух TFLite моделей и inference pipeline. |
| `ImageUtils.kt` | RGB preprocessing, tensor buffers и преобразование output в `Bitmap`. |
| `BenchmarkResult.kt` | Формат метрик для UI. |

Пользователь может:

1. Выбрать content image из галереи или сделать снимок камерой.
2. Выбрать style image из галереи или встроенных assets.
3. Включить GPU Delegate и выбрать число CPU threads: `1`, `2`, `4`.
4. Нажать **Применить стиль**.
5. Увидеть исходник, стиль, результат и benchmark.
6. Сохранить PNG результата в `Pictures/NeuralCanvas`.

Runner динамически читает shapes и типы tensors из моделей, выполняет
нормализацию RGB, поддерживает float/quantized image tensors и возвращает
готовый `Bitmap`. Если GPU Delegate недоступен или падает на выполнении,
приложение повторяет inference через CPU. Если модели отсутствуют,
показывается понятное сообщение о требуемых файлах.

## 8. Веб-прототип

В `demo/flask-prototype/` реализован резервный UI:

- загрузка content и style изображений;
- реальный двухстадийный TensorFlow Lite inference;
- три визуальных блока: input, style, output;
- вывод benchmark;
- сохранение результата в `static/outputs/`;
- понятная ошибка при отсутствии моделей.

Это дает возможность защитить алгоритмическую часть через браузер, даже если
не получится собрать или продемонстрировать Android APK.

### Выполненный контрольный inference

Контрольный запуск выполнен **23 мая 2026 года** в Flask-прототипе на CPU с
файлами `demo/inputs/content_example.png` и
`demo/inputs/style_example.png`. Результат сохранен в
`demo/outputs/stylized_example.png`.

| Метрика | Значение |
| --- | ---: |
| Input Image Size | `384 x 384` |
| Pre-process execution time | `126.48 ms` |
| Predicting style execution time | `7.28 ms` |
| Transferring style execution time | `210.13 ms` |
| Post-process execution time | `39.53 ms` |
| Full execution time | `383.52 ms` |

Эти значения относятся к desktop CPU-запуску веб-прототипа, а не к Android
устройству; мобильные метрики необходимо снять отдельно при записи видео.

## 9. Ограничения проекта

- В `screenshots/` сейчас есть только UI mockup placeholder, а не снимки
  реального запуска.
- Видео демонстрации еще не записано.
- Android APK и inference на физическом устройстве в текущей среде не
  проверены; доказан запуск fallback Flask-прототипа на той же паре моделей.
- Полное обучение Magenta требует legacy TensorFlow окружения и вычислительных
  ресурсов; для короткой защиты разумно использовать pretrained models.
- Качество и выходное разрешение определяются выбранными TFLite моделями;
  mobile-модели обычно дают компактный результат, а не полноразмерный
  production export.

## 10. Соответствие критериям оценивания

| Критерий | Что реализовано | Что показать на защите |
| --- | --- | --- |
| Подготовка данных | COCO/DTD download scripts, TFRecord writer, структура datasets | README training и сформированный TFRecord либо объяснение pretrained path |
| Обучение/конвертация | Mobile training template, TFLite export pipeline, честное предупреждение о совместимости | Скрипты и две рабочие `.tflite` модели |
| Android/прототип | Kotlin Android UI и Flask fallback | Живой запуск одного из интерфейсов |
| Реальный inference | Реализован в обоих UI; Flask inference подтвержден сохраненным output | Показать готовый output и повторить запуск вживую |
| Метрики | В UI заложены времена этапов, GPU/CPU/thread параметры | Показать benchmark после запуска |
| Документация | README, training/android manuals, report, demo script, checklist | Открыть GitHub README и отчет |
| Evidence | Созданы папки и сценарий | Добавить результат, screenshots и MP4 |

## 11. Что необходимо сделать перед отправкой

Минимальный путь к убедительной защите:

1. Открыть подготовленный Flask-прототип либо собрать Android с уже
   добавленными pretrained моделями.
2. Повторить реальный inference на готовых input-примерах вживую.
3. Сделать три скриншота: input/style, output/metrics,
   сохраненный файл.
4. Записать короткое видео по `docs/demo_script.md`.
5. При мобильной защите заменить/дополнить desktop benchmark измерениями с
   устройства.
6. Приложить ссылку на репозиторий и, если доступно, APK или ссылку на видео.

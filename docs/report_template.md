# Отчет: Neural Canvas

## 1. Постановка задачи

Цель: реализовать arbitrary image style transfer, при котором одна мобильная модель переносит стиль любого выбранного изображения на пользовательскую фотографию.

## 2. Данные

| Domain | Источник | Количество | Лицензия/ссылка |
| --- | --- | ---: | --- |
| Content images | _заполнить_ | _заполнить_ | _заполнить_ |
| Style images | _заполнить_ | _заполнить_ | _заполнить_ |

Подготовка данных: преобразование в RGB, center crop, resize до `256x256`, разбиение на train/validation. Приложить содержимое `training/data/processed/manifest.json`.

## 3. Модель и обучение

Модель принимает два изображения: content и style. Style encoder вычисляет параметры модуляции, content encoder сохраняет пространственные признаки, decoder формирует результат. Для обучения используются perceptual content/style losses на frozen VGG19 и total variation regularization.

| Параметр | Значение |
| --- | --- |
| Input resolution | `256x256` |
| Batch size | _из config_ |
| Epochs / steps | _из config_ |
| Learning rate | _из config_ |
| TFLite model size | _после экспорта_ |

Добавить фрагмент training log или график loss.

## 4. Инференс

Модель экспортирована в TensorFlow Lite и запускается локально в Android-приложении. Изображения не передаются на сервер.

| Content | Style | Stylized output |
| --- | --- | --- |
| `demo/inputs/content_example.png` | `demo/inputs/style_example.png` | `demo/outputs/stylized_example.png` |

Добавить фактические изображения и измеренное время обработки на устройстве.

## 5. Android-прототип

Приложение предоставляет выбор content/style изображения, запуск преобразования, просмотр и сохранение результата. Вставить кадры из `screenshots/`.

## 6. Выводы

Описать достигнутое качество, ограничения разрешения/датасета и следующие улучшения: quantization, ускорение через delegate, более длительное обучение.

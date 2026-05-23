# Чек-лист сдачи: Neural Canvas

Статус на дату подготовки комплекта: **код, документация, pretrained models и
фактический Flask TFLite inference подготовлены; для полной мобильной защиты
требуются Android-запуск, скриншоты и видео**.

## Обязательная структура

| Материал | Статус | Расположение |
| --- | --- | --- |
| Основной README | Готово | `README.md` |
| Training README | Готово | `training/README_training.md` |
| Android README | Готово | `android-app/README_android.md` |
| Android-приложение | Реализовано с моделями, сборка не подтверждена | `android-app/` |
| Pipeline подготовки/обучения/экспорта | Готов | `training/scripts/` |
| Запасной Flask-прототип | Проверен реальным TFLite inference | `demo/flask-prototype/` |
| Папка скриншотов | Создана, реальных кадров нет | `screenshots/` |
| Папка видео | Создана, видео нет | `demo/video/` |
| Папка результатов | Есть фактический Flask output | `demo/outputs/stylized_example.png` |
| Итоговый отчет | Готов как честный отчет о состоянии | `docs/report.md` |
| Сценарий демонстрации | Готов | `docs/demo_script.md` |

## Перед демонстрацией

- [x] Скачать официальную pretrained пару либо экспортировать собственные
  `style_predict.tflite` и `style_transform.tflite`.
- [x] Положить модели в `android-app/app/src/main/assets/models/`.
- [x] Для fallback также положить модели в `demo/flask-prototype/models/`.
- [x] Запустить реальный inference на `demo/inputs/content_example.png` и
  `demo/inputs/style_example.png`.
- [x] Сохранить фактический результат в `demo/outputs/stylized_example.png`.
- [ ] Не выдавать `training/artifacts/smoke_inference_output.png` за результат:
  он относится к smoke-проверке/случайным весам.

## Data Preparation

- [x] Создан pipeline скачивания COCO val2017 и DTD.
- [x] Создана папка `training/datasets/style/custom_styles/`.
- [x] Реализовано создание TFRecord через `create_image_tfrecord.py`.
- [ ] Скачать фактические датасеты либо зафиксировать, что для демонстрации
  использована pretrained модель.
- [ ] Добавить в отчет количество реально использованных изображений и
  лицензии/источники данных.

## Training and TensorFlow Lite

- [x] Описан Magenta-compatible training pipeline и предупреждение о legacy
  TensorFlow окружении.
- [x] Реализован экспорт пары TFLite моделей.
- [ ] Запустить обучение либо явно указать использование pretrained pair.
- [x] Получить рабочие pretrained `.tflite` файлы и подтвердить inference
  результатом через Flask prototype.

## Application and Prototype

- [x] Реализован Android UI с gallery/camera content input и gallery/assets
  style input.
- [x] Реализован двухстадийный TFLite runner, метрики, GPU switch, выбор CPU
  threads и fallback на CPU.
- [x] Реализовано сохранение результата в галерею.
- [x] Реализован и проверен Flask fallback с двумя upload inputs и сохранением
  output.
- [ ] Собрать Android APK и запустить на устройстве/эмуляторе.
- [ ] Проверить камеру, GPU/CPU fallback и сохранение изображения.

## Evidence

- [x] Включены примерные content/style inputs в `demo/inputs/`.
- [x] Добавить реальный stylized output.
- [ ] Добавить реальные screenshots вместо/в дополнение к mockup placeholder.
- [ ] Записать видео на 1-2 минуты по `docs/demo_script.md`.
- [x] Вставить в отчет фактические метрики веб-прототипа.

## Финальная отправка

- [ ] Проверить, что README содержит точные команды запуска.
- [ ] Добавить ссылку на репозиторий и, при наличии, APK/видео.
- [ ] Открыть результат на защите и выполнить один inference вживую.

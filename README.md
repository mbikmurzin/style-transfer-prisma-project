# Neural Canvas: Arbitrary Image Style Transfer on Android

Учебный проект мобильного приложения в духе Prisma: пользователь выбирает **content image** (что сохранить по композиции) и **style image** (какую визуальную манеру перенести), после чего приложение генерирует **stylized image** полностью на устройстве через **TensorFlow Lite**.

Проект закрывает полный цикл задания: подготовку изображений, обучение arbitrary style transfer модели, экспорт `.tflite`, Android-прототип, сценарий демонстрации и материалы для оформления сдачи.

## Quick Start

Самый быстрый способ проверить фактический inference - запустить Flask
prototype с уже включенной pretrained TensorFlow Lite парой:

```bash
cd demo/flask-prototype
pip install -r requirements.txt
python app.py
```

На ПК откройте `http://127.0.0.1:5000`, загрузите content/style images и
нажмите **Применить стиль**. Страница покажет результат и времена этапов, а
PNG сохранится в `demo/flask-prototype/static/outputs/`.

## Структура проекта

```text
style-transfer-project/
  README.md                         <- этот документ и быстрый старт
  android-app/                      <- Kotlin Android-приложение с TFLite inference
    app/src/main/assets/models/     <- сюда помещаются style_predict.tflite и style_transform.tflite
    app/src/main/assets/styles/     <- встроенные примеры стилей
  training/                         <- dataset pipeline, обучение и конвертация
    datasets/                       <- COCO/DTD/custom styles и TFRecord
    checkpoints/                    <- pretrained/training checkpoints
    outputs/tflite/                 <- экспортированная пара mobile models
    scripts/                        <- download, TFRecord, training и export
  demo/                             <- входные картинки и выходы inference
    flask-prototype/                <- резервный веб-интерфейс с TFLite
    video/                          <- запись демонстрации либо ссылка
  screenshots/                      <- кадры приложения для отчета
  docs/                             <- архитектура, отчет и сценарий видео
  submit_checklist.md               <- проверка перед отправкой
```

## Что реально реализовано

- Android UI: выбор content из галереи/камеры, style из галереи/assets, запуск inference, показ и сохранение PNG результата.
- `StyleTransferModelRunner.kt`: двухстадийный запуск `style_predict.tflite` + `style_transform.tflite`, GPU fallback, CPU threads и timing metrics.
- Magenta training pipeline: скачивание COCO val2017/DTD, формирование
  TFRecord, команда-заготовка mobile training и экспорт пары `.tflite`.
- Flask fallback: выбор двух изображений, двухстадийный TFLite inference,
  сохранение результата и показ benchmark в браузере.

## Контракт модели

Android-приложение следует официальной схеме TensorFlow Lite Style Transfer и
использует две модели:

| Model | Input / output | Назначение |
| --- | --- | --- | --- |
| `style_predict.tflite` | style image -> style bottleneck | Кодирует выбранный стиль. |
| `style_transform.tflite` | content image + bottleneck -> result | Применяет стиль к content. |

`training/` содержит воспроизводимый pipeline подготовки датасетов и обучения. Для быстрой сдачи допустимо поместить официальную pretrained пару `.tflite` моделей в Android assets, а pipeline оставить как доказательство воспроизводимого пути обучения и конвертации.

Magenta-compatible путь описан в `training/README_training.md`; Android runner уже реализует его двухстадийный contract `style_predict` + `style_transform`.

## TensorFlow Lite Pipeline

Для демонстрации в репозитории присутствует совместимая pretrained пара
моделей. Обработка выполняется по цепочке:

```text
style bitmap
  -> RGB preprocessing / normalization
  -> style_predict.tflite
  -> style bottleneck [1, 1, 1, 100]

content bitmap + style bottleneck
  -> style_transform.tflite
  -> stylized RGB bitmap
```

Фактически проверенные размеры downloaded pair: style input `256x256`,
content/output `384x384`. И Android runner, и Flask prototype читают tensor
metadata моделей и не используют фиксированный размер результата в коде.

## Быстрый путь: подключить мобильные модели

Текущий Android-клиент запускает пару Magenta/TensorFlow Lite моделей:
`style_predict.tflite` и `style_transform.tflite`. Полный путь подготовки
COCO val2017, DTD, TFRecord, обучения и экспорта описан в
`training/README_training.md`.

После появления экспортированных SavedModel запустите:

```bash
cd training
bash scripts/06_export_tflite.sh
```

Скрипт создает две `.tflite` модели в `training/outputs/tflite/` и копирует
их в `android-app/app/src/main/assets/models/`. Если обучение не запускалось,
в `training/README_training.md` приведен быстрый путь с совместимой
предобученной парой моделей.

## Получение примера инференса

В `demo/inputs/` уже находятся иллюстративные входные PNG
(`content_example.png` и `style_example.png`), а style-пример также встроен
в Android assets. Фактический контрольный inference уже выполнен через
Flask-прототип с pretrained TFLite pair; его результат сохранен в
`demo/outputs/stylized_example.png`. Для мобильной демонстрации запустите
приложение с теми же двумя моделями в assets, выберите content/style, нажмите
**Применить стиль** и сохраните результат.

## Запуск Android-приложения

1. Откройте каталог `android-app/` в Android Studio.
2. Убедитесь, что в `app/src/main/assets/models/` лежат `style_predict.tflite` и `style_transform.tflite`.
3. Используйте JDK 17 и Android SDK 35, выполните Gradle Sync.
4. Запустите приложение на устройстве или эмуляторе с Android 10 (API 29)+.
5. Выберите или сфотографируйте content image, выберите стиль, нажмите **Применить стиль**, затем **Сохранить результат**.

Приложение не отправляет изображения в сеть: inference выполняется локально через TensorFlow Lite.

Проверка Android inference:

1. Выберите content image из галереи или камеры.
2. Выберите встроенный стиль или изображение из галереи.
3. Выберите `1`, `2` или `4` CPU threads и при необходимости включите GPU.
4. Нажмите **Применить стиль**.
5. Проверьте result image и блок метрик: input size, GPU enabled, threads,
   preprocessing, prediction, transfer, postprocessing и full execution time.
6. Нажмите **Сохранить результат** для записи PNG в галерею.

## Mobile Demo

Flask-прототип можно открыть на iPhone как демонстрационный мобильный UI:

1. ПК и iPhone должны быть подключены к одной Wi-Fi сети.
2. Запустите `demo/flask-prototype/app.py`. В коде сервер слушает сеть через:

```python
app.run(host="0.0.0.0", port=5000, debug=True)
```

3. Откройте в Safari адрес, напечатанный как `Phone / LAN`, например:

```text
http://LOCAL_IP:5000
```

Для текущей настроенной сети это `http://192.168.0.106:5000`. Если соединение
блокируется, разрешите Python для частных сетей в Windows Firewall; подробная
команда приведена в `demo/flask-prototype/README.md`.

## Как получить максимальные баллы

- **12 баллов:** обязательно показать правдоподобный фактический inference
  рабочей модели: content image, style image и полученный stylized output, а
  не UI mockup или результат случайных весов.
- **20 баллов:** показать работающее приложение или прототип с выбором
  изображения с камеры, применением стиля и отображением результата.
- **Дополнительные баллы:** приложить видео работы приложения, ссылку на
  приложение/APK или репозиторий и продемонстрировать запуск на мобильном
  устройстве.

Pretrained пара `style_predict.tflite` / `style_transform.tflite` и реальный
output Flask inference уже добавлены. До полной мобильной защиты необходимо
сделать реальные screenshots Android-запуска и записать видео; текущий статус
подробно зафиксирован в `submit_checklist.md`.

## Данные для обучения

Практичный учебный набор:

- content domain: фотографии сцен, объектов и городов, например подмножество COCO;
- style domain: изображения картин и иллюстраций, например WikiArt;
- минимум для smoke test: 20 content и 20 style изображений;
- для результата, похожего на продукт: тысячи изображений в каждом домене.

Не включайте сторонние датасеты в репозиторий. В отчете укажите источник, лицензию, число изображений и выполненные преобразования (`RGB`, center crop, resize `256x256`, split).

## Demo Materials

- `demo/inputs/` содержит content/style images для повторяемого inference.
- `demo/outputs/stylized_example.png` содержит фактический результат
  TensorFlow Lite inference, полученный Flask prototype.
- `screenshots/` предназначена для реальных кадров запуска; пользователь
  добавит screenshots работы с ПК перед отправкой преподавателю.
- `demo/video/` предназначена для видео мобильной демонстрации; пользователь
  добавит запись с телефона.

## Как преподавателю быстро проверить проект

1. Открыть `docs/report.md` и `submit_checklist.md`.
2. Сравнить `demo/inputs/content_example.png`,
   `demo/inputs/style_example.png` и `demo/outputs/stylized_example.png`.
3. Выполнить команды из **Quick Start**.
4. В веб-интерфейсе загрузить два input-файла и нажать **Применить стиль**.
5. Проверить появление результата и блока metrics.
6. Для мобильного сценария открыть адрес из **Mobile Demo** на iPhone.
7. Для Android-части открыть `android-app/README_android.md` и Kotlin runner.

## Документация сдачи

- [docs/architecture.md](docs/architecture.md) описывает сеть и поток inference.
- [training/README_training.md](training/README_training.md) описывает Magenta-compatible pipeline с COCO/DTD, mobile training и экспортом пары TFLite-моделей.
- [docs/report.md](docs/report.md) содержит подготовленный отчет о проекте и текущем статусе evidence.
- [docs/report_template.md](docs/report_template.md) оставлен как исходный каркас отчета.
- [docs/demo_script.md](docs/demo_script.md) задает сценарий демонстрации на 1-2 минуты.
- [demo/README.md](demo/README.md) фиксирует, какие примеры положить в проект.
- [demo/flask-prototype/README.md](demo/flask-prototype/README.md) описывает запасной веб-интерфейс.
- [demo/video/README.md](demo/video/README.md) определяет место для записи демонстрации.
- [screenshots/README.md](screenshots/README.md) перечисляет обязательные кадры.
- [submit_checklist.md](submit_checklist.md) используется перед отправкой.

## Ограничения

Это учебная мобильная архитектура: разрешение результата равно `256x256`, а качество зависит от длительности обучения и разнообразия style images. Для production-версии потребуются более сильная архитектура, quantization/benchmarking и обработка изображения в исходном разрешении.

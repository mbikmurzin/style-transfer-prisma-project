# Magenta Mobile Arbitrary Image Stylization: Training Pipeline

Этот pipeline следует `magenta/models/arbitrary_image_stylization` и готовит
данные для мобильной архитектуры Magenta, которую затем можно экспортировать в
TensorFlow Lite.

## Важное предупреждение о совместимости

Код Magenta использует `tensorflow.compat.v1`, очереди TF1 и `tf_slim`.
Текущий `setup.py` репозитория Magenta фиксирует `tensorflow==2.9.1`,
`numpy==1.21.6` и `tf_slim==1.1.0`. Такой стек следует запускать в отдельном
окружении **Python 3.7-3.10**. Python 3.11/3.12 и современный TensorFlow из
основного Android demo pipeline для этой части не подходят.

Скрипты написаны для Linux/macOS/WSL или Git Bash. В Windows PowerShell
запускайте их через WSL/Git Bash.

## Отличие от оригинального рецепта

Оригинальная инструкция Magenta использует **ImageNet** как content dataset и
DTD/Painter by Number как style datasets. Для легкого учебного запуска здесь:

- content: **COCO val2017**, около 5 000 фотографий;
- style: **DTD**, указанный в инструкции Magenta;
- optional style: `datasets/style/custom_styles/`.

COCO не является штатной заменой ImageNet из статьи. Скрипт
`create_image_tfrecord.py` формирует универсальные records: поля
`image_raw`/`label` нужны style reader, а `image/encoded` и
`image/class/label` читаются legacy content reader Magenta. Для content
создается также имя `train-00000-of-00001`, потому что Magenta жестко ищет
файлы `train-*` внутри `--imagenet_data_dir`. Это практичный bridge для
демонстрации pipeline, но для сравнимого с оригинальной работой качества
следует перейти на ImageNet и большой набор PBN + DTD/WikiArt.

## Структура

```text
training/
  README_training.md
  requirements_training.txt
  scripts/
    00_setup_env.sh
    01_download_content_dataset.sh
    02_download_style_dataset.sh
    03_prepare_folders.sh
    04_create_tfrecords.sh
    05_train_mobile_model.sh
    06_export_tflite.sh
    _common.sh
    create_image_tfrecord.py
    export_to_tflite.py
  third_party/
    magenta/                       # cloned by setup
    tensorflow-models/research/slim/ # cloned by setup for nets.mobilenet
  datasets/
    content/coco/val2017/        # COCO images after download
    style/dtd/images/            # DTD images after download
    style/custom_styles/         # user images
    tfrecords/                   # model inputs
  checkpoints/
    pretrained/                  # VGG16 and MobileNetV2 initial checkpoints
    mobile_model/                # trained Magenta checkpoint
  outputs/tflite/                # exported mobile models
  logs/
```

## 1. Environment

Install prerequisites: `bash`, `python3.10`, `git`, `wget` or `curl`, `unzip`
and `tar`. Then run:

```bash
cd training
bash scripts/00_setup_env.sh
source .venv/bin/activate
```

`00_setup_env.sh` creates `.venv`, installs the legacy-compatible package pins,
clones the Magenta source into `third_party/magenta`, installs it editable, and
checks out `tensorflow-models/research/slim` for the `nets.mobilenet` source
required by Magenta mobile training.

## 2. Скачать датасеты

```bash
bash scripts/01_download_content_dataset.sh
bash scripts/02_download_style_dataset.sh
bash scripts/03_prepare_folders.sh
```

Скрипты скачивают **COCO val2017** в `datasets/content/coco/val2017/` и
**DTD** в `datasets/style/dtd/images/`. Они не скачивают и не распаковывают
данные заново, если изображения уже присутствуют. Архивы кэшируются в
`datasets/downloads/`.

Для этого задания DTD является быстрым основным style dataset. Для более
широкого покрытия можно использовать **WikiArt** или **Painter by Number
(PBN)**. Собственные JPG/PNG стили положите непосредственно в
`datasets/style/custom_styles/`; `04_create_tfrecords.sh` создаст для них
отдельный optional record.

## 3. Pretrained Checkpoints

Mobile training requires VGG16 for perceptual losses and MobileNetV2 for style
prediction initialization. Download and extract them before step 5:

```bash
mkdir -p checkpoints/pretrained
cd checkpoints/pretrained
wget -c http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
tar -xzf vgg_16_2016_08_28.tar.gz
wget -c https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz
tar -xzf mobilenet_v2_1.0_224.tgz
cd ../..
```

Expected checkpoint prefixes (TensorFlow may store each prefix as
`.index`/`.data-*`/`.meta` files):

```text
checkpoints/pretrained/vgg_16.ckpt
checkpoints/pretrained/mobilenet_v2_1.0_224.ckpt
```

If the extracted MobileNet file is in an inner folder, pass its real path:

```bash
export MOBILENET_CHECKPOINT=/absolute/path/to/mobilenet_v2_1.0_224.ckpt
```

## 4. Сформировать TFRecord

```bash
bash scripts/04_create_tfrecords.sh
```

Outputs:

```text
datasets/tfrecords/style/style_train.tfrecord
datasets/tfrecords/content/content_train.tfrecord
datasets/tfrecords/content/train-00000-of-00001  # alias read by Magenta trainer
datasets/tfrecords/style/style_custom.tfrecord    # only when custom images exist
```

`scripts/create_image_tfrecord.py` рекурсивно ищет `.jpg`, `.jpeg`, `.png`,
приводит каждое исправное изображение к `RGB`, кодирует его в JPEG bytes,
пропускает поврежденные файлы с warning и выводит число обработанных файлов.
Каждая запись содержит style-compatible поля `image_raw`, `label` и
content-compatible поля `image/encoded`, `image/class/label`.

Его можно вызвать отдельно:

```bash
python scripts/create_image_tfrecord.py \
  --input_dir datasets/style/dtd/images \
  --output_file datasets/tfrecords/style/style_train.tfrecord

python scripts/create_image_tfrecord.py \
  --input_dir datasets/content/coco/val2017 \
  --output_file datasets/tfrecords/content/content_train.tfrecord
```

Для выбранной legacy-команды `arbitrary_image_stylization_train_mobile`
content reader принимает не имя файла, а каталог с TFRecord под именем
`train-*`; поэтому используйте `04_create_tfrecords.sh`, который автоматически
создает совместимую копию `train-00000-of-00001`.

## 5. Train the Mobile Model

The defaults are deliberately short for an educational smoke run:
`TRAIN_STEPS=1000`, `BATCH_SIZE=4`, `IMAGE_SIZE=256`, `ALPHA=0.25`.

```bash
bash scripts/05_train_mobile_model.sh
```

Override values without editing the script:

```bash
TRAIN_STEPS=10000 BATCH_SIZE=8 bash scripts/05_train_mobile_model.sh
```

Скрипт является командой-заготовкой и определяет переменные:
`DATA_DIR`, `STYLE_TFRECORD`, `CONTENT_DIR`, `CHECKPOINT_DIR`, `TRAIN_DIR`.
Он проверяет наличие mobile-модуля Magenta и выдаст инструкцию по legacy
environment/Colab вместо заявления, что обучение запустится на новом
TensorFlow.

To train using custom styles rather than DTD:

```bash
STYLE_TFRECORD="$PWD/datasets/tfrecords/style/style_custom.tfrecord" \
  bash scripts/05_train_mobile_model.sh
```

Checkpoints appear in `checkpoints/mobile_model/`, and logs are written to
`logs/train_mobile.log`. To inspect learning progress:

```bash
tensorboard --logdir checkpoints/mobile_model
```

## 6. Экспортировать TensorFlow Lite Models

```bash
bash scripts/06_export_tflite.sh
```

`06_export_tflite.sh` работает в следующем порядке:

1. Ищет готовую пару SavedModel в `outputs/saved_model/`,
   `outputs/saved_models/`, `checkpoints/mobile_model/saved_model/` или
   каталоге из переменной `SAVED_MODEL_DIR`.
2. Если пара найдена, запускает `scripts/export_to_tflite.py`.
3. Если SavedModel отсутствуют, но найден checkpoint в
   `checkpoints/mobile_model/`, пытается запустить legacy Magenta exporter.
4. Если training/export артефакты отсутствуют или legacy Magenta module не
   установлен, выводит инструкцию по официальным pretrained TFLite models.

Для явного каталога SavedModel используйте:

```bash
SAVED_MODEL_DIR="$PWD/outputs/saved_model" bash scripts/06_export_tflite.sh
```

Ожидаемая структура SavedModel:

```text
outputs/saved_model/
  predict/saved_model.pb
  transform/saved_model.pb
```

Python converter также можно вызвать напрямую:

```bash
python scripts/export_to_tflite.py \
  --saved_model_dir outputs/saved_model \
  --output_dir outputs/tflite
```

Он пытается включить `tf.lite.Optimize.DEFAULT`; если optimized conversion не
поддерживается конкретным SavedModel, повторяет экспорт без optimizations и
явно пишет это в log.

Magenta mobile model состоит из **двух stages**, а не одной двухвходовой модели:

```text
outputs/tflite/style_predict.tflite
outputs/tflite/style_transform.tflite
android-app/app/src/main/assets/models/style_predict.tflite
android-app/app/src/main/assets/models/style_transform.tflite
```

`style_predict` receives a style image and produces the style bottleneck.
`style_transform` receives a content image plus that bottleneck and produces
the stylized output. Android inference based on exported Magenta models must
load and run both stages.

При использовании штатного legacy Magenta checkpoint exporter в
`outputs/tflite/` могут дополнительно появиться quantized/calibrated
варианты; shell script копирует в Android assets основные float-файлы
`style_predict.tflite` и `style_transform.tflite`.

## Быстрый путь для сдачи

Если обучение не выполнялось или legacy Magenta окружение не удалось поднять,
используйте официальные pretrained TensorFlow Lite style transfer models из
TensorFlow Lite Artistic Style Transfer example. Официальный example
предоставляет пару моделей prediction/transfer для arbitrary stylization.

1. Скачайте pretrained models под именами:

```text
style_predict.tflite
style_transform.tflite
```

2. Положите их в Android assets:

```text
android-app/app/src/main/assets/models/
```

Официальный TensorFlow Lite tutorial использует TF Hub Magenta int8 model pair;
его можно загрузить командами из корня `training/`:

```bash
mkdir -p ../android-app/app/src/main/assets/models
curl -L -o ../android-app/app/src/main/assets/models/style_predict.tflite \
  "https://tfhub.dev/google/lite-model/magenta/arbitrary-image-stylization-v1-256/int8/prediction/1?lite-format=tflite"
curl -L -o ../android-app/app/src/main/assets/models/style_transform.tflite \
  "https://tfhub.dev/google/lite-model/magenta/arbitrary-image-stylization-v1-256/int8/transfer/1?lite-format=tflite"
```

3. Запустите Android-приложение с двухстадийным executor-ом:

```text
style image -> style_predict -> style bottleneck
content image + style bottleneck -> style_transform -> result image
```

Текущий однофайловый demo executor из первоначального шаблона ожидает
`style_transfer.tflite`; перед запуском именно Magenta pretrained pair его
необходимо заменить на двухстадийный вызов моделей.

## Source Alignment

This project follows these parts of the Magenta recipe:

- DTD is an officially described style dataset, together with PBN.
- Mobile training uses `arbitrary_image_stylization_train_mobile` with VGG16
  and MobileNetV2 checkpoints.
- Checkpoint export can use `arbitrary_image_stylization_convert_tflite`,
  while already exported SavedModels are converted by local
  `scripts/export_to_tflite.py`.

The local COCO content adapter is the only intentional educational deviation
from the original ImageNet-based data setup.

## Команды по порядку

```bash
cd training
bash scripts/00_setup_env.sh
source .venv/bin/activate
bash scripts/01_download_content_dataset.sh
bash scripts/02_download_style_dataset.sh
bash scripts/03_prepare_folders.sh
bash scripts/04_create_tfrecords.sh
# После добавления VGG16 и MobileNetV2 checkpoints:
bash scripts/05_train_mobile_model.sh
bash scripts/06_export_tflite.sh
```

Результаты подготовки данных лежат в `datasets/tfrecords/`, training
checkpoints и TensorBoard summaries - в `checkpoints/mobile_model/`,
консольные логи - в `logs/`, а экспортированные модели - в `outputs/tflite/`.

# Android: Neural Canvas

Приложение выполняет arbitrary image style transfer локально через TensorFlow
Lite:

```text
style bitmap -> style_predict.tflite -> style bottleneck
content bitmap + style bottleneck -> style_transform.tflite -> stylized Bitmap
```

Размеры входов и выхода читаются из tensor metadata загруженных моделей.
Интерфейс показывает время preprocessing, style prediction, style transfer,
postprocessing и полного inference.

## Как открыть проект

1. Установите Android Studio с Android SDK 35 и JDK 17.
2. Откройте каталог `android-app/` как Gradle-проект.
3. Дождитесь завершения Gradle Sync.
4. Выберите физическое Android-устройство или эмулятор с API 29+.

Для проверки камеры удобнее использовать физическое устройство.

## Куда положить модели

Приложению нужны два совместимых TensorFlow Lite файла с точными именами:

```text
app/src/main/assets/models/style_predict.tflite
app/src/main/assets/models/style_transform.tflite
```

Получить их можно экспортом через `../training/scripts/06_export_tflite.sh`
или по быстрому пути с pretrained-моделями из
`../training/README_training.md`.

Если файлов нет, на экране появится сообщение:

```text
Models not found. Put style_predict.tflite and style_transform.tflite into assets/models.
```

## Как запустить inference

1. Запустите конфигурацию `app` в Android Studio.
2. Для исходного изображения нажмите `Галерея` или `Камера`.
3. Для стиля нажмите `Галерея` или `Примеры`; один встроенный style asset уже
   находится в приложении.
4. При необходимости включите `Использовать GPU Delegate` и выберите число
   CPU threads: `1`, `2` или `4`.
5. Нажмите `Применить стиль`.
6. После появления результата проверьте блок метрик.
7. Нажмите `Сохранить результат`: PNG сохраняется в
   `Pictures/NeuralCanvas`.

Если GPU Delegate недоступен или завершается ошибкой, приложение повторяет
вычисление на CPU и показывает фактически использованный режим в метриках.

## Как записать видео для сдачи

1. Начните запись экрана средствами Android Studio Device Mirroring либо
   встроенной функцией записи экрана на телефоне.
2. Покажите главный экран приложения.
3. Выберите content image и style image.
4. Покажите переключатель GPU и выбранное число потоков.
5. Нажмите `Применить стиль` и дождитесь результата.
6. Задержитесь на экране с content, style, result и блоком метрик.
7. Нажмите `Сохранить результат` и покажите Toast об успешном сохранении.
8. Сохраните MP4 или укажите ссылку на видео в материалах сдачи.

# Inference Demo

Этот каталог хранит изображения для демонстрации результата:

```text
inputs/content_example.png       <- исходное изображение
inputs/style_example.png         <- референс стиля
outputs/stylized_example.png     <- результат фактического инференса
```

Для текущего Android workflow:

1. Поместите `style_predict.tflite` и `style_transform.tflite` в
   `android-app/app/src/main/assets/models/`.
2. Запустите приложение, выберите `inputs/content_example.png` как content.
3. Выберите стиль из галереи или встроенный пример, затем нажмите
   **Применить стиль**.
4. Сохраните полученный результат и добавьте его в `outputs/` для отчета.

В отчет включите content, style, output и метрики с экрана приложения.

## Запасной веб-прототип

Если Android-приложение нельзя собрать или показать на устройстве, используйте
[`flask-prototype/`](flask-prototype/). Он запускает ту же двухстадийную
TensorFlow Lite пару через браузер и сохраняет результат в
`flask-prototype/static/outputs/`.

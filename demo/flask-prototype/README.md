# Flask Style Transfer Prototype

Запасной веб-интерфейс для демонстрации arbitrary image style transfer, если
Android-сборка недоступна. Прототип выполняет реальный двухстадийный
TensorFlow Lite inference:

```text
style image -> style_predict.tflite -> style bottleneck
content image + style bottleneck -> style_transform.tflite -> output image
```

## Подготовка моделей

Поместите совместимую пару моделей в:

```text
models/style_predict.tflite
models/style_transform.tflite
```

Модели можно скопировать из `../../training/outputs/tflite/` после экспорта
или использовать pretrained-пару, описанную в
`../../training/README_training.md`.

Если моделей нет, интерфейс покажет:

```text
Models not found. Put style_predict.tflite and style_transform.tflite into models/.
```

## Запуск

```bash
pip install -r requirements.txt
python app.py
```

Локально на ПК откройте адрес:

```text
http://127.0.0.1:5000
```

## Открытие с iPhone по Wi-Fi

1. ПК и iPhone должны быть подключены к одной Wi-Fi сети.
2. На ПК запустите прототип:

```bash
pip install -r requirements.txt
python app.py
```

3. При запуске приложение напечатает адрес `Phone / LAN`, автоматически
   определенный для текущей сети. Для указанной локальной сети откройте в
   Safari на iPhone:

```text
http://192.168.0.106:5000
```

4. Если сайт не открывается, разрешите Python в Windows Firewall для частных
   сетей.
5. Если окно Firewall не появилось, откройте PowerShell от имени
   администратора и выполните:

```powershell
netsh advfirewall firewall add rule name="Flask5000" dir=in action=allow protocol=TCP localport=5000
```

Сервер слушает все сетевые интерфейсы (`0.0.0.0`), поэтому адрес для телефона
должен совпадать с LAN IP компьютера, напечатанным при запуске.

## Использование

1. Загрузите content image в формате JPG или PNG.
2. Загрузите style image в формате JPG или PNG.
3. Нажмите **Применить стиль**.
4. Сохраненный результат появится в `static/outputs/` и отобразится в третьем
   блоке страницы.

Размер изображения для inference определяется динамически из tensor metadata
модели. Прототип нормализует RGB inputs и поддерживает `float32`, `uint8` и
`int8` image tensors.

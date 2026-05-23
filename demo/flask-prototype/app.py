"""Minimal Flask UI for two-stage TensorFlow Lite arbitrary style transfer."""

from __future__ import annotations

import time
import uuid
import socket
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, render_template, request, url_for
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

try:
    import tensorflow as tf
except ImportError as error:  # pragma: no cover - shown as a user-facing setup error.
    tf = None
    TENSORFLOW_IMPORT_ERROR = str(error)
else:
    TENSORFLOW_IMPORT_ERROR = ""


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
OUTPUT_DIR = BASE_DIR / "static" / "outputs"
PREDICT_MODEL = MODEL_DIR / "style_predict.tflite"
TRANSFORM_MODEL = MODEL_DIR / "style_transform.tflite"
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png"}
MODEL_ERROR = (
    "Models not found. Put style_predict.tflite and style_transform.tflite into models/."
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


class InferenceError(RuntimeError):
    """Error that can be displayed to a demo user without a traceback."""


def ensure_directories() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def store_upload(file_storage: Any, prefix: str) -> Path:
    if not file_storage or not file_storage.filename:
        raise InferenceError("Загрузите исходное изображение и изображение стиля.")
    original = secure_filename(file_storage.filename)
    suffix = Path(original).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise InferenceError("Поддерживаются изображения JPG, JPEG и PNG.")
    target = UPLOAD_DIR / f"{prefix}_{uuid.uuid4().hex}{suffix}"
    file_storage.save(target)
    try:
        with Image.open(target) as image:
            image.verify()
    except (UnidentifiedImageError, OSError):
        target.unlink(missing_ok=True)
        raise InferenceError("Один из загруженных файлов не является корректным изображением.")
    return target


def load_interpreter(path: Path) -> Any:
    if tf is None:
        raise InferenceError(
            "TensorFlow is not installed. Run: pip install -r requirements.txt"
        )
    if not PREDICT_MODEL.is_file() or not TRANSFORM_MODEL.is_file():
        raise InferenceError(MODEL_ERROR)
    try:
        interpreter = tf.lite.Interpreter(model_path=str(path))
        interpreter.allocate_tensors()
        return interpreter
    except Exception as error:
        raise InferenceError(f"Не удалось открыть TFLite model: {error}") from error


def image_tensor_from_file(path: Path, tensor: dict[str, Any]) -> np.ndarray:
    shape = tuple(int(value) for value in tensor["shape"])
    if len(shape) != 4 or shape[0] != 1 or shape[-1] != 3 or min(shape) <= 0:
        raise InferenceError(
            f"Ожидался image input tensor [1, height, width, 3], получен {shape}."
        )
    height, width = shape[1], shape[2]
    try:
        with Image.open(path) as image:
            rgb = image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
            normalized = np.asarray(rgb, dtype=np.float32)[np.newaxis, ...] / 255.0
    except (UnidentifiedImageError, OSError) as error:
        raise InferenceError(f"Не удалось прочитать изображение: {path.name}") from error
    return quantize_normalized(normalized, tensor)


def quantize_normalized(values: np.ndarray, tensor: dict[str, Any]) -> np.ndarray:
    dtype = tensor["dtype"]
    if dtype == np.float32:
        return values.astype(np.float32)
    if dtype not in (np.uint8, np.int8):
        raise InferenceError(f"Неподдерживаемый image tensor type: {dtype}.")
    scale, zero_point = tensor["quantization"]
    if scale:
        encoded = np.rint(values / scale + zero_point)
    else:
        encoded = np.rint(values * 255.0)
    limits = np.iinfo(dtype)
    return np.clip(encoded, limits.min, limits.max).astype(dtype)


def image_from_output(values: np.ndarray, tensor: dict[str, Any]) -> Image.Image:
    shape = tuple(int(value) for value in tensor["shape"])
    if len(shape) != 4 or shape[0] != 1 or shape[-1] != 3:
        raise InferenceError(f"Ожидался output tensor изображения, получен {shape}.")
    dtype = tensor["dtype"]
    if dtype == np.float32:
        normalized = values.astype(np.float32)
    elif dtype in (np.uint8, np.int8):
        scale, zero_point = tensor["quantization"]
        normalized = (
            (values.astype(np.float32) - zero_point) * scale
            if scale
            else values.astype(np.float32) / 255.0
        )
    else:
        raise InferenceError(f"Неподдерживаемый output tensor type: {dtype}.")
    pixels = np.clip(normalized[0] * 255.0, 0, 255).round().astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def find_transform_inputs(details: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(details) != 2:
        raise InferenceError("style_transform должен иметь два inputs.")
    for detail in details:
        shape = tuple(int(value) for value in detail["shape"])
        name = str(detail.get("name", "")).lower()
        if "content" in name or (len(shape) == 4 and shape[-1] == 3):
            other = details[1] if detail is details[0] else details[0]
            return detail, other
    raise InferenceError("Не удалось определить content input модели style_transform.")


def validate_bottleneck(prediction: np.ndarray, bottleneck_input: dict[str, Any]) -> None:
    expected_shape = tuple(int(value) for value in bottleneck_input["shape"])
    if prediction.shape != expected_shape or prediction.dtype != bottleneck_input["dtype"]:
        raise InferenceError(
            "Модели несовместимы: style_predict output не совпадает "
            "с style_transform bottleneck input."
        )


def run_style_transfer(content_path: Path, style_path: Path) -> tuple[Path, dict[str, float | str]]:
    predict = load_interpreter(PREDICT_MODEL)
    transform = load_interpreter(TRANSFORM_MODEL)
    prediction_input = predict.get_input_details()[0]
    prediction_output = predict.get_output_details()[0]
    content_input, bottleneck_input = find_transform_inputs(transform.get_input_details())
    transform_output = transform.get_output_details()[0]

    full_start = time.perf_counter_ns()
    start = time.perf_counter_ns()
    style_tensor = image_tensor_from_file(style_path, prediction_input)
    content_tensor = image_tensor_from_file(content_path, content_input)
    preprocess_ms = elapsed_ms(start)

    try:
        start = time.perf_counter_ns()
        predict.set_tensor(prediction_input["index"], style_tensor)
        predict.invoke()
        style_bottleneck = predict.get_tensor(prediction_output["index"])
        predicting_ms = elapsed_ms(start)

        validate_bottleneck(style_bottleneck, bottleneck_input)
        start = time.perf_counter_ns()
        transform.set_tensor(content_input["index"], content_tensor)
        transform.set_tensor(bottleneck_input["index"], style_bottleneck)
        transform.invoke()
        result_tensor = transform.get_tensor(transform_output["index"])
        transferring_ms = elapsed_ms(start)
    except Exception as error:
        if isinstance(error, InferenceError):
            raise
        raise InferenceError(f"TFLite inference завершился ошибкой: {error}") from error

    start = time.perf_counter_ns()
    result_image = image_from_output(result_tensor, transform_output)
    output_path = OUTPUT_DIR / f"stylized_{uuid.uuid4().hex}.png"
    result_image.save(output_path)
    postprocess_ms = elapsed_ms(start)
    height, width = (int(value) for value in content_input["shape"][1:3])
    benchmark = {
        "input_size": f"{width} x {height}",
        "preprocess_ms": preprocess_ms,
        "predicting_ms": predicting_ms,
        "transferring_ms": transferring_ms,
        "postprocess_ms": postprocess_ms,
        "full_ms": elapsed_ms(full_start),
    }
    return output_path, benchmark


def elapsed_ms(start_ns: int) -> float:
    return round((time.perf_counter_ns() - start_ns) / 1_000_000, 2)


def get_lan_ip() -> str:
    """Return a phone-accessible private IPv4 address when available."""
    candidates: list[str] = []
    try:
        hostname = socket.gethostname()
        candidates.extend(
            info[4][0]
            for info in socket.getaddrinfo(hostname, None, socket.AF_INET)
            if info[4][0] and not info[4][0].startswith("127.")
        )
    except OSError:
        pass
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as connection:
            connection.connect(("8.8.8.8", 80))
            address = connection.getsockname()[0]
            if address and not address.startswith("127."):
                candidates.append(address)
    except OSError:
        pass
    for prefix in ("192.168.", "10.", "172."):
        for address in candidates:
            if address.startswith(prefix):
                return address
    return "192.168.0.106"


@app.route("/", methods=["GET", "POST"])
def index() -> str:
    context: dict[str, Any] = {
        "error": None,
        "content_url": None,
        "style_url": None,
        "result_url": None,
        "benchmark": None,
    }
    if request.method == "POST":
        try:
            content_path = store_upload(request.files.get("content_image"), "content")
            style_path = store_upload(request.files.get("style_image"), "style")
            context["content_url"] = url_for("static", filename=f"uploads/{content_path.name}")
            context["style_url"] = url_for("static", filename=f"uploads/{style_path.name}")
            result_path, benchmark = run_style_transfer(content_path, style_path)
            context["result_url"] = url_for("static", filename=f"outputs/{result_path.name}")
            context["benchmark"] = benchmark
        except InferenceError as error:
            context["error"] = str(error)
    return render_template("index.html", **context)


if __name__ == "__main__":
    ensure_directories()
    lan_ip = get_lan_ip()
    print("\nLocal:")
    print("http://127.0.0.1:5000")
    print("\nPhone / LAN:")
    print(f"http://{lan_ip}:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=True)

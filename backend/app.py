"""Local Flask server for the plain HTML/CSS/JavaScript SpikeFusion-GI UI."""

from __future__ import annotations

import ast
import base64
import hmac
import io
import tempfile
import threading
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, render_template, request, session
from PIL import Image, ImageOps
from torchvision import transforms

from configs import config as cfg
from models import SpikeFusion
from xai.gradcam import GradCAM


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_CHECKPOINT = ROOT / "checkpoints" / "active_model.pth"
SECRETS_FILE = ROOT / ".streamlit" / "secrets.toml"
DEVICE = torch.device(cfg.DEVICE)
MODEL = None
MODEL_LOCK = threading.RLock()
VIDEO_SETTINGS = {"change_threshold": 12.0, "min_gap": 0.8, "max_gap": 3.0}

app = Flask(
    __name__,
    template_folder=str(ROOT / "frontend" / "templates"),
    static_folder=str(ROOT / "frontend" / "static"),
    static_url_path="/static",
)
app.secret_key = __import__("secrets").token_bytes(32)
app.config["MAX_CONTENT_LENGTH"] = 300 * 1024 * 1024
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.IMAGENET_MEAN, std=cfg.IMAGENET_STD),
    ]
)


def read_admin_password() -> str:
    """Read APP_PASSWORD from the local Streamlit-compatible TOML file."""
    if not SECRETS_FILE.exists():
        return ""
    for line in SECRETS_FILE.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() == "APP_PASSWORD":
            try:
                parsed = ast.literal_eval(value.strip())
                return parsed if isinstance(parsed, str) else ""
            except (ValueError, SyntaxError):
                return ""
    return ""


MODEL_CONFIG_KEYS = {
    "num_classes",
    "img_size",
    "embed_dim",
    "mamba_depth",
    "mamba_d_state",
    "mamba_d_conv",
    "mamba_expand",
    "mamba_drop_path",
    "snn_timesteps",
    "snn_tau",
    "snn_vth",
    "snn_surrogate",
    "cnn_base_width",
}


def build_model(model_config=None):
    """Build the architecture recorded in a training checkpoint, when supplied."""
    model_config = model_config or {}
    if not isinstance(model_config, dict):
        raise ValueError("Checkpoint model_config must be a dictionary.")

    model_args = {
        "num_classes": cfg.NUM_CLASSES,
        "img_size": cfg.IMG_SIZE,
        "embed_dim": cfg.EMBED_DIM,
        "mamba_depth": cfg.MAMBA_DEPTH,
        "mamba_d_state": cfg.MAMBA_D_STATE,
        "mamba_d_conv": cfg.MAMBA_D_CONV,
        "mamba_expand": cfg.MAMBA_EXPAND,
        "mamba_drop_path": 0.0,
        "snn_timesteps": cfg.SNN_TIMESTEPS,
        "snn_tau": cfg.SNN_TAU,
        "snn_vth": cfg.SNN_VTH,
        "snn_surrogate": cfg.SNN_SURROGATE,
        "cnn_base_width": cfg.CNN_BASE_WIDTH,
    }
    # Keep only constructor settings; checkpoints can also contain training metadata.
    model_args.update({key: value for key, value in model_config.items() if key in MODEL_CONFIG_KEYS})
    if model_args["num_classes"] != len(cfg.CLASS_NAMES):
        raise ValueError(
            f"This app supports {len(cfg.CLASS_NAMES)} classes, but the checkpoint uses "
            f"{model_args['num_classes']}. The class labels must match before upload."
        )
    if model_args["img_size"] != cfg.IMG_SIZE:
        raise ValueError(
            f"This app expects {cfg.IMG_SIZE}×{cfg.IMG_SIZE} images, but the checkpoint "
            f"uses {model_args['img_size']}×{model_args['img_size']}."
        )
    return SpikeFusion(
        **model_args,
    ).to(DEVICE)


def load_model_from_bytes(checkpoint_bytes: bytes):
    try:
        numpy_scalar = np._core.multiarray.scalar
    except AttributeError:
        numpy_scalar = np.core.multiarray.scalar
    safe_types = [numpy_scalar, np.dtype]
    numpy_dtypes = getattr(np, "dtypes", None)
    float64_dtype = getattr(numpy_dtypes, "Float64DType", None)
    if float64_dtype is not None:
        safe_types.append(float64_dtype)

    with torch.serialization.safe_globals(safe_types):
        checkpoint = torch.load(
            io.BytesIO(checkpoint_bytes), map_location=DEVICE, weights_only=True
        )
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must contain a PyTorch state dictionary.")

    model_config = checkpoint.get("model_config", {})
    model = build_model(model_config)
    state_dict = checkpoint.get(
        "model_state",
        checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint)),
    )
    if not isinstance(state_dict, dict) or not state_dict or not all(
        isinstance(value, torch.Tensor) for value in state_dict.values()
    ):
        raise ValueError(
            "Could not find model weights. Expected 'model_state', 'model_state_dict', "
            "'state_dict', or a weights-only checkpoint."
        )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def ensure_model():
    global MODEL
    with MODEL_LOCK:
        if MODEL is not None:
            return MODEL
        if ACTIVE_CHECKPOINT.exists():
            MODEL = load_model_from_bytes(ACTIVE_CHECKPOINT.read_bytes())
        return MODEL


def image_tensor(image: Image.Image):
    image = ImageOps.exif_transpose(image).convert("RGB")
    display_image = image.resize((cfg.IMG_SIZE, cfg.IMG_SIZE))
    return IMAGE_TRANSFORM(image).unsqueeze(0).to(DEVICE), display_image


def image_data_url(image: Image.Image | np.ndarray, fmt="PNG") -> str:
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{encoded}"


def classify(image: Image.Image):
    tensor, display_image = image_tensor(image)
    with MODEL_LOCK, torch.inference_mode():
        logits, _ = ensure_model()(tensor)
        probabilities = F.softmax(logits, dim=1)[0].detach().cpu()
    pred_index = int(probabilities.argmax())
    return {
        "label": cfg.CLASS_NAMES[pred_index],
        "confidence": float(probabilities[pred_index]),
        "probabilities": [
            {"label": name, "value": float(probabilities[i])}
            for i, name in enumerate(cfg.CLASS_NAMES)
        ],
        "display_image": display_image,
        "tensor": tensor,
        "pred_index": pred_index,
    }


def make_gradcam(model, tensor, display_image, target_class):
    gradcam = GradCAM(model, target_layer_name="cnn_branch.stage6")
    try:
        cam, _ = gradcam.generate(tensor, target_class=target_class)
        original = np.asarray(display_image, dtype=np.uint8)
        overlay = gradcam.overlay(original, cam)
    finally:
        gradcam.remove_hooks()
    heatmap = (cam * 255).clip(0, 255).astype(np.uint8)
    return original, heatmap, overlay


def video_frames(video_path: str, change_threshold: float, min_gap: float, max_gap: float):
    """Yield candidate frames using low-resolution image changes plus a time fallback."""
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError("This video could not be opened. Try an MP4 video.")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 0:
        fps = 25.0

    previous_small = None
    last_selected_time = -max_gap
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = index / fps
            index += 1
            small = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            elapsed = timestamp - last_selected_time
            changed = False
            if previous_small is not None and elapsed >= min_gap:
                changed = float(cv2.absdiff(gray, previous_small).mean()) >= change_threshold
            if previous_small is None or changed or elapsed >= max_gap:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield timestamp, Image.fromarray(rgb)
                last_selected_time = timestamp
            previous_small = gray
    finally:
        capture.release()


def require_admin():
    return bool(session.get("is_admin"))


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/api/status")
def status():
    try:
        model_ready = ensure_model() is not None
        model_error = None
    except Exception as error:  # Report checkpoint errors without breaking public uploads.
        model_ready = False
        model_error = str(error)
    return jsonify(
        {
            "model_ready": model_ready,
            "model_error": model_error,
            "admin_configured": bool(read_admin_password()),
            "admin_authenticated": require_admin(),
            "video_settings": VIDEO_SETTINGS,
            "labels": cfg.CLASS_NAMES,
        }
    )


@app.post("/api/admin/login")
def admin_login():
    expected = read_admin_password()
    supplied = request.get_json(silent=True) or {}
    if not expected:
        return jsonify({"error": "Set APP_PASSWORD in .streamlit/secrets.toml first."}), 400
    if not hmac.compare_digest(str(supplied.get("password", "")), expected):
        return jsonify({"error": "That passcode did not match."}), 401
    session["is_admin"] = True
    return jsonify({"ok": True})


@app.post("/api/admin/logout")
def admin_logout():
    session.pop("is_admin", None)
    return jsonify({"ok": True})


@app.post("/api/admin/settings")
def admin_settings():
    if not require_admin():
        return jsonify({"error": "Admin access required."}), 401
    values = request.get_json(silent=True) or {}
    try:
        change_threshold = min(40.0, max(3.0, float(values["change_threshold"])))
        min_gap = min(3.0, max(0.2, float(values["min_gap"])))
        max_gap = min(10.0, max(1.0, float(values["max_gap"])))
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Invalid frame selection settings."}), 400
    VIDEO_SETTINGS.update(
        change_threshold=change_threshold,
        min_gap=min_gap,
        max_gap=max(max_gap, min_gap),
    )
    return jsonify({"ok": True, "video_settings": VIDEO_SETTINGS})


@app.post("/api/admin/checkpoint")
def upload_checkpoint():
    global MODEL
    if not require_admin():
        return jsonify({"error": "Admin access required."}), 401
    checkpoint_file = request.files.get("checkpoint")
    if not checkpoint_file or not checkpoint_file.filename:
        return jsonify({"error": "Choose a .pth, .pt, or PyTorch .zip checkpoint."}), 400
    checkpoint_bytes = checkpoint_file.read()
    try:
        candidate_model = load_model_from_bytes(checkpoint_bytes)
    except Exception as error:
        return jsonify({"error": f"Could not load this checkpoint: {error}"}), 400

    ACTIVE_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = ACTIVE_CHECKPOINT.with_suffix(".pth.tmp")
    temporary_path.write_bytes(checkpoint_bytes)
    temporary_path.replace(ACTIVE_CHECKPOINT)
    with MODEL_LOCK:
        MODEL = candidate_model
    return jsonify({"ok": True, "message": "Checkpoint saved and is now active."})


@app.post("/api/predict/image")
def predict_image():
    try:
        model = ensure_model()
    except Exception as error:
        return jsonify({"error": f"Saved checkpoint could not be loaded: {error}"}), 500
    if model is None:
        return jsonify({"error": "An admin must upload a trained checkpoint before predictions can run."}), 409
    uploaded = request.files.get("file")
    if not uploaded:
        return jsonify({"error": "Choose a photo first."}), 400
    try:
        image = Image.open(uploaded.stream)
        with MODEL_LOCK:
            result = classify(image)
            original, heatmap, overlay = make_gradcam(
                model,
                result["tensor"],
                result["display_image"],
                result["pred_index"],
            )
        return jsonify(
            {
                "label": result["label"],
                "confidence": result["confidence"],
                "probabilities": result["probabilities"],
                "original": image_data_url(original),
                "heatmap": image_data_url(heatmap),
                "overlay": image_data_url(overlay),
            }
        )
    except Exception as error:
        return jsonify({"error": f"Could not analyze this photo: {error}"}), 400


@app.post("/api/predict/video")
def predict_video():
    try:
        model = ensure_model()
    except Exception as error:
        return jsonify({"error": f"Saved checkpoint could not be loaded: {error}"}), 500
    if model is None:
        return jsonify({"error": "An admin must upload a trained checkpoint before predictions can run."}), 409
    uploaded = request.files.get("file")
    if not uploaded:
        return jsonify({"error": "Choose a video first."}), 400

    try:
        settings = VIDEO_SETTINGS.copy()
        suffix = Path(uploaded.filename or "video.mp4").suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
            temporary_file.write(uploaded.read())
            temporary_path = temporary_file.name
        results = []
        try:
            for timestamp, frame in video_frames(
                temporary_path,
                settings["change_threshold"],
                settings["min_gap"],
                settings["max_gap"],
            ):
                with MODEL_LOCK:
                    prediction = classify(frame)
                preview = prediction["display_image"].copy()
                preview.thumbnail((360, 360))
                results.append(
                    {
                        "seconds": timestamp,
                        "time": f"{int(timestamp) // 60:02d}:{int(timestamp) % 60:02d}",
                        "label": prediction["label"],
                        "confidence": prediction["confidence"],
                        "preview": image_data_url(preview, fmt="JPEG"),
                    }
                )
        finally:
            Path(temporary_path).unlink(missing_ok=True)

        if not results:
            return jsonify({"error": "No frames could be read from this video."}), 400
        counts = {}
        for result in results:
            counts[result["label"]] = counts.get(result["label"], 0) + 1
        return jsonify({"results": results, "counts": counts, "selected_count": len(results)})
    except Exception as error:
        return jsonify({"error": f"Could not analyze this video: {error}"}), 400


@app.errorhandler(413)
def upload_too_large(_error):
    return jsonify({"error": "This file is too large. The current upload limit is 300 MB."}), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)

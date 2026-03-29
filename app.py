
#
#  (recommended — fixes encoding on Windows):
#       $env:PYTHONIOENCODING="utf-8"; venv\Scripts\python app.py
#
#
#  STEP 3 — Open your browser at:
#       http://127.0.0.1:5000
#
# =============================================================

"""
Flask Disease Detection App
============================
Loads the trained EfficientNet checkpoint and serves a web UI
for single-image plant / flower disease classification.
"""
import sys
import os
import io
import json
import base64
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import (
    efficientnet_b0, EfficientNet_B0_Weights,
    efficientnet_b1, EfficientNet_B1_Weights,
    efficientnet_b2, EfficientNet_B2_Weights,
    efficientnet_b3, EfficientNet_B3_Weights,
    efficientnet_b4, EfficientNet_B4_Weights,
    efficientnet_b5, EfficientNet_B5_Weights,
    efficientnet_b6, EfficientNet_B6_Weights,
    efficientnet_b7, EfficientNet_B7_Weights,
)

from flask import Flask, request, jsonify, send_from_directory

# ─────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────
CHECKPOINTS_DIR = Path("./checkpoints")
UPLOAD_FOLDER   = Path("./uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp", "gif"}

EFFICIENTNET_INPUT_SIZES = {
    "b0": 224, "b1": 240, "b2": 260, "b3": 300,
    "b4": 380, "b5": 456, "b6": 528, "b7": 600,
}

EFFICIENTNET_REGISTRY = {
    "b0": (efficientnet_b0, EfficientNet_B0_Weights.IMAGENET1K_V1),
    "b1": (efficientnet_b1, EfficientNet_B1_Weights.IMAGENET1K_V1),
    "b2": (efficientnet_b2, EfficientNet_B2_Weights.IMAGENET1K_V1),
    "b3": (efficientnet_b3, EfficientNet_B3_Weights.IMAGENET1K_V1),
    "b4": (efficientnet_b4, EfficientNet_B4_Weights.IMAGENET1K_V1),
    "b5": (efficientnet_b5, EfficientNet_B5_Weights.IMAGENET1K_V1),
    "b6": (efficientnet_b6, EfficientNet_B6_Weights.IMAGENET1K_V1),
    "b7": (efficientnet_b7, EfficientNet_B7_Weights.IMAGENET1K_V1),
}

# Friendly display names for class labels
CLASS_DISPLAY = {
    "Chrysanthemum_Bacterial_Leaf_Spot": "Chrysanthemum - Bacterial Leaf Spot",
    "Chrysanthemum_Healthy":             "Chrysanthemum - Healthy",
    "Chrysanthemum_Septoria_Leaf_Spot":  "Chrysanthemum - Septoria Leaf Spot",
    "Jasmine_Healthy":                   "Jasmine - Healthy",
    "Jasmine_Multiple":                  "Jasmine - Multiple Diseases",
    "Jasmine_Rust":                      "Jasmine - Rust",
}

# ─────────────────────────────────────────────────────────
#  Model helpers (mirror the training code)
# ─────────────────────────────────────────────────────────
def build_inference_model(version: str, num_classes: int, dropout_rate: float = 0.35):
    fn, weights = EFFICIENTNET_REGISTRY[version]
    model = fn(weights=None)                  # we'll load custom weights
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, 1024),
        nn.BatchNorm1d(1024),
        nn.GELU(),
        nn.Dropout(p=dropout_rate * 0.5),
        nn.Linear(1024, num_classes),
    )
    return model


def get_eval_transform(img_size: int):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.143)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def find_checkpoint():
    """Return the first .pth file found in checkpoints dir."""
    pths = list(CHECKPOINTS_DIR.glob("*.pth"))
    if not pths:
        return None
    return pths[0]


# ─────────────────────────────────────────────────────────
#  Load model at startup
# ─────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = None
class_names    = []
model_version  = "b0"
eval_transform = None
checkpoint_name = ""

def load_model():
    global model, class_names, model_version, eval_transform, checkpoint_name

    ckpt_path = find_checkpoint()
    if ckpt_path is None:
        print("[WARN] No checkpoint found in ./checkpoints - model not loaded.")
        return False

    checkpoint_name = ckpt_path.name
    print(f"  Loading checkpoint : {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)

    # Pull metadata saved during training
    class_names   = checkpoint.get("class_names", [])
    config        = checkpoint.get("config",      {})
    model_version = config.get("model_version", "b0")
    dropout_rate  = config.get("dropout_rate",  0.35)

    print(f"  Model version  : EfficientNet-{model_version.upper()}")
    print(f"  Classes        : {class_names}")
    print(f"  Device         : {device}")

    # Build model skeleton & load weights
    num_classes = len(class_names)
    m = build_inference_model(model_version, num_classes, dropout_rate)

    # Handle DataParallel-wrapped state dicts
    state = checkpoint["model_state"]
    new_state = {}
    for k, v in state.items():
        new_key = k.replace("module.", "", 1) if k.startswith("module.") else k
        new_state[new_key] = v

    m.load_state_dict(new_state, strict=True)
    m.eval()
    m.to(device)
    model = m

    img_size = EFFICIENTNET_INPUT_SIZES[model_version]
    eval_transform = get_eval_transform(img_size)

    print(f"  [OK] Model ready  |  Input: {img_size}x{img_size}")
    return True


# ─────────────────────────────────────────────────────────
#  Inference helper
# ─────────────────────────────────────────────────────────
def predict_image(pil_img: Image.Image):
    """Returns list of {label, display_label, confidence} dicts (top-N)."""
    tensor = eval_transform(pil_img).unsqueeze(0).to(device, non_blocking=True)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    top_n  = min(len(class_names), 6)
    top_idx = np.argsort(probs)[::-1][:top_n]

    predictions = []
    for idx in top_idx:
        label = class_names[idx]
        predictions.append({
            "label":         label,
            "display_label": CLASS_DISPLAY.get(label, label.replace("_", " ")),
            "confidence":    float(probs[idx]),
            "percentage":    round(float(probs[idx]) * 100, 2),
        })
    return predictions


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ─────────────────────────────────────────────────────────
#  Flask app
# ─────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16 MB limit


@app.route("/")
def index():
    return send_from_directory("templates", "index.html")


@app.route("/api/status")
def status():
    return jsonify({
        "model_loaded":   model is not None,
        "model_version":  f"EfficientNet-{model_version.upper()}",
        "classes":        class_names,
        "checkpoint":     checkpoint_name,
        "device":         str(device),
        "num_classes":    len(class_names),
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Check checkpoints directory."}), 503

    if "file" not in request.files:
        return jsonify({"error": "No file part in request."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"File type not allowed. Use: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    try:
        img_bytes = file.read()
        pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Encode thumbnail for display in UI
        thumb = pil_img.copy()
        thumb.thumbnail((400, 400))
        buf = io.BytesIO()
        thumb.save(buf, format="JPEG", quality=85)
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        predictions = predict_image(pil_img)

        top = predictions[0]
        is_healthy = "healthy" in top["label"].lower()

        return jsonify({
            "success":     True,
            "image_b64":   img_b64,
            "predictions": predictions,
            "top_label":   top["display_label"],
            "top_confidence": top["percentage"],
            "is_healthy":  is_healthy,
            "filename":    file.filename,
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


# ─────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  Plant / Flower Disease Detection Flask App")
    print("=" * 55)
    success = load_model()
    if not success:
        print("  [WARN] Starting without model - upload won't work until")
        print("         a .pth checkpoint is placed in ./checkpoints/")
    print("\n  Open: http://127.0.0.1:5000")
    print("=" * 55 + "\n")
    app.run(debug=False, host="0.0.0.0", port=5000)

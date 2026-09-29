"""Flask inference server. Run python app.py and visit localhost:5000."""
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge

from dataset import CLASSES, DISPLAY_NAMES
from imaging import MAX_UPLOAD_BYTES, image_array


def load_bundle(folder):
    from tensorflow import keras
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    if (metadata.get("schema_version") != 1 or metadata.get("classes") != CLASSES
            or metadata.get("preprocessing") != "pillow_rgb_bilinear_0_255"
            or metadata.get("image_size") != 224 or not metadata.get("training_complete")):
        raise ValueError("Incompatible metadata or incomplete training.")
    model = keras.models.load_model(folder / "model.keras", compile=False)
    if tuple(model.input_shape[1:]) != (224, 224, 3) or model.output_shape[-1] != 5:
        raise ValueError("Model shape does not match the class mapping.")
    return model, metadata


def create_app(model_dir=None, bundle=None):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES + 1024 * 1024
    folder = Path(model_dir or os.environ.get("MODEL_DIR", str(Path(__file__).parent / "artifacts")))
    model, metadata = None, {}
    state = "Awaiting trained model"
    if bundle is not None:  # Dependency injection for endpoint tests.
        model, metadata = bundle
        state = "Trained model loaded"
    elif all((folder / name).is_file() for name in ["model.keras", "metadata.json"]):
        try:
            model, metadata = load_bundle(folder)
            state = "Trained model loaded"
        except Exception:
            app.logger.exception("Model loading failed")
            state = "Model could not be loaded. Check server logs and model files."
    
    inference_lock = threading.Lock()

    @app.get("/")
    def index():
        return render_template("index.html", labels=DISPLAY_NAMES)

    @app.get("/api/status")
    def status():
        return jsonify(ready=model is not None, message=state, classes=DISPLAY_NAMES,
                       evaluation_scope=metadata.get("evaluation_scope"))

    @app.post("/api/predict")
    def predict():
        if model is None:
            return jsonify(error="Train and load a model before requesting predictions."), 503
        upload = request.files.get("image")
        if upload is None or not upload.filename:
            return jsonify(error="Choose a JPEG or PNG tissue image."), 400
        try:
            payload = upload.read(MAX_UPLOAD_BYTES + 1)
            started = time.perf_counter()
            pixels = image_array(payload)
            with inference_lock:
                scores = np.asarray(model.predict(pixels[None, ...], verbose=0))[0]
            if (scores.shape != (5,) or not np.isfinite(scores).all() or (scores < 0).any()
                    or (scores > 1).any() or not np.isclose(scores.sum(), 1, atol=1e-3)):
                raise RuntimeError("Invalid model scores")
            winner = CLASSES[int(scores.argmax())]
            return jsonify(predicted_class=winner, label=DISPLAY_NAMES[winner],
                           scores=[{"class": label, "label": DISPLAY_NAMES[label], "score": float(score)}
                                   for label, score in zip(CLASSES, scores)],
                           processing_ms=round((time.perf_counter() - started) * 1000, 1),
                           model_created_at=metadata.get("created_at"),
                           evaluation_scope=metadata.get("evaluation_scope"),
                           purpose="Research only; not a clinical diagnosis.")
        except ValueError as error:
            return jsonify(error=str(error)), 400
        except Exception:
            app.logger.exception("Prediction failed")
            return jsonify(error="Prediction failed. Check the server logs and try again."), 500

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(error):
        return jsonify(error="Choose an image smaller than 10 MB."), 413

    @app.after_request
    def security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        return response

    return app


if __name__ == "__main__":
    from waitress import serve
    serve(create_app(), host="127.0.0.1", port=5000)

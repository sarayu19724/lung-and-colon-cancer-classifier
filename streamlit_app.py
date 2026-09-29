"""Run: python -m streamlit run streamlit_app.py"""
import hashlib
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import streamlit as st

from app import load_bundle
from dataset import CLASSES, DISPLAY_NAMES
from imaging import image_array, read_image

st.set_page_config(page_title="TissueLens", page_icon="🔬", layout="wide")


@st.cache_resource
def cached_model(directory, model_stamp, metadata_stamp):
    model, metadata = load_bundle(Path(directory))
    return model, metadata, threading.Lock()


st.caption("TISSUELENS · LUNG & COLON HISTOPATHOLOGY")
st.title("A closer look at tissue.")
st.write("Upload a microscopy tissue patch to explore five predicted tissue-class scores.")
st.caption("Research and education only. Not validated for clinical diagnosis. "
           "Use tissue microscopy images, not CT scans, X-rays or photographs.")

folder = Path(os.environ.get("MODEL_DIR", str(Path(__file__).parent / "artifacts")))
model_file, metadata_file = folder / "model.keras", folder / "metadata.json"
if not model_file.is_file() or not metadata_file.is_file():
    st.info("The saved model is missing. Include artifacts/model.keras and artifacts/metadata.json "
            "with this app. No dataset or retraining is required to run inference.")
    st.stop()

try:
    stamps = (model_file.stat().st_mtime_ns, metadata_file.stat().st_mtime_ns)
    with st.spinner("Loading trained model…"):
        model, metadata, lock = cached_model(str(folder), *stamps)
except Exception:
    st.error("The model could not be loaded. Check the server logs and install requirements.txt.")
    import logging
    logging.exception("Streamlit model loading failed")
    st.stop()


left, right = st.columns(2, gap="large")
payload = None
with left:
    st.subheader("1. Upload a tissue patch")
    upload = st.file_uploader("JPEG or PNG · up to 10 MB", type=["jpg", "jpeg", "png"])
    if upload is not None:
        try:
            payload = upload.getvalue()
            preview = read_image(payload)
            st.image(preview, use_container_width=True)
        except ValueError as error:
            st.error(str(error))
            payload = None
    st.caption("Uploads are processed in memory and are not saved by this app.")

# Results belong to one upload and one model revision, never to another session.
identity = (hashlib.sha256(payload).hexdigest(), str(folder), stamps) if payload else None
if st.session_state.get("result_identity") != identity:
    st.session_state.pop("prediction", None)
    st.session_state["result_identity"] = identity

with right:
    st.subheader("2. Explore the prediction")
    if st.button("Analyze tissue patch", type="primary", disabled=payload is None):
        try:
            with st.spinner("Analyzing tissue patterns…"):
                started = time.perf_counter()
                pixels = image_array(payload)
                with lock:
                    scores = np.asarray(model.predict(pixels[None, ...], verbose=0))[0]
                if (scores.shape != (5,) or not np.isfinite(scores).all()
                        or (scores < 0).any() or (scores > 1).any()
                        or not np.isclose(scores.sum(), 1, atol=1e-3)):
                    raise ValueError("Invalid model scores.")
                winner = CLASSES[int(scores.argmax())]
                st.session_state.prediction = {
                    "predicted_class": winner, "label": DISPLAY_NAMES[winner],
                    "scores": [{"label": DISPLAY_NAMES[label], "score": float(score)}
                               for label, score in zip(CLASSES, scores)],
                    "processing_ms": round((time.perf_counter() - started) * 1000, 1),
                    "model_created_at": metadata.get("created_at"),
                    "purpose": "Research only; not a clinical diagnosis.",
                }
        except Exception:
            st.session_state.pop("prediction", None)
            st.error("Prediction failed. Try another image and check the server logs.")
            import logging
            logging.exception("Streamlit prediction failed")
    result = st.session_state.get("prediction")
    if result:
        st.caption("HIGHEST-SCORING CLASS")
        st.subheader(result["label"])
        for item in sorted(result["scores"], key=lambda item: item["score"], reverse=True):
            st.progress(item["score"], text=f"{item['label']} · {item['score']:.1%}")
        st.caption(f"Server processing: {result['processing_ms']} ms")
        st.download_button("Download result", json.dumps(result, indent=2),
                           file_name="tissue_prediction.json", mime="application/json")
        st.caption("Scores are not calibrated probabilities of disease. "
                   "Unrelated images can also receive high scores.")
    else:
        st.info("Choose an image, then click Analyze tissue patch.")

st.divider()
st.caption("SUPPORTED CLASSES")
st.write(" · ".join(DISPLAY_NAMES.values()))

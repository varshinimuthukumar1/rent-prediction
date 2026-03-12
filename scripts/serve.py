"""
Flask app: load .joblib model once, expose POST /predict to send JSON and get predicted rent(s).
Run with: gunicorn -w 1 -b 0.0.0.0:8000 scripts.serve:app
"""
from __future__ import annotations

import os
from pathlib import Path

# Ensure project root on path when running as gunicorn module
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_ROOT))

from flask import Flask, request, jsonify

from scripts.inference import load_model, run_inference

app = Flask(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "models/rent_price_model.joblib")
_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = load_model(MODEL_PATH)
    return _pipeline


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept JSON body: one object or array of objects (listing features).
    Returns {"predictions_eur": [<rent>, ...]}.
    """
    try:
        data = request.get_json(force=True, silent=False)
    except Exception as e:
        return jsonify({"error": f"Invalid JSON: {e}"}), 400
    if data is None:
        return jsonify({"error": "Request body must be JSON"}), 400
    try:
        pipeline = get_pipeline()
        predictions_eur = run_inference(pipeline, data)
        return jsonify({"predictions_eur": predictions_eur})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

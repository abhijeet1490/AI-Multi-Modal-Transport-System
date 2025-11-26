"""
app.py

Flask backend for the "AI-Powered Multi-Modal Transport System".

Responsibilities:
    - Load the trained logistics model (`logistics_model.pkl`) if available.
    - Expose a `/predict_route` POST endpoint that accepts:
          { "distance": <float>, "weight": <float>, "traffic": <int> }
    - Return predicted or mock:
          { "success": true, "data": { "cost": <float>, "time": <float>, "co2": <float> } }

If the model file is missing or fails to load, the API switches to MOCK_MODE
and uses simple deterministic formulas to keep the frontend functional.
"""

import json
import pickle
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS


MODEL_PATH = "logistics_model.pkl"


@dataclass
class PredictionResult:
    cost: float
    time: float
    co2: float


def load_model(path: str = MODEL_PATH) -> Tuple[Optional[Any], bool]:
    """
    Attempt to load the trained model from disk.

    Returns:
        (model, mock_mode)
        - model: the loaded model instance or None
        - mock_mode: True if we should fall back to mock predictions
    """
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        print(f"[INFO] Loaded model from '{path}'.")
        return model, False
    except FileNotFoundError:
        print(f"[WARN] Model file '{path}' not found. Starting in MOCK_MODE.")
    except Exception as exc:
        print(f"[ERROR] Failed to load model from '{path}': {exc}")

    return None, True


def make_mock_prediction(
    distance_km: float, weight_kg: float, traffic_level: float
) -> PredictionResult:
    """
    Simple deterministic formulas used when the ML model is not available.
    These are intentionally straightforward and transparent.
    """
    # Cost: similar structure to training script but simplified
    base_cost_per_km = 1.4
    cost_per_kg = 0.04
    traffic_surcharge = 6.0

    cost = (
        distance_km * base_cost_per_km
        + weight_kg * cost_per_kg
        + traffic_level * traffic_surcharge
    )

    # Time: assume base speed reduced by traffic
    base_speed_kmph = 65.0
    traffic_penalty = 2.5
    effective_speed = max(20.0, base_speed_kmph - traffic_penalty * traffic_level)
    time_hours = distance_km / effective_speed

    # CO2: distance + weight + traffic factor
    co2_per_km = 0.16
    co2_per_kg = 0.00035
    co2_traffic_factor = 1.2

    co2_kg = (
        distance_km * co2_per_km
        + weight_kg * co2_per_kg
        + traffic_level * co2_traffic_factor
    )

    return PredictionResult(cost=cost, time=time_hours, co2=co2_kg)


def make_model_prediction(
    model: Any, distance_km: float, weight_kg: float, traffic_level: float
) -> PredictionResult:
    """
    Use the trained model to make a prediction.
    """
    features = np.array([[distance_km, weight_kg, traffic_level]], dtype=float)
    cost, time_hours, co2_kg = model.predict(features)[0]
    return PredictionResult(cost=float(cost), time=float(time_hours), co2=float(co2_kg))


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)  # enable CORS for all routes

    model, mock_mode = load_model()

    @app.route("/health", methods=["GET"])
    def health() -> Any:
        return jsonify(
            {
                "status": "ok",
                "mock_mode": mock_mode,
            }
        )

    @app.route("/predict_route", methods=["POST"])
    def predict_route() -> Any:
        nonlocal model, mock_mode

        try:
            # Accept both application/json and text/json
            payload: Dict[str, Any] = request.get_json(force=True, silent=False)
        except Exception:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Invalid JSON payload.",
                    }
                ),
                400,
            )

        # Basic validation and type coercion
        missing_fields = [
            field
            for field in ("distance", "weight", "traffic")
            if field not in payload
        ]
        if missing_fields:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": f"Missing fields: {', '.join(missing_fields)}",
                    }
                ),
                400,
            )

        try:
            distance_km = float(payload["distance"])
            weight_kg = float(payload["weight"])
            traffic_level = float(payload["traffic"])
        except (TypeError, ValueError):
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "distance, weight, and traffic must be numeric.",
                    }
                ),
                400,
            )

        # Minimal sanity checks
        if distance_km <= 0 or weight_kg <= 0:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "distance and weight must be positive.",
                    }
                ),
                400,
            )

        if traffic_level < 1 or traffic_level > 10:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "traffic must be between 1 and 10.",
                    }
                ),
                400,
            )

        # Decide between real model and mock formulas
        try:
            if not mock_mode and model is not None:
                result = make_model_prediction(
                    model, distance_km, weight_kg, traffic_level
                )
            else:
                result = make_mock_prediction(
                    distance_km, weight_kg, traffic_level
                )
        except Exception as exc:
            # If prediction fails for any reason, fall back to mock mode
            print(f"[ERROR] Prediction failed with model: {exc}. Falling back to MOCK.")
            mock_mode = True
            result = make_mock_prediction(distance_km, weight_kg, traffic_level)

        response = {
            "success": True,
            "data": {
                "cost": result.cost,
                "time": result.time,
                "co2": result.co2,
            },
            "mock_mode": mock_mode,
        }
        return jsonify(response)

    return app


app = create_app()


if __name__ == "__main__":
    # For local development only; in production use a proper WSGI server.
    app.run(host="0.0.0.0", port=5001, debug=True)



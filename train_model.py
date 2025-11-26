"""
train_model.py

Generates synthetic logistics data, trains a RandomForest-based
multi-output regression model, and saves it to `logistics_model.pkl`.

Inputs (features):
    - distance_km: float, shipment distance in kilometers
    - weight_kg: float, shipment weight in kilograms
    - traffic_level: int, 1 (very light) to 10 (heavy congestion)

Targets (outputs):
    - cost_usd
    - time_hours
    - co2_kg
"""

import pickle
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor


RANDOM_SEED = 42
N_SAMPLES = 1000


@dataclass
class SyntheticDataConfig:
    n_samples: int = N_SAMPLES
    distance_min_km: float = 10.0
    distance_max_km: float = 2000.0
    weight_min_kg: float = 10.0
    weight_max_kg: float = 5000.0
    traffic_min: int = 1
    traffic_max: int = 10


def generate_synthetic_data(
    config: SyntheticDataConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic shipment data and targets.

    Returns:
        X: np.ndarray of shape (n_samples, 3)
           Columns: [distance_km, weight_kg, traffic_level]
        y: np.ndarray of shape (n_samples, 3)
           Columns: [cost_usd, time_hours, co2_kg]
    """
    rng = np.random.default_rng(RANDOM_SEED)

    # Features
    distance_km = rng.uniform(
        config.distance_min_km, config.distance_max_km, size=config.n_samples
    )
    weight_kg = rng.uniform(
        config.weight_min_kg, config.weight_max_kg, size=config.n_samples
    )
    traffic_level = rng.integers(
        config.traffic_min, config.traffic_max + 1, size=config.n_samples
    )

    # --- Target generation (simple but "realistic-ish") ---
    # Cost model:
    #   base cost per km + heavier shipments cost more
    #   traffic adds a surcharge due to delays / fuel / driver time
    base_cost_per_km = 1.5
    cost_per_kg = 0.05
    traffic_surcharge = 8.0  # per traffic level

    cost_usd = (
        distance_km * base_cost_per_km
        + weight_kg * cost_per_kg
        + traffic_level * traffic_surcharge
    )
    cost_usd += rng.normal(0, 50.0, size=config.n_samples)  # noise

    # Time model:
    #   Effective speed decreases with traffic.
    #   Clamp speed to a minimum to avoid unrealistic huge times.
    base_speed_kmph = 70.0
    traffic_penalty_per_level = 3.0
    effective_speed = base_speed_kmph - traffic_penalty_per_level * traffic_level
    effective_speed = np.clip(effective_speed, 20.0, None)

    time_hours = distance_km / effective_speed
    time_hours += rng.normal(0, 0.3, size=config.n_samples)  # noise
    time_hours = np.clip(time_hours, 0.1, None)

    # CO2 model:
    #   Proportional to distance and weight; traffic increases idling & stop-go.
    co2_per_km = 0.18  # kg CO2 per km (approx for a truck)
    co2_per_kg = 0.0004
    co2_traffic_factor = 1.5  # per traffic level

    co2_kg = (
        distance_km * co2_per_km
        + weight_kg * co2_per_kg
        + traffic_level * co2_traffic_factor
    )
    co2_kg += rng.normal(0, 10.0, size=config.n_samples)  # noise
    co2_kg = np.clip(co2_kg, 0.0, None)

    # Stack features and targets
    X = np.vstack([distance_km, weight_kg, traffic_level]).T
    y = np.vstack([cost_usd, time_hours, co2_kg]).T

    return X, y


def train_model(X: np.ndarray, y: np.ndarray) -> MultiOutputRegressor:
    """
    Train a RandomForestRegressor wrapped in MultiOutputRegressor.
    """
    base_rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model = MultiOutputRegressor(base_rf)
    model.fit(X, y)
    return model


def save_model(model: MultiOutputRegressor, path: str = "logistics_model.pkl") -> None:
    """
    Pickle the trained model to disk.
    """
    with open(path, "wb") as f:
        pickle.dump(model, f)


def run_sample_prediction(model: MultiOutputRegressor) -> None:
    """
    Run and print a sample prediction to verify the pipeline.
    """
    # Example shipment:
    #   750 km distance, 1,200 kg weight, medium traffic level 5
    sample_features = np.array([[750.0, 1200.0, 5]])
    prediction = model.predict(sample_features)[0]
    cost_usd, time_hours, co2_kg = prediction

    print("\nSample Prediction (single shipment):")
    print(f"  Input  -> distance_km=750, weight_kg=1200, traffic_level=5")
    print(f"  Output -> cost_usd={cost_usd:,.2f}")
    print(f"           time_hours={time_hours:.2f}")
    print(f"           co2_kg={co2_kg:,.2f}\n")


def main() -> None:
    print("Generating synthetic data...")
    config = SyntheticDataConfig()
    X, y = generate_synthetic_data(config)

    print("Training RandomForest multi-output model...")
    model = train_model(X, y)

    print("Saving model to 'logistics_model.pkl'...")
    save_model(model, path="logistics_model.pkl")

    print("Running a sample prediction...")
    run_sample_prediction(model)

    print("Done.")


if __name__ == "__main__":
    main()



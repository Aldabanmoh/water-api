"""
AgroAqua V2 — Flask API for Irrigation Need Prediction
Deployed on Render: https://water-api-y5bd.onrender.com

Endpoints:
  POST /predict  — Predict irrigation need (mm/day) + MAD alert + volumes
  GET  /health   — Health check
  GET  /info     — Model info, supported crops/soils/methods
"""

import os
import json
import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

# ============================================================
# APP SETUP
# ============================================================

app = Flask(__name__)
CORS(app)

# ============================================================
# LOAD MODEL & METADATA
# ============================================================

MODEL_PATH = os.environ.get("MODEL_PATH", "agroaqua_v2_model.pkl")
ENCODERS_PATH = os.environ.get("ENCODERS_PATH", "agroaqua_v2_encoders.pkl")
METADATA_PATH = os.environ.get("METADATA_PATH", "agroaqua_v2_metadata.json")

model = joblib.load(MODEL_PATH)
label_encoders = joblib.load(ENCODERS_PATH)

with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

FEATURE_COLUMNS = metadata["feature_columns"]
CATEGORICAL_FEATURES = metadata["categorical_features"]

print(f"[AgroAqua V2] Model loaded: {metadata['model_name']}")
print(f"[AgroAqua V2] R²={metadata['metrics']['R2']}, RMSE={metadata['metrics']['RMSE']}")

# ============================================================
# SCIENTIFIC CONSTANTS
# ============================================================

CROPS = {
    "wheat":   {"kc": [0.30, 0.70, 1.15, 0.40], "row_spacing_range": [12, 25],   "plant_spacing_range": [3, 10]},
    "corn":    {"kc": [0.30, 0.75, 1.20, 0.60], "row_spacing_range": [60, 90],   "plant_spacing_range": [20, 35]},
    "rice":    {"kc": [1.05, 1.10, 1.20, 0.90], "row_spacing_range": [20, 30],   "plant_spacing_range": [15, 25]},
    "tomato":  {"kc": [0.60, 0.80, 1.15, 0.80], "row_spacing_range": [60, 150],  "plant_spacing_range": [30, 80]},
    "olive":   {"kc": [0.50, 0.55, 0.65, 0.55], "row_spacing_range": [400, 800], "plant_spacing_range": [400, 800]},
    "alfalfa": {"kc": [0.40, 0.80, 1.20, 1.05], "row_spacing_range": [15, 30],   "plant_spacing_range": [5, 15]},
    "citrus":  {"kc": [0.65, 0.65, 0.70, 0.65], "row_spacing_range": [400, 700], "plant_spacing_range": [300, 600]},
    "potato":  {"kc": [0.50, 0.80, 1.15, 0.75], "row_spacing_range": [60, 90],   "plant_spacing_range": [25, 40]},
}

SOILS = {
    "sandy":     {"field_capacity": 15.0, "wilting_point": 5.0},
    "loam":      {"field_capacity": 30.0, "wilting_point": 12.0},
    "clay_loam": {"field_capacity": 38.0, "wilting_point": 18.0},
    "clay":      {"field_capacity": 42.0, "wilting_point": 22.0},
    "silt":      {"field_capacity": 35.0, "wilting_point": 15.0},
}

IRRIGATION_METHODS = {
    "drip": 0.92,
    "sprinkler": 0.75,
    "flood": 0.55,
    "center_pivot": 0.85,
}

GROWTH_STAGES = ["initial", "development", "mid_season", "late_season"]
GROWTH_STAGE_INDEX = {stage: i for i, stage in enumerate(GROWTH_STAGES)}

CANOPY_STAGE_FACTOR = {
    "initial": 0.15,
    "development": 0.50,
    "mid_season": 0.90,
    "late_season": 0.70,
}

MAD_FRACTION = 0.50


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def compute_canopy_cover(row_spacing_cm, plant_spacing_cm, growth_stage):
    """Compute canopy cover and plant density from spacing."""
    row_m = row_spacing_cm / 100.0
    plant_m = plant_spacing_cm / 100.0
    density = 10000.0 / (row_m * plant_m)
    density_norm = min(max(np.log10(density + 1) / 6.0, 0.05), 1.0)
    stage_factor = CANOPY_STAGE_FACTOR[growth_stage]
    cover = min(max(density_norm * stage_factor, 0.05), 0.98)
    return cover, density


def compute_mad_threshold(field_capacity, wilting_point):
    """MAD threshold = FC - 50% * (FC - WP)"""
    available_water = field_capacity - wilting_point
    return field_capacity - MAD_FRACTION * available_water


def validate_input(data):
    """Validate all required fields and return errors if any."""
    errors = []

    required_fields = [
        "crop_type", "soil_type", "growth_stage", "irrigation_method",
        "temperature", "humidity", "rainfall", "wind_speed", "solar_radiation",
        "soil_moisture", "row_spacing_cm", "plant_spacing_cm",
    ]

    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")

    if errors:
        return errors

    # Validate categorical values
    if data["crop_type"] not in CROPS:
        errors.append(f"Invalid crop_type: {data['crop_type']}. Must be one of: {list(CROPS.keys())}")
    if data["soil_type"] not in SOILS:
        errors.append(f"Invalid soil_type: {data['soil_type']}. Must be one of: {list(SOILS.keys())}")
    if data["growth_stage"] not in GROWTH_STAGES:
        errors.append(f"Invalid growth_stage: {data['growth_stage']}. Must be one of: {GROWTH_STAGES}")
    if data["irrigation_method"] not in IRRIGATION_METHODS:
        errors.append(f"Invalid irrigation_method: {data['irrigation_method']}. Must be one of: {list(IRRIGATION_METHODS.keys())}")

    # Validate numeric ranges
    numeric_checks = {
        "temperature": (-10, 55),
        "humidity": (0, 100),
        "rainfall": (0, 200),
        "wind_speed": (0, 30),
        "solar_radiation": (0, 40),
        "soil_moisture": (0, 60),
        "row_spacing_cm": (1, 1000),
        "plant_spacing_cm": (1, 1000),
    }

    for field, (low, high) in numeric_checks.items():
        val = data.get(field)
        if val is not None:
            try:
                val = float(val)
                if val < low or val > high:
                    errors.append(f"{field} must be between {low} and {high}, got {val}")
            except (ValueError, TypeError):
                errors.append(f"{field} must be a number, got {val}")

    return errors


# ============================================================
# ENDPOINTS
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict irrigation need.

    Required JSON body:
    {
        "crop_type": "tomato",
        "soil_type": "clay_loam",
        "growth_stage": "mid_season",
        "irrigation_method": "drip",
        "temperature": 38.0,
        "humidity": 25.0,
        "rainfall": 0.0,
        "wind_speed": 3.5,
        "solar_radiation": 28.0,
        "soil_moisture": 22.0,
        "row_spacing_cm": 100,
        "plant_spacing_cm": 50,
        "field_area_ha": 2.0  (optional, default=1.0)
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        # Validate input
        errors = validate_input(data)
        if errors:
            return jsonify({"error": "Validation failed", "details": errors}), 400

        # Extract values
        crop_name = data["crop_type"]
        soil_name = data["soil_type"]
        growth_stage = data["growth_stage"]
        irrig_method = data["irrigation_method"]
        field_area_ha = float(data.get("field_area_ha", 1.0))

        crop = CROPS[crop_name]
        soil = SOILS[soil_name]
        fc = soil["field_capacity"]
        wp = soil["wilting_point"]

        # Derived features
        kc = crop["kc"][GROWTH_STAGE_INDEX[growth_stage]]
        efficiency = IRRIGATION_METHODS[irrig_method]
        canopy_cover, plant_density = compute_canopy_cover(
            float(data["row_spacing_cm"]),
            float(data["plant_spacing_cm"]),
            growth_stage,
        )

        # Build feature dict
        features = {
            "crop_type": crop_name,
            "soil_type": soil_name,
            "growth_stage": growth_stage,
            "irrigation_method": irrig_method,
            "temperature": float(data["temperature"]),
            "humidity": float(data["humidity"]),
            "rainfall": float(data["rainfall"]),
            "wind_speed": float(data["wind_speed"]),
            "solar_radiation": float(data["solar_radiation"]),
            "soil_moisture": float(data["soil_moisture"]),
            "row_spacing_cm": float(data["row_spacing_cm"]),
            "plant_spacing_cm": float(data["plant_spacing_cm"]),
            "plant_density": plant_density,
            "canopy_cover": canopy_cover,
            "kc": kc,
            "irrigation_efficiency": efficiency,
        }

        # Encode categoricals
        for col in CATEGORICAL_FEATURES:
            features[col] = int(label_encoders[col].transform([features[col]])[0])

        # Build feature array in correct order
        X_input = np.array([[features[col] for col in FEATURE_COLUMNS]])

        # Predict
        irrigation_mm = float(model.predict(X_input)[0])
        irrigation_mm = max(0.0, round(irrigation_mm, 2))

        # Volume conversions (field_area_ha only used here, NOT as model feature)
        volume_liters = round(irrigation_mm * field_area_ha * 10000, 0)
        volume_m3 = round(volume_liters / 1000, 2)

        # MAD alert
        mad_threshold = compute_mad_threshold(fc, wp)
        soil_moisture = float(data["soil_moisture"])
        is_below_mad = soil_moisture < mad_threshold

        # Response
        response = {
            "prediction": {
                "irrigation_need_mm_per_day": irrigation_mm,
                "volume_liters": volume_liters,
                "volume_m3": volume_m3,
                "field_area_ha": field_area_ha,
            },
            "crop_info": {
                "crop_type": crop_name,
                "kc": kc,
                "growth_stage": growth_stage,
                "plant_density_per_ha": round(plant_density, 0),
                "canopy_cover": round(canopy_cover, 3),
                "row_spacing_cm": float(data["row_spacing_cm"]),
                "plant_spacing_cm": float(data["plant_spacing_cm"]),
            },
            "soil_info": {
                "soil_type": soil_name,
                "soil_moisture": soil_moisture,
                "field_capacity": fc,
                "wilting_point": wp,
            },
            "irrigation_info": {
                "method": irrig_method,
                "efficiency": efficiency,
            },
            "mad_alert": {
                "is_below_mad": is_below_mad,
                "mad_threshold": round(mad_threshold, 1),
                "soil_moisture": soil_moisture,
                "message": (
                    f"URGENT: Soil moisture ({soil_moisture}%) is below "
                    f"MAD threshold ({mad_threshold:.1f}%). Irrigate immediately!"
                    if is_below_mad else
                    f"Soil moisture ({soil_moisture}%) is above "
                    f"MAD threshold ({mad_threshold:.1f}%). No urgent irrigation needed."
                ),
                "severity": "critical" if is_below_mad else "ok",
            },
            "model_info": {
                "version": metadata["version"],
                "model_name": metadata["model_name"],
            },
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "version": metadata["version"],
        "model": metadata["model_name"],
    }), 200


@app.route("/info", methods=["GET"])
def info():
    """Return model info, supported crops, soils, methods, and spacing ranges."""
    return jsonify({
        "version": metadata["version"],
        "model_name": metadata["model_name"],
        "metrics": metadata["metrics"],
        "supported_crops": list(CROPS.keys()),
        "supported_soils": list(SOILS.keys()),
        "supported_irrigation_methods": list(IRRIGATION_METHODS.keys()),
        "supported_growth_stages": GROWTH_STAGES,
        "crop_kc_values": {
            name: {stage: kc for stage, kc in zip(GROWTH_STAGES, crop["kc"])}
            for name, crop in CROPS.items()
        },
        "crop_spacing_ranges": {
            name: {
                "row_spacing_cm": crop["row_spacing_range"],
                "plant_spacing_cm": crop["plant_spacing_range"],
            }
            for name, crop in CROPS.items()
        },
        "soil_properties": SOILS,
        "irrigation_efficiencies": IRRIGATION_METHODS,
        "mad_fraction": MAD_FRACTION,
        "features_required": [
            "crop_type", "soil_type", "growth_stage", "irrigation_method",
            "temperature", "humidity", "rainfall", "wind_speed", "solar_radiation",
            "soil_moisture", "row_spacing_cm", "plant_spacing_cm",
        ],
        "optional_fields": {
            "field_area_ha": "Field area in hectares (default: 1.0). Used only for volume conversion, NOT as model feature.",
        },
    }), 200


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

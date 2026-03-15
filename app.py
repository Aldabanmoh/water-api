# ═══════════════════════════════════════════════════════════════
# Agricultural Water Needs Prediction API
# Flask backend that loads the trained ML model and serves predictions
# ═══════════════════════════════════════════════════════════════

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import os

app = Flask(__name__)

# ── CORS Configuration ──
# In development: allow all origins
# In production: replace with your Lovable URL
ALLOWED_ORIGINS = os.environ.get('ALLOWED_ORIGINS', '*')
if ALLOWED_ORIGINS == '*':
    CORS(app)
else:
    CORS(app, origins=ALLOWED_ORIGINS.split(','))

# ── Load Model & Encoders at Startup ──
MODEL_DIR = os.environ.get('MODEL_DIR', '.')

print("Loading model and encoders...")
model = joblib.load(os.path.join(MODEL_DIR, 'water_model.pkl'))
le_crop = joblib.load(os.path.join(MODEL_DIR, 'encoder_crop.pkl'))
le_soil = joblib.load(os.path.join(MODEL_DIR, 'encoder_soil.pkl'))
le_stage = joblib.load(os.path.join(MODEL_DIR, 'encoder_stage.pkl'))
le_irrig = joblib.load(os.path.join(MODEL_DIR, 'encoder_irrig.pkl'))
feature_cols = joblib.load(os.path.join(MODEL_DIR, 'feature_cols.pkl'))
metadata = joblib.load(os.path.join(MODEL_DIR, 'model_metadata.pkl'))
print(f"✅ Model loaded: {metadata['model_type']} (R²={metadata['test_r2']:.4f})")


@app.route('/', methods=['GET'])
def home():
    """Landing page with API info."""
    return jsonify({
        'name': 'Agricultural Water Needs Prediction API',
        'version': '1.0',
        'status': 'running',
        'endpoints': {
            '/predict': 'POST — predict water needs',
            '/health': 'GET — check API health',
            '/info': 'GET — model info and valid input values',
        }
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'model_type': metadata['model_type'],
        'test_r2': metadata['test_r2'],
        'test_mae': metadata['test_mae'],
    })


@app.route('/info', methods=['GET'])
def info():
    """Return valid input values so the frontend can populate dropdowns."""
    return jsonify({
        'crops': list(le_crop.classes_),
        'soil_types': list(le_soil.classes_),
        'growth_stages': list(le_stage.classes_),
        'irrigation_methods': list(le_irrig.classes_),
        'input_ranges': {
            'temp_max': {'min': -5, 'max': 50, 'unit': '°C'},
            'temp_min': {'min': -10, 'max': 40, 'unit': '°C'},
            'humidity': {'min': 5, 'max': 100, 'unit': '%'},
            'wind_speed': {'min': 0, 'max': 15, 'unit': 'm/s'},
            'solar_radiation': {'min': 2, 'max': 35, 'unit': 'MJ/m²/day'},
            'rainfall': {'min': 0, 'max': 100, 'unit': 'mm/day'},
            'soil_moisture': {'min': 0, 'max': 50, 'unit': '%'},
            'field_area_ha': {'min': 0.1, 'max': 500, 'unit': 'hectares'},
        },
        'model_accuracy': {
            'MAE': f"{metadata['test_mae']:.3f} mm/day",
            'R2': f"{metadata['test_r2']:.4f}",
            'RMSE': f"{metadata['test_rmse']:.3f} mm/day",
        }
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint.
    
    Expected JSON body:
    {
        "crop_type": "wheat",
        "growth_stage": "mid_season",
        "temp_max": 38.0,
        "temp_min": 22.0,
        "humidity": 35.0,
        "wind_speed": 3.2,
        "solar_radiation": 25.0,
        "rainfall": 0.0,
        "soil_type": "clay_loam",
        "soil_moisture": 28.0,
        "field_area_ha": 2.0,
        "irrigation_method": "drip"
    }
    """
    try:
        data = request.get_json()

        # ── Validate required fields ──
        required_fields = [
            'crop_type', 'growth_stage', 'temp_max', 'temp_min',
            'humidity', 'wind_speed', 'solar_radiation', 'rainfall',
            'soil_type', 'soil_moisture', 'field_area_ha', 'irrigation_method'
        ]
        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({'error': f'Missing fields: {missing}'}), 400

        # ── Validate categorical values ──
        if data['crop_type'] not in le_crop.classes_:
            return jsonify({'error': f"Invalid crop_type: '{data['crop_type']}'. Valid: {list(le_crop.classes_)}"}), 400
        if data['soil_type'] not in le_soil.classes_:
            return jsonify({'error': f"Invalid soil_type: '{data['soil_type']}'. Valid: {list(le_soil.classes_)}"}), 400
        if data['growth_stage'] not in le_stage.classes_:
            return jsonify({'error': f"Invalid growth_stage: '{data['growth_stage']}'. Valid: {list(le_stage.classes_)}"}), 400
        if data['irrigation_method'] not in le_irrig.classes_:
            return jsonify({'error': f"Invalid irrigation_method: '{data['irrigation_method']}'. Valid: {list(le_irrig.classes_)}"}), 400

        # ── Validate numeric ranges ──
        if not (-10 <= float(data['temp_min']) <= float(data['temp_max']) <= 55):
            return jsonify({'error': 'temp_min must be <= temp_max, both between -10 and 55°C'}), 400
        if not (0 <= float(data['humidity']) <= 100):
            return jsonify({'error': 'humidity must be between 0 and 100%'}), 400
        if float(data['rainfall']) < 0:
            return jsonify({'error': 'rainfall cannot be negative'}), 400
        if float(data['field_area_ha']) <= 0:
            return jsonify({'error': 'field_area_ha must be positive'}), 400

        # ── Encode categorical inputs ──
        crop_enc = le_crop.transform([data['crop_type']])[0]
        soil_enc = le_soil.transform([data['soil_type']])[0]
        stage_enc = le_stage.transform([data['growth_stage']])[0]
        irrig_enc = le_irrig.transform([data['irrigation_method']])[0]

        # ── Compute derived features ──
        temp_max = float(data['temp_max'])
        temp_min = float(data['temp_min'])
        humidity = float(data['humidity'])
        wind_speed = float(data['wind_speed'])
        solar_radiation = float(data['solar_radiation'])
        rainfall = float(data['rainfall'])
        soil_moisture = float(data['soil_moisture'])
        field_area_ha = float(data['field_area_ha'])

        temp_range = temp_max - temp_min
        temp_mean = (temp_max + temp_min) / 2
        vpd = (0.6108 * np.exp(17.27 * temp_max / (temp_max + 237.3))) * (1 - humidity / 100)

        if rainfall <= 0:
            effective_rain = 0
        elif rainfall < 20:
            effective_rain = rainfall * 0.8
        elif rainfall < 50:
            effective_rain = max(0, rainfall * 0.7 - 2)
        else:
            effective_rain = rainfall * 0.5

        aridity = temp_max / (rainfall + 1)
        heat_index = temp_max * humidity / 100
        wind_temp = wind_speed * temp_mean
        solar_humidity_ratio = solar_radiation / (humidity + 1)

        today = datetime.now()
        month = today.month
        day_of_year = today.timetuple().tm_yday
        season_sin = float(np.sin(2 * np.pi * day_of_year / 365))
        season_cos = float(np.cos(2 * np.pi * day_of_year / 365))

        # ── Build feature array in exact training order ──
        features = pd.DataFrame([{
            'temp_max': temp_max,
            'temp_min': temp_min,
            'humidity': humidity,
            'wind_speed': wind_speed,
            'solar_radiation': solar_radiation,
            'rainfall': rainfall,
            'soil_moisture': soil_moisture,
            'field_area_ha': field_area_ha,
            'crop_encoded': crop_enc,
            'soil_encoded': soil_enc,
            'stage_encoded': stage_enc,
            'irrig_encoded': irrig_enc,
            'temp_range': temp_range,
            'temp_mean': temp_mean,
            'vpd': vpd,
            'effective_rain': effective_rain,
            'aridity': aridity,
            'heat_index': heat_index,
            'wind_temp': wind_temp,
            'solar_humidity_ratio': solar_humidity_ratio,
            'month': month,
            'day_of_year': day_of_year,
            'season_sin': season_sin,
            'season_cos': season_cos,
        }])[feature_cols]

        # ── Predict ──
        water_mm = float(model.predict(features)[0])
        water_mm = max(0, water_mm)  # Can't be negative

        area_m2 = field_area_ha * 10000
        volume_liters = water_mm * area_m2
        volume_m3 = volume_liters / 1000

        # ── Response ──
        return jsonify({
            'water_need_mm': round(water_mm, 2),
            'volume_liters': round(volume_liters, 0),
            'volume_m3': round(volume_m3, 1),
            'recommendation': get_recommendation(water_mm, data['irrigation_method']),
            'input_summary': {
                'crop': data['crop_type'],
                'stage': data['growth_stage'],
                'area_ha': field_area_ha,
                'irrigation': data['irrigation_method'],
            }
        })

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


def get_recommendation(water_mm, irrigation_method):
    """Generate a human-readable irrigation recommendation."""
    if water_mm < 1:
        return "Very low water demand. Light irrigation or skip today if soil moisture is adequate."
    elif water_mm < 3:
        return "Moderate water demand. Standard irrigation cycle recommended."
    elif water_mm < 6:
        return "High water demand. Ensure full irrigation cycle. Consider early morning application to reduce evaporation loss."
    elif water_mm < 10:
        return "Very high water demand. Maximum irrigation needed. Monitor soil moisture closely and consider splitting into two applications."
    else:
        return "Extreme water demand. Critical irrigation needed. Check for heat stress and ensure irrigation system is at full capacity."


# ── Start Server ──
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)

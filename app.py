from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import gdown

app = Flask(__name__)

# ============================================================
# DOWNLOAD AND LOAD MODELS ON STARTUP
# ============================================================
model_files = {
    'crop_label_encoder.pkl': '1Wzx_fwy38GD0eRqYfGDDLV-4wSdNYOe3',
    'crop_model.pkl':         '1W_n2sgcgw-VbJ7Wq5qIeU9anpCGmRmM1',
    'fertility_model.pkl':    '1FsvHWw9P7jvnjG768EDrETJGzz2-qG6m',
    'scaler_crop.pkl':        '1gwZSfLlTLjxIeM24JUYOKpq_99svOKPF',
    'scaler_fertility.pkl':   '1-_5TEvP7OGplN_kvfocSMPHJQPfQ4vFT',
    'yield_label_encoder.pkl':'1Or1-KIIJ2FEpQiAlE68JCxoFEOJwOCQS',
    'yield_model.pkl':        '1eW9PQSY1JeNMW2pzOR6qLDJ6Z4eX5Th3',
}

os.makedirs('models', exist_ok=True)

for filename, file_id in model_files.items():
    filepath = f'models/{filename}'
    if not os.path.exists(filepath):
        print(f'Downloading {filename}...')
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, filepath, quiet=False)
    else:
        print(f'{filename} already exists, skipping.')

print('All models downloaded! Loading into memory...')

fertility_model  = joblib.load('models/fertility_model.pkl')
scaler_fertility = joblib.load('models/scaler_fertility.pkl')
crop_model       = joblib.load('models/crop_model.pkl')
scaler_crop      = joblib.load('models/scaler_crop.pkl')
le_crop          = joblib.load('models/crop_label_encoder.pkl')
yield_model      = joblib.load('models/yield_model.pkl')
le_yield         = joblib.load('models/yield_label_encoder.pkl')

print('All models loaded and ready!')

# ============================================================
# FERTILITY YIELD RANGES
# ============================================================
FERTILITY_RANGES = {
    'High':   (70, 100),
    'Medium': (45, 69),
    'Low':    (25, 44),
}

def apply_fertility_guardrail(raw_score, fertility_status):
    min_val, max_val = FERTILITY_RANGES.get(fertility_status, (25, 100))
    scaled = min_val + ((raw_score - 25) / (100 - 25)) * (max_val - min_val)
    return round(float(np.clip(scaled, min_val, max_val)), 1)

# ============================================================
# CROP PROFILES AND GENERAL HIGH FERTILITY RANGES
# ============================================================
general_high_ranges = {
    "Nitrogen":     {"min": 75.23, "max": 112.0},
    "Phosphorus":   {"min": 56.64, "max": 72.0},
    "Potassium":    {"min": 45.0,  "max": 75.98},
    "Humidity":     {"min": 64.27, "max": 81.17},
    "Temperature":  {"min": 19.44, "max": 31.05},
    "Soil_Moisture":{"min": 15.18, "max": 25.26},
}

crop_profiles = {
    "APPLE":        {"Nitrogen": {"min": 10.0, "max": 31.0}, "Phosphorus": {"min": 126.0, "max": 141.0}, "Potassium": {"min": 197.0, "max": 203.0}, "Humidity": {"min": 90.98, "max": 93.52}, "Temperature": {"min": 22.16, "max": 23.34}, "Soil_Moisture": {"min": 15.72, "max": 26.26}},
    "BANANA":       {"Nitrogen": {"min": 91.0, "max": 108.0}, "Phosphorus": {"min": 75.0, "max": 88.0}, "Potassium": {"min": 47.0, "max": 53.0}, "Humidity": {"min": 78.07, "max": 82.96}, "Temperature": {"min": 26.13, "max": 28.65}, "Soil_Moisture": {"min": 15.25, "max": 25.73}},
    "BARLEY":       {"Nitrogen": {"min": 70.0, "max": 70.0}, "Phosphorus": {"min": 60.0, "max": 60.0}, "Potassium": {"min": 50.0, "max": 50.0}, "Humidity": {"min": 69.17, "max": 80.0}, "Temperature": {"min": 17.01, "max": 30.83}, "Soil_Moisture": {"min": 45.0, "max": 65.0}},
    "BLACK GRAM":   {"Nitrogen": {"min": 56.8, "max": 57.73}, "Phosphorus": {"min": 77.55, "max": 140.83}, "Potassium": {"min": 62.19, "max": 97.18}, "Humidity": {"min": 63.11, "max": 67.89}, "Temperature": {"min": 27.83, "max": 31.87}, "Soil_Moisture": {"min": 16.4, "max": 26.46}},
    "CHICKPEA":     {"Nitrogen": {"min": 47.0, "max": 57.0}, "Phosphorus": {"min": 62.0, "max": 73.0}, "Potassium": {"min": 78.0, "max": 84.0}, "Humidity": {"min": 15.76, "max": 18.67}, "Temperature": {"min": 17.48, "max": 19.71}, "Soil_Moisture": {"min": 14.99, "max": 25.84}},
    "COCONUT":      {"Nitrogen": {"min": 73.04, "max": 80.11}, "Phosphorus": {"min": 55.06, "max": 68.88}, "Potassium": {"min": 60.64, "max": 71.81}, "Humidity": {"min": 92.97, "max": 96.98}, "Temperature": {"min": 26.26, "max": 28.83}, "Soil_Moisture": {"min": 15.88, "max": 24.65}},
    "COFFEE":       {"Nitrogen": {"min": 67.53, "max": 117.0}, "Phosphorus": {"min": 31.0, "max": 61.71}, "Potassium": {"min": 30.0, "max": 75.48}, "Humidity": {"min": 54.3, "max": 64.73}, "Temperature": {"min": 24.17, "max": 26.65}, "Soil_Moisture": {"min": 14.77, "max": 25.92}},
    "CORIANDER":    {"Nitrogen": {"min": 10.0, "max": 10.0}, "Phosphorus": {"min": 20.0, "max": 20.0}, "Potassium": {"min": 20.0, "max": 20.0}, "Humidity": {"min": 85.93, "max": 87.29}, "Temperature": {"min": 26.52, "max": 28.65}, "Soil_Moisture": {"min": 40.0, "max": 60.0}},
    "CORN":         {"Nitrogen": {"min": 78.0, "max": 84.0}, "Phosphorus": {"min": 60.5, "max": 66.0}, "Potassium": {"min": 45.0, "max": 50.0}, "Humidity": {"min": 70.06, "max": 80.0}, "Temperature": {"min": 17.29, "max": 29.94}, "Soil_Moisture": {"min": 50.0, "max": 70.0}},
    "COTTON":       {"Nitrogen": {"min": 71.5, "max": 108.0}, "Phosphorus": {"min": 56.0, "max": 66.0}, "Potassium": {"min": 23.0, "max": 60.0}, "Humidity": {"min": 73.51, "max": 80.0}, "Temperature": {"min": 20.0, "max": 26.49}, "Soil_Moisture": {"min": 15.9, "max": 25.23}},
    "GARLIC":       {"Nitrogen": {"min": 50.0, "max": 50.0}, "Phosphorus": {"min": 10.0, "max": 10.0}, "Potassium": {"min": 60.0, "max": 60.0}, "Humidity": {"min": 86.11, "max": 88.27}, "Temperature": {"min": 22.89, "max": 28.65}, "Soil_Moisture": {"min": 50.0, "max": 70.0}},
    "GRAPES":       {"Nitrogen": {"min": 12.0, "max": 35.0}, "Phosphorus": {"min": 125.0, "max": 139.0}, "Potassium": {"min": 197.0, "max": 203.0}, "Humidity": {"min": 80.86, "max": 82.9}, "Temperature": {"min": 16.28, "max": 31.11}, "Soil_Moisture": {"min": 14.56, "max": 24.47}},
    "GROUND NUT":   {"Nitrogen": {"min": 63.29, "max": 65.36}, "Phosphorus": {"min": 60.91, "max": 65.03}, "Potassium": {"min": 63.79, "max": 70.52}, "Humidity": {"min": 60.94, "max": 69.71}, "Temperature": {"min": 31.15, "max": 33.36}, "Soil_Moisture": {"min": 50.0, "max": 70.0}},
    "HORSEGRAM":    {"Nitrogen": {"min": 20.0, "max": 20.0}, "Phosphorus": {"min": 60.0, "max": 60.0}, "Potassium": {"min": 20.0, "max": 20.0}, "Humidity": {"min": 47.17, "max": 63.86}, "Temperature": {"min": 22.6, "max": 29.27}, "Soil_Moisture": {"min": 30.0, "max": 50.0}},
    "JUTE":         {"Nitrogen": {"min": 69.87, "max": 90.0}, "Phosphorus": {"min": 52.0, "max": 77.26}, "Potassium": {"min": 39.0, "max": 85.04}, "Humidity": {"min": 73.8, "max": 83.31}, "Temperature": {"min": 23.86, "max": 25.76}, "Soil_Moisture": {"min": 12.51, "max": 23.94}},
    "KIDNEY BEANS": {"Nitrogen": {"min": 28.0, "max": 51.87}, "Phosphorus": {"min": 35.76, "max": 76.0}, "Potassium": {"min": 18.0, "max": 48.78}, "Humidity": {"min": 19.19, "max": 23.4}, "Temperature": {"min": 17.48, "max": 22.84}, "Soil_Moisture": {"min": 15.43, "max": 25.98}},
    "LENTIL":       {"Nitrogen": {"min": 53.69, "max": 54.38}, "Phosphorus": {"min": 92.97, "max": 143.31}, "Potassium": {"min": 58.84, "max": 103.6}, "Humidity": {"min": 62.23, "max": 68.14}, "Temperature": {"min": 22.4, "max": 27.07}, "Soil_Moisture": {"min": 15.73, "max": 25.16}},
    "MAIZE":        {"Nitrogen": {"min": 86.54, "max": 131.02}, "Phosphorus": {"min": 37.54, "max": 75.73}, "Potassium": {"min": 39.53, "max": 82.7}, "Humidity": {"min": 48.72, "max": 82.67}, "Temperature": {"min": 19.2, "max": 35.71}, "Soil_Moisture": {"min": 15.78, "max": 25.41}},
    "MANGO":        {"Nitrogen": {"min": 57.85, "max": 58.38}, "Phosphorus": {"min": 98.84, "max": 113.82}, "Potassium": {"min": 97.35, "max": 108.79}, "Humidity": {"min": 49.03, "max": 52.42}, "Temperature": {"min": 29.4, "max": 32.19}, "Soil_Moisture": {"min": 13.55, "max": 25.5}},
    "MILLET":       {"Nitrogen": {"min": 51.3, "max": 51.52}, "Phosphorus": {"min": 81.28, "max": 102.09}, "Potassium": {"min": 133.84, "max": 141.77}, "Humidity": {"min": 11.41, "max": 13.25}, "Temperature": {"min": 43.98, "max": 47.84}, "Soil_Moisture": {"min": 35.0, "max": 55.0}},
    "MOTH BEANS":   {"Nitrogen": {"min": 34.0, "max": 53.17}, "Phosphorus": {"min": 42.89, "max": 57.0}, "Potassium": {"min": 20.0, "max": 42.96}, "Humidity": {"min": 45.89, "max": 57.7}, "Temperature": {"min": 27.09, "max": 30.38}, "Soil_Moisture": {"min": 15.94, "max": 24.85}},
    "MUNG BEAN":    {"Nitrogen": {"min": 55.73, "max": 57.35}, "Phosphorus": {"min": 97.81, "max": 145.58}, "Potassium": {"min": 55.92, "max": 95.75}, "Humidity": {"min": 83.55, "max": 87.89}, "Temperature": {"min": 27.87, "max": 29.24}, "Soil_Moisture": {"min": 16.48, "max": 25.57}},
    "MUSKMELON":    {"Nitrogen": {"min": 104.0, "max": 117.0}, "Phosphorus": {"min": 18.0, "max": 26.0}, "Potassium": {"min": 48.0, "max": 54.0}, "Humidity": {"min": 90.97, "max": 93.88}, "Temperature": {"min": 28.0, "max": 29.17}, "Soil_Moisture": {"min": 13.99, "max": 25.82}},
    "ONION":        {"Nitrogen": {"min": 120.0, "max": 120.0}, "Phosphorus": {"min": 60.0, "max": 60.0}, "Potassium": {"min": 65.0, "max": 65.0}, "Humidity": {"min": 82.51, "max": 83.11}, "Temperature": {"min": 22.89, "max": 28.68}, "Soil_Moisture": {"min": 50.0, "max": 70.0}},
    "ORANGE":       {"Nitrogen": {"min": 60.09, "max": 67.01}, "Phosphorus": {"min": 75.57, "max": 140.67}, "Potassium": {"min": 64.32, "max": 113.35}, "Humidity": {"min": 91.12, "max": 93.27}, "Temperature": {"min": 17.68, "max": 30.26}, "Soil_Moisture": {"min": 15.98, "max": 24.6}},
    "PAPAYA":       {"Nitrogen": {"min": 66.99, "max": 74.26}, "Phosphorus": {"min": 67.0, "max": 139.15}, "Potassium": {"min": 52.0, "max": 116.69}, "Humidity": {"min": 91.66, "max": 93.61}, "Temperature": {"min": 28.28, "max": 39.3}, "Soil_Moisture": {"min": 15.78, "max": 24.56}},
    "PEAS":         {"Nitrogen": {"min": 50.62, "max": 50.7}, "Phosphorus": {"min": 71.19, "max": 140.09}, "Potassium": {"min": 107.03, "max": 144.74}, "Humidity": {"min": 13.15, "max": 13.82}, "Temperature": {"min": 16.91, "max": 18.08}, "Soil_Moisture": {"min": 50.0, "max": 70.0}},
    "PIGEON PEAS":  {"Nitrogen": {"min": 59.42, "max": 63.28}, "Phosphorus": {"min": 68.12, "max": 142.53}, "Potassium": {"min": 87.94, "max": 144.66}, "Humidity": {"min": 44.82, "max": 56.31}, "Temperature": {"min": 24.19, "max": 31.09}, "Soil_Moisture": {"min": 14.96, "max": 24.42}},
    "POMEGRANATE":  {"Nitrogen": {"min": 60.67, "max": 62.88}, "Phosphorus": {"min": 89.13, "max": 144.77}, "Potassium": {"min": 70.06, "max": 143.19}, "Humidity": {"min": 88.3, "max": 92.53}, "Temperature": {"min": 19.88, "max": 23.74}, "Soil_Moisture": {"min": 16.12, "max": 26.47}},
    "POTATO":       {"Nitrogen": {"min": 77.0, "max": 138.78}, "Phosphorus": {"min": 53.07, "max": 63.92}, "Potassium": {"min": 45.0, "max": 89.85}, "Humidity": {"min": 58.01, "max": 81.12}, "Temperature": {"min": 19.52, "max": 33.29}, "Soil_Moisture": {"min": 60.0, "max": 80.0}},
    "RAGI":         {"Nitrogen": {"min": 50.0, "max": 50.0}, "Phosphorus": {"min": 40.0, "max": 40.0}, "Potassium": {"min": 20.0, "max": 20.0}, "Humidity": {"min": 81.38, "max": 85.05}, "Temperature": {"min": 25.57, "max": 29.27}, "Soil_Moisture": {"min": 35.0, "max": 55.0}},
    "RAPESEED":     {"Nitrogen": {"min": 50.0, "max": 50.0}, "Phosphorus": {"min": 40.0, "max": 40.0}, "Potassium": {"min": 20.0, "max": 20.0}, "Humidity": {"min": 56.74, "max": 81.09}, "Temperature": {"min": 20.8, "max": 23.56}, "Soil_Moisture": {"min": 45.0, "max": 65.0}},
    "RICE":         {"Nitrogen": {"min": 78.0, "max": 99.18}, "Phosphorus": {"min": 58.29, "max": 72.0}, "Potassium": {"min": 44.0, "max": 56.95}, "Humidity": {"min": 64.21, "max": 80.0}, "Temperature": {"min": 18.4, "max": 32.02}, "Soil_Moisture": {"min": 14.57, "max": 23.66}},
    "RUBBER":       {"Nitrogen": {"min": 79.64, "max": 89.04}, "Phosphorus": {"min": 42.01, "max": 46.25}, "Potassium": {"min": 63.96, "max": 71.82}, "Humidity": {"min": 68.23, "max": 75.89}, "Temperature": {"min": 23.22, "max": 31.59}, "Soil_Moisture": {"min": 60.0, "max": 80.0}},
    "SORGHUM":      {"Nitrogen": {"min": 80.0, "max": 80.0}, "Phosphorus": {"min": 40.0, "max": 40.0}, "Potassium": {"min": 40.0, "max": 40.0}, "Humidity": {"min": 81.25, "max": 85.93}, "Temperature": {"min": 25.6, "max": 30.43}, "Soil_Moisture": {"min": 35.0, "max": 55.0}},
    "SOYBEAN":      {"Nitrogen": {"min": 58.5, "max": 63.0}, "Phosphorus": {"min": 55.0, "max": 60.0}, "Potassium": {"min": 49.5, "max": 55.0}, "Humidity": {"min": 69.14, "max": 80.0}, "Temperature": {"min": 17.61, "max": 30.86}, "Soil_Moisture": {"min": 50.0, "max": 70.0}},
    "SUGARCANE":    {"Nitrogen": {"min": 78.0, "max": 106.86}, "Phosphorus": {"min": 60.0, "max": 72.0}, "Potassium": {"min": 48.0, "max": 60.71}, "Humidity": {"min": 63.27, "max": 80.0}, "Temperature": {"min": 17.92, "max": 32.38}, "Soil_Moisture": {"min": 60.0, "max": 80.0}},
    "SUNFLOWER":    {"Nitrogen": {"min": 70.0, "max": 70.0}, "Phosphorus": {"min": 66.0, "max": 66.0}, "Potassium": {"min": 55.0, "max": 55.0}, "Humidity": {"min": 68.58, "max": 80.0}, "Temperature": {"min": 17.37, "max": 31.42}, "Soil_Moisture": {"min": 45.0, "max": 65.0}},
    "SWEET POTATO": {"Nitrogen": {"min": 90.0, "max": 90.0}, "Phosphorus": {"min": 20.0, "max": 20.0}, "Potassium": {"min": 120.0, "max": 120.0}, "Humidity": {"min": 80.8, "max": 81.26}, "Temperature": {"min": 26.52, "max": 28.65}, "Soil_Moisture": {"min": 55.0, "max": 75.0}},
    "TEA":          {"Nitrogen": {"min": 68.47, "max": 71.18}, "Phosphorus": {"min": 37.85, "max": 38.71}, "Potassium": {"min": 53.46, "max": 55.13}, "Humidity": {"min": 57.85, "max": 62.86}, "Temperature": {"min": 22.98, "max": 26.39}, "Soil_Moisture": {"min": 60.0, "max": 80.0}},
    "TOBACCO":      {"Nitrogen": {"min": 56.28, "max": 57.56}, "Phosphorus": {"min": 51.75, "max": 53.42}, "Potassium": {"min": 52.02, "max": 52.94}, "Humidity": {"min": 62.88, "max": 67.76}, "Temperature": {"min": 22.18, "max": 26.15}, "Soil_Moisture": {"min": 45.0, "max": 65.0}},
    "TOMATO":       {"Nitrogen": {"min": 89.56, "max": 129.21}, "Phosphorus": {"min": 36.63, "max": 74.3}, "Potassium": {"min": 40.32, "max": 84.95}, "Humidity": {"min": 48.93, "max": 82.95}, "Temperature": {"min": 19.16, "max": 35.87}, "Soil_Moisture": {"min": 50.0, "max": 70.0}},
    "TURMERIC":     {"Nitrogen": {"min": 25.0, "max": 25.0}, "Phosphorus": {"min": 60.0, "max": 60.0}, "Potassium": {"min": 100.0, "max": 100.0}, "Humidity": {"min": 86.82, "max": 88.48}, "Temperature": {"min": 23.74, "max": 28.55}, "Soil_Moisture": {"min": 55.0, "max": 75.0}},
    "WATERMELON":   {"Nitrogen": {"min": 102.0, "max": 119.0}, "Phosphorus": {"min": 19.25, "max": 30.0}, "Potassium": {"min": 49.0, "max": 55.0}, "Humidity": {"min": 83.04, "max": 87.82}, "Temperature": {"min": 24.9, "max": 26.21}, "Soil_Moisture": {"min": 16.21, "max": 23.87}},
    "WHEAT":        {"Nitrogen": {"min": 77.0, "max": 125.08}, "Phosphorus": {"min": 49.15, "max": 70.72}, "Potassium": {"min": 45.0, "max": 77.92}, "Humidity": {"min": 53.94, "max": 80.0}, "Temperature": {"min": 18.37, "max": 34.19}, "Soil_Moisture": {"min": 45.0, "max": 65.0}},
}

# ============================================================
# IMPROVEMENT SUGGESTIONS HELPER
# Generates advice for all 6 parameters against a target range
# ============================================================
param_units = {
    "Nitrogen": "mg/kg", "Phosphorus": "mg/kg", "Potassium": "mg/kg",
    "Humidity": "%", "Temperature": "°C", "Soil_Moisture": "%"
}

param_sentences = {
    "Nitrogen": {
        "low":  "Apply nitrogen-based fertilizer such as urea or NPK blend.",
        "high": "Reduce nitrogen application and allow natural depletion before next planting.",
        "good": "No change needed."
    },
    "Phosphorus": {
        "low":  "Apply phosphorus-rich fertilizer such as single superphosphate or DAP.",
        "high": "Avoid further phosphorus application and allow levels to deplete gradually.",
        "good": "No change needed."
    },
    "Potassium": {
        "low":  "Apply potassium-rich fertilizer such as muriate of potash or potassium sulfate.",
        "high": "Reduce potassium input. Excess potassium can block magnesium and calcium uptake.",
        "good": "No change needed."
    },
    "Soil_Moisture": {
        "low":  "Increase irrigation frequency or duration.",
        "high": "Reduce irrigation and improve field drainage to prevent waterlogging.",
        "good": "No change needed."
    },
    "Temperature": {
        "low":  "Consider greenhouse covering or row covers to trap heat.",
        "high": "Provide shade nets or increase irrigation to create a cooling effect.",
        "good": "No change needed."
    },
    "Humidity": {
        "low":  "Increase irrigation or use mulching to raise ambient moisture.",
        "high": "Improve air circulation and field drainage to reduce excess humidity and lower disease risk.",
        "good": "No change needed."
    },
}

def generate_advice(sensor_values, target_ranges, goal_label):
    """
    Compares sensor values against target ranges and generates
    one advice sentence per parameter.
    sensor_values: dict with current readings
    target_ranges: dict with min/max per parameter
    goal_label: string shown at top e.g. 'High fertility'
    """
    lines = [f"To reach {goal_label}:", ""]
    param_order = ["Nitrogen", "Phosphorus", "Potassium",
                   "Humidity", "Temperature", "Soil_Moisture"]

    for param in param_order:
        if param not in target_ranges:
            continue
        val   = sensor_values.get(param, 0)
        lo    = target_ranges[param]["min"]
        hi    = target_ranges[param]["max"]
        unit  = param_units.get(param, "")
        short = param.replace("_", " ")

        if val < lo:
            status = "Low"
            action = param_sentences[param]["low"]
        elif val > hi:
            status = "High"
            action = param_sentences[param]["high"]
        else:
            status = "Good"
            action = param_sentences[param]["good"]

        lines.append(f"{short}: {status} ({val} {unit}, target: {lo}-{hi} {unit})")
        lines.append(f"→ {action}")
        lines.append("")

    return "\n".join(lines).strip()

# ============================================================
# API ROUTES
# ============================================================
@app.route('/', methods=['GET'])
def home():
    return jsonify({'status': 'Soil Quality Monitoring API is running'})

# ============================================================
# EXISTING PREDICT ENDPOINT — now also returns improvement advice
# ============================================================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        nitrogen      = float(data['nitrogen'])
        phosphorus    = float(data['phosphorus'])
        potassium     = float(data['potassium'])
        humidity      = float(data['humidity'])
        temperature   = float(data['temperature'])
        soil_moisture = float(data['soil_moisture'])

        sensor_input = [[nitrogen, phosphorus, potassium,
                         humidity, temperature, soil_moisture]]

        # Fertility prediction
        input_scaled_f = scaler_fertility.transform(sensor_input)
        fertility = str(fertility_model.predict(input_scaled_f)[0])

        # Crop recommendation
        input_scaled_c = scaler_crop.transform(sensor_input)
        crop_probs = crop_model.predict_proba(input_scaled_c)[0]
        top_index = np.argmax(crop_probs)
        recommended_crop = le_crop.classes_[top_index]
        confidence = round(float(crop_probs[top_index]) * 100, 1)

        # Yield prediction
        if recommended_crop in le_yield.classes_:
            crop_encoded_yield = le_yield.transform([recommended_crop])[0]
        else:
            crop_encoded_yield = 0

        yield_input = [[nitrogen, phosphorus, potassium,
                        humidity, temperature, float(crop_encoded_yield)]]
        raw_score = float(yield_model.predict(yield_input)[0])
        yield_score = apply_fertility_guardrail(raw_score, fertility)

        # Enhancement 1 — improvement advice using general high fertility ranges
        sensor_values = {
            "Nitrogen": nitrogen, "Phosphorus": phosphorus,
            "Potassium": potassium, "Humidity": humidity,
            "Temperature": temperature, "Soil_Moisture": soil_moisture
        }
        improvement_advice = generate_advice(
            sensor_values, general_high_ranges, "High fertility"
        )

        return jsonify({
            'status':             'success',
            'fertility_status':   fertility,
            'recommended_crop':   str(recommended_crop),
            'crop_confidence':    confidence,
            'yield_score':        yield_score,
            'improvement_advice': improvement_advice
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

# ============================================================
# NEW CROP REQUIREMENTS ENDPOINT — Enhancement 2
# ============================================================
@app.route('/crop_requirements', methods=['POST'])
def crop_requirements():
    try:
        data = request.get_json()

        desired_crop  = str(data['desired_crop']).upper().strip()
        nitrogen      = float(data['nitrogen'])
        phosphorus    = float(data['phosphorus'])
        potassium     = float(data['potassium'])
        humidity      = float(data['humidity'])
        temperature   = float(data['temperature'])
        soil_moisture = float(data['soil_moisture'])

        if desired_crop not in crop_profiles:
            available = sorted(crop_profiles.keys())
            return jsonify({
                'status': 'error',
                'message': f'Crop "{desired_crop}" not found.',
                'available_crops': available
            }), 400

        sensor_values = {
            "Nitrogen": nitrogen, "Phosphorus": phosphorus,
            "Potassium": potassium, "Humidity": humidity,
            "Temperature": temperature, "Soil_Moisture": soil_moisture
        }

        crop_advice = generate_advice(
            sensor_values,
            crop_profiles[desired_crop],
            f"{desired_crop.title()} cultivation"
        )

        return jsonify({
            'status':       'success',
            'desired_crop': desired_crop,
            'crop_advice':  crop_advice
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import gdown
import threading

app = Flask(__name__)

# ============================================================
# DOWNLOAD MODELS FROM GOOGLE DRIVE
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

# Global model variables
models = {}
models_loaded = False
models_loading = False

def download_and_load_models():
    global models, models_loaded, models_loading
    models_loading = True

    os.makedirs('models', exist_ok=True)

    for filename, file_id in model_files.items():
        filepath = f'models/{filename}'
        if not os.path.exists(filepath):
            print(f'Downloading {filename}...')
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, filepath, quiet=False)
        else:
            print(f'{filename} already exists, skipping.')

    print('All models downloaded! Loading...')

    models['fertility_model']  = joblib.load('models/fertility_model.pkl')
    models['scaler_fertility'] = joblib.load('models/scaler_fertility.pkl')
    models['crop_model']       = joblib.load('models/crop_model.pkl')
    models['scaler_crop']      = joblib.load('models/scaler_crop.pkl')
    models['le_crop']          = joblib.load('models/crop_label_encoder.pkl')
    models['yield_model']      = joblib.load('models/yield_model.pkl')
    models['le_yield']         = joblib.load('models/yield_label_encoder.pkl')

    models_loaded = True
    models_loading = False
    print('All models loaded and ready!')

# Start loading models in background thread immediately on startup
# This way gunicorn starts fast and models load in background
thread = threading.Thread(target=download_and_load_models)
thread.daemon = True
thread.start()

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
# API ROUTES
# ============================================================
@app.route('/', methods=['GET'])
def home():
    if models_loaded:
        status = 'Soil Quality Monitoring API is running'
    elif models_loading:
        status = 'API is starting up - models loading, please wait...'
    else:
        status = 'API is initializing...'
    return jsonify({'status': status, 'models_ready': models_loaded})

@app.route('/predict', methods=['POST'])
def predict():
    if not models_loaded:
        return jsonify({
            'status': 'error',
            'message': 'Models still loading, please wait 2-3 minutes and try again.'
        }), 503

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
        input_scaled_f = models['scaler_fertility'].transform(sensor_input)
        fertility = str(models['fertility_model'].predict(input_scaled_f)[0])

        # Crop recommendation
        input_scaled_c = models['scaler_crop'].transform(sensor_input)
        crop_probs = models['crop_model'].predict_proba(input_scaled_c)[0]
        top_index = np.argmax(crop_probs)
        recommended_crop = models['le_crop'].classes_[top_index]
        confidence = round(float(crop_probs[top_index]) * 100, 1)

        # Yield prediction
        if recommended_crop in models['le_yield'].classes_:
            crop_encoded_yield = models['le_yield'].transform([recommended_crop])[0]
        else:
            crop_encoded_yield = 0

        yield_input = [[nitrogen, phosphorus, potassium,
                        humidity, temperature, float(crop_encoded_yield)]]
        raw_score = float(models['yield_model'].predict(yield_input)[0])
        yield_score = apply_fertility_guardrail(raw_score, fertility)

        return jsonify({
            'status':           'success',
            'fertility_status': fertility,
            'recommended_crop': str(recommended_crop),
            'crop_confidence':  confidence,
            'yield_score':      yield_score
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

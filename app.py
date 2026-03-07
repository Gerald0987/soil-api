from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import gdown

app = Flask(__name__)

# ============================================================
# DOWNLOAD MODELS FROM GOOGLE DRIVE ON STARTUP
# Since models are too large for GitHub, we store them on
# Google Drive and download them when the API starts.
# gdown is a library that downloads files from Google Drive.
# ============================================================

# Map each model filename to its Google Drive file ID
model_files = {
    'crop_label_encoder.pkl': '1Wzx_fwy38GD0eRqYfGDDLV-4wSdNYOe3',
    'crop_model.pkl':         '1W_n2sgcgw-VbJ7Wq5qIeU9anpCGmRmM1',
    'fertility_model.pkl':    '1FsvHWw9P7jvnjG768EDrETJGzz2-qG6m',
    'scaler_crop.pkl':        '1gwZSfLlTLjxIeM24JUYOKpq_99svOKPF',
    'scaler_fertility.pkl':   '1-_5TEvP7OGplN_kvfocSMPHJQPfQ4vFT',
    'scaler_yield.pkl':       '1yBEsg8YNKegTXJmNFN2FLcNvoxeFwAAg',
    'yield_label_encoder.pkl':'1Or1-KIIJ2FEpQiAlE68JCxoFEOJwOCQS',
    'yield_model.pkl':        '1eW9PQSY1JeNMW2pzOR6qLDJ6Z4eX5Th3',
}

# Create models folder if it doesn't exist
os.makedirs('models', exist_ok=True)

# Download each model file if it doesn't already exist
for filename, file_id in model_files.items():
    filepath = f'models/{filename}'
    if not os.path.exists(filepath):
        print(f'Downloading {filename}...')
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, filepath, quiet=False)
    else:
        print(f'{filename} already exists, skipping download.')

print('All models ready!')

# ============================================================
# LOAD ALL MODELS INTO MEMORY
# ============================================================
fertility_model  = joblib.load('models/fertility_model.pkl')
scaler_fertility = joblib.load('models/scaler_fertility.pkl')
crop_model       = joblib.load('models/crop_model.pkl')
scaler_crop      = joblib.load('models/scaler_crop.pkl')
le_crop          = joblib.load('models/crop_label_encoder.pkl')
yield_model      = joblib.load('models/yield_model.pkl')
scaler_yield     = joblib.load('models/scaler_yield.pkl')
le_yield         = joblib.load('models/yield_label_encoder.pkl')

print('All models loaded!')

# ============================================================
# API ROUTES
# ============================================================
@app.route('/', methods=['GET'])
def home():
    return jsonify({'status': 'Soil Quality Monitoring API is running'})

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
        fertility = fertility_model.predict(input_scaled_f)[0]

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
        yield_scaled = scaler_yield.transform(yield_input)
        yield_score = round(float(yield_model.predict(yield_scaled)[0]), 1)

        return jsonify({
            'status': 'success',
            'fertility_status': str(fertility),
            'recommended_crop': str(recommended_crop),
            'crop_confidence': confidence,
            'yield_score': yield_score
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

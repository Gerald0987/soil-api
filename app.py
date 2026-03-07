from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

fertility_model  = joblib.load('models/fertility_model.pkl')
scaler_fertility = joblib.load('models/scaler_fertility.pkl')
crop_model       = joblib.load('models/crop_model.pkl')
scaler_crop      = joblib.load('models/scaler_crop.pkl')
le_crop          = joblib.load('models/crop_label_encoder.pkl')
yield_model      = joblib.load('models/yield_model.pkl')
scaler_yield     = joblib.load('models/scaler_yield.pkl')
le_yield         = joblib.load('models/yield_label_encoder.pkl')

@app.route('/', methods=['GET'])
def home():
    return jsonify({'status': 'Soil API is running'})

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

        input_scaled_f = scaler_fertility.transform(sensor_input)
        fertility = fertility_model.predict(input_scaled_f)[0]

        input_scaled_c = scaler_crop.transform(sensor_input)
        crop_probs = crop_model.predict_proba(input_scaled_c)[0]
        top_index = np.argmax(crop_probs)
        recommended_crop = le_crop.classes_[top_index]
        confidence = round(float(crop_probs[top_index]) * 100, 1)

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

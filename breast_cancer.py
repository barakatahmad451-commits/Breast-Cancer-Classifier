from datetime import datetime
from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__, static_folder='static', template_folder='templates')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'top_features_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'top_features_scaler.pkl')
FEATURES_PATH = os.path.join(BASE_DIR, 'top_features_columns.pkl')

model = pickle.load(open(MODEL_PATH, 'rb'))
scaler = pickle.load(open(SCALER_PATH, 'rb'))
features = pickle.load(open(FEATURES_PATH, 'rb'))

CLASS_LABELS = {
    0: 'Malignancy risk predicted',
    1: 'Benign likely'
}

FEATURE_INFO = {
    'concave_points_worst': {
        'label': 'Maximum Concave Points',
        'description': 'Proportion of concave points on the tumor boundary. Higher values indicate more irregularity.',
        'range': '0.0 - 0.30',
        'unit': 'ratio'
    },
    'perimeter_worst': {
        'label': 'Largest Tumor Perimeter',
        'description': 'The longest tumor boundary measurement from the worst sample. Large perimeters are associated with advanced growth.',
        'range': '50 - 260',
        'unit': 'mm'
    },
    'concave_points_mean': {
        'label': 'Average Concave Points',
        'description': 'Mean proportion of concave points across tumor samples. Higher averages signal surface irregularity.',
        'range': '0.0 - 0.20',
        'unit': 'ratio'
    },
    'radius_worst': {
        'label': 'Maximum Tumor Radius',
        'description': 'The largest tumor radius measured. Larger radii are a strong indicator of malignancy.',
        'range': '5 - 30',
        'unit': 'mm'
    },
    'perimeter_mean': {
        'label': 'Average Tumor Perimeter',
        'description': 'Mean tumor boundary length. Values above normal indicate irregular or large tumors.',
        'range': '40 - 180',
        'unit': 'mm'
    },
    'area_worst': {
        'label': 'Largest Tumor Area',
        'description': 'The largest observed tumor area. Larger areas are commonly linked to higher malignancy risk.',
        'range': '200 - 3000',
        'unit': 'mm²'
    }
}

@app.route('/')
def home():
    return render_template(
        'index.html',
        features=features,
        feature_info=FEATURE_INFO,
        input_values={},
        prediction=None,
        confidence=None,
        status=None,
        timestamp=None,
        risk_score=None,
        risk_details=None,
    )

@app.route('/predict', methods=['POST'])
def predict():
    input_values = {}
    values = []

    for feature in features:
        raw = request.form.get(feature, '')
        try:
            value = float(raw) if raw != '' else 0.0
        except (ValueError, TypeError):
            value = 0.0
        values.append(value)
        input_values[feature] = value

    final_input = np.array(values).reshape(1, -1)
    final_input_scaled = scaler.transform(final_input)

    prediction = int(model.predict(final_input_scaled)[0])
    result_label = CLASS_LABELS.get(prediction, f'Class {prediction}')

    probability = 0.0
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(final_input_scaled)[0]
        probability = float(probabilities[prediction]) * 100
    elif hasattr(model, 'decision_function'):
        margin = model.decision_function(final_input_scaled)
        probability = float(1 / (1 + np.exp(-margin))[0]) * 100

    confidence = round(max(0.0, min(100.0, probability)), 1)
    status = 'High Malignancy Risk' if prediction == 0 else 'Low Malignancy Risk'
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    risk_score = calculate_risk_score(input_values)
    risk_details = summarize_risk_drivers(input_values)

    return render_template(
        'index.html',
        features=features,
        feature_info=FEATURE_INFO,
        prediction=result_label,
        status=status,
        confidence=confidence,
        risk_score=risk_score,
        risk_details=risk_details,
        input_values=input_values,
        timestamp=timestamp,
    )


def calculate_risk_score(input_dict):
    score = 50
    # concave point severity
    cpw = input_dict.get('concave_points_worst', 0)
    if cpw > 0.18:
        score += 18
    elif cpw > 0.12:
        score += 12
    elif cpw > 0.08:
        score += 6

    pcm = input_dict.get('concave_points_mean', 0)
    if pcm > 0.12:
        score += 10
    elif pcm > 0.08:
        score += 5

    prw = input_dict.get('perimeter_worst', 0)
    if prw > 170:
        score += 12
    elif prw > 140:
        score += 7

    prm = input_dict.get('perimeter_mean', 0)
    if prm > 140:
        score += 8
    elif prm > 120:
        score += 4

    arw = input_dict.get('area_worst', 0)
    if arw > 1500:
        score += 10
    elif arw > 1000:
        score += 5

    rrw = input_dict.get('radius_worst', 0)
    if rrw > 18:
        score += 10
    elif rrw > 14:
        score += 5

    return max(0, min(100, score))


def summarize_risk_drivers(input_dict):
    details = []

    cpw = input_dict.get('concave_points_worst', 0)
    if cpw > 0.18:
        details.append({'name': 'Maximum Concave Points', 'details': f'High value ({cpw}) indicates severe tumor boundary irregularity.'})
    elif cpw > 0.12:
        details.append({'name': 'Maximum Concave Points', 'details': f'Elevated value ({cpw}) suggests irregular tumor surface.'})
    else:
        details.append({'name': 'Maximum Concave Points', 'details': f'Normal range ({cpw}) for concave points.'})

    prw = input_dict.get('perimeter_worst', 0)
    if prw > 170:
        details.append({'name': 'Largest Tumor Perimeter', 'details': f'High perimeter ({prw} mm) often associates with larger malignant tumors.'})
    elif prw > 140:
        details.append({'name': 'Largest Tumor Perimeter', 'details': f'Moderately elevated perimeter ({prw} mm) may warrant careful review.'})
    else:
        details.append({'name': 'Largest Tumor Perimeter', 'details': f'Perimeter ({prw} mm) is within a more typical range.'})

    rrw = input_dict.get('radius_worst', 0)
    if rrw > 18:
        details.append({'name': 'Maximum Tumor Radius', 'details': f'Radius ({rrw} mm) is high and can indicate a growing lesion.'})
    elif rrw > 14:
        details.append({'name': 'Maximum Tumor Radius', 'details': f'Moderately sized radius ({rrw} mm) should be reviewed clinically.'})
    else:
        details.append({'name': 'Maximum Tumor Radius', 'details': f'Radius ({rrw} mm) is within a lower-risk range.'})

    arw = input_dict.get('area_worst', 0)
    if arw > 1500:
        details.append({'name': 'Largest Tumor Area', 'details': f'Area ({arw} mm²) is large, consistent with high-risk lesions.'})
    elif arw > 1000:
        details.append({'name': 'Largest Tumor Area', 'details': f'Area ({arw} mm²) is elevated and requires attention.'})
    else:
        details.append({'name': 'Largest Tumor Area', 'details': f'Area ({arw} mm²) is in a lower-risk zone.'})

    return details

if __name__ == '__main__':
    app.run(debug=True)

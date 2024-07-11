from flask import Flask, request, render_template, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Load the GMM model and scaler
model = joblib.load('new_gmm_model.joblib')
scaler = joblib.load('scaler.joblib')

@app.route('/', methods=['GET'])
def index():
    # Define the fields and their labels for the form
    fields = {
        'anchor_age': 'Age at Admission',
        'gender': 'Gender',
        'sofa_score': 'SOFA Score',
        'avg_heart_rate': 'Average Heart Rate',
        'avg_sbp': 'Average Systolic BP',
        'avg_dbp': 'Average Diastolic BP',
        'avg_resp_rate': 'Average Respiratory Rate',
        'avg_temperature': 'Average Temperature',
        'avg_spo2': 'Average SpO2',
        'avg_aniongap': 'Average Anion Gap',
        'avg_bicarbonate': 'Average Bicarbonate',
        'avg_bun': 'Average BUN',
        'avg_calcium': 'Average Calcium',
        'avg_chloride': 'Average Chloride',
        'avg_creatinine': 'Average Creatinine',
        'avg_glucose': 'Average Glucose',
        'avg_sodium': 'Average Sodium',
        'avg_potassium': 'Average Potassium',
        'avg_hematocrit': 'Average Hematocrit',
        'avg_platelet': 'Average Platelet Count',
        'avg_wbc': 'Average WBC Count',
        'avg_pco2': 'Average pCO2',
        'avg_ph': 'Average pH',
        'avg_po2': 'Average pO2',
        'avg_inr': 'Average INR',
        'avg_ptt': 'Average PTT',
        'avg_Alanine_transaminase': 'Average Alanine Transaminase',
        'avg_Alkaline_phosphatase': 'Average Alkaline Phosphatase',
        'avg_Total_bilirubin': 'Average Total Bilirubin',
        'hypertension': 'Hypertension',
        'Diabetes_Mellitus': 'Diabetes Mellitus',
        'Congestive_Heart_Failure': 'Congestive Heart Failure',
        'Acute_Myocardial_Infarction': 'Acute Myocardial Infarction',
        'COPD': 'COPD',
        'Chronic_Kidney_Disease': 'Chronic Kidney Disease',
        'Old_Myocardial_Infarction': 'Old Myocardial Infarction',
        'ventilation_status': 'Ventilation Status',
        'use_of_vasopressor_medications': 'Use of Vasopressor Medications',
        'dialysis_active': 'Active Dialysis'
    }
    return render_template('index.html', fields=fields)

@app.route('/about.html')
def about():
    return render_template('about.html')

@app.route('/Contact.html')
def Contact():
    return render_template('Contact.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect data and convert as needed
        data = {key: float(request.form[key]) for key in request.form if key not in [
            'gender', 'hypertension', 'Diabetes_Mellitus', 'Congestive_Heart_Failure',
            'Acute_Myocardial_Infarction', 'COPD', 'Chronic_Kidney_Disease',
            'Old_Myocardial_Infarction', 'ventilation_status', 'use_of_vasopressor_medications', 'dialysis_active'
        ]}
        binary_fields = {
            'gender': {'Male': 0, 'Female': 1},
            'hypertension': {'No': 0, 'Yes': 1},
            'Diabetes_Mellitus': {'No': 0, 'Yes': 1},
            'Congestive_Heart_Failure': {'No': 0, 'Yes': 1},
            'Acute_Myocardial_Infarction': {'No': 0, 'Yes': 1},
            'COPD': {'No': 0, 'Yes': 1},
            'Chronic_Kidney_Disease': {'No': 0, 'Yes': 1},
            'Old_Myocardial_Infarction': {'No': 0, 'Yes': 1},
            'ventilation_status': {'No': 0, 'Yes': 1},
            'use_of_vasopressor_medications': {'No': 0, 'Yes': 1},
            'dialysis_active': {'No': 0, 'Yes': 1}
        }
        for field, mapping in binary_fields.items():
            data[field] = mapping[request.form[field]]

        # Arrange data into a list in the correct order for the model
        feature_order = [
            'anchor_age', 'gender', 'sofa_score', 'avg_heart_rate', 'avg_sbp',
            'avg_dbp', 'avg_resp_rate', 'avg_temperature', 'avg_spo2', 'avg_aniongap',
            'avg_bicarbonate', 'avg_bun', 'avg_calcium', 'avg_chloride', 'avg_creatinine',
            'avg_glucose', 'avg_sodium', 'avg_potassium', 'avg_hematocrit', 'avg_platelet',
            'avg_wbc', 'avg_pco2', 'avg_ph', 'avg_po2', 'avg_inr', 'avg_ptt',
            'avg_Alanine_transaminase', 'avg_Alkaline_phosphatase', 'avg_Total_bilirubin',
            'hypertension', 'Diabetes_Mellitus', 'Congestive_Heart_Failure', 'Acute_Myocardial_Infarction',
            'COPD', 'Chronic_Kidney_Disease', 'Old_Myocardial_Infarction', 'ventilation_status',
            'use_of_vasopressor_medications', 'dialysis_active'
        ]
        model_input = [data[feature] for feature in feature_order]
        scaled_data = scaler.transform([model_input])
        cluster = model.predict(scaled_data)

        # Return a rendered template with the results
        return render_template('predict.html', cluster=int(cluster[0]))
    
    except Exception as e:
        # Handle errors gracefully
        return render_template('error.html', message=str(e))

if __name__ == '__main__':
    app.run(debug=True)


import pandas as pd
import joblib

# Paths for the model and scaler
model_path = "/Users/gokulb/Downloads/app1/artifacts1\model.joblib"
scaler_path = "/Users/gokulb/Downloads/app1/artifacts1\scaler.joblib"

# Load the model and scaler
model = joblib.load(model_path)
scaled = joblib.load(scaler_path)


# Function to calculate normalized risk score
def calculate_normalized_risk(medical_history):
    risk_scores = {
        "diabetes": 6,
        "heart disease": 8,
        "high blood pressure": 6,
        "thyroid": 5,
        "no disease": 0
    }
    diseases = medical_history.lower().split(" & ")
    total_risk_score = sum(risk_scores.get(disease, 0) for disease in diseases)
    max_score, min_score = 14, 0  # Max and Min risk scores
    normalized_risk_score = (total_risk_score - min_score) / (max_score - min_score)
    return normalized_risk_score


# Function to calculate lifestyle risk score
def calculate_lifestyle_risk(physical_activity, stress_level):
    activity_mapping = {'High': 0, 'Medium': 1, 'Low': 4}
    stress_mapping = {'High': 4, 'Medium': 1, 'Low': 0}
    physical_activity_score = activity_mapping.get(physical_activity, 1)
    stress_level_score = stress_mapping.get(stress_level, 1)
    total_lifestyle_risk = physical_activity_score + stress_level_score
    max_lifestyle_risk, min_lifestyle_risk = 8, 0
    life_style_risk_score = (total_lifestyle_risk - min_lifestyle_risk) / (max_lifestyle_risk - min_lifestyle_risk)
    return life_style_risk_score


# Function to preprocess input data
def preprocess_input(input_dict):
    expected_columns = [
        'age', 'number_of_dependants', 'income_lakhs', 'insurance_plan', 'normalized_risk_score',
        'life_style_risk_score', 'gender_Male', 'region_Northwest', 'region_Southeast', 'region_Southwest',
        'marital_status_Unmarried', 'bmi_category_Obesity', 'bmi_category_Overweight',
        'bmi_category_Underweight', 'smoking_status_Occasional', 'smoking_status_Regular',
        'employment_status_Salaried', 'employment_status_Self-Employed'
    ]

    insurance_plan_encoding = {'Bronze': 1, 'Silver': 2, 'Gold': 3}
    df = pd.DataFrame(0, columns=expected_columns, index=[0])

    # Map inputs to the DataFrame
    for key, value in input_dict.items():
        if key == 'Gender' and value == 'Male':
            df['gender_Male'] = 1
        elif key == 'Region':
            if value == 'Northwest':
                df['region_Northwest'] = 1
            elif value == 'Southeast':
                df['region_Southeast'] = 1
            elif value == 'Southwest':
                df['region_Southwest'] = 1
        elif key == 'Marital Status' and value == 'Unmarried':
            df['marital_status_Unmarried'] = 1
        elif key == 'BMI Category':
            if value == 'Obesity':
                df['bmi_category_Obesity'] = 1
            elif value == 'Overweight':
                df['bmi_category_Overweight'] = 1
            elif value == 'Underweight':
                df['bmi_category_Underweight'] = 1
        elif key == 'Smoking Status':
            if value == 'Occasional':
                df['smoking_status_Occasional'] = 1
            elif value == 'Regular':
                df['smoking_status_Regular'] = 1
        elif key == 'Employment Status':
            if value == 'Salaried':
                df['employment_status_Salaried'] = 1
            elif value == 'Self-Employed':
                df['employment_status_Self-Employed'] = 1
        elif key == 'Insurance Plan':
            df['insurance_plan'] = insurance_plan_encoding.get(value, 1)
        elif key == 'Age':
            df['age'] = value
        elif key == 'Number of Dependants':
            df['number_of_dependants'] = value
        elif key == 'Income in Lakhs':
            df['income_lakhs'] = value

    # Add normalized risk score
    df['normalized_risk_score'] = calculate_normalized_risk(input_dict.get('Medical History', 'no disease'))
    # Add lifestyle risk score
    df['life_style_risk_score'] = calculate_lifestyle_risk(
        input_dict.get('Physical Activity', 'Medium'),
        input_dict.get('Stress Level', 'Medium')
    )

    # Scale the relevant columns
    df = handle_scaling(df)

    return df


# Function to handle scaling
def handle_scaling(df):
    scaler_object = scaled
    cols_to_scale = scaler_object['cols_to_scale']
    scaler = scaler_object['scaler']

    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    return df


# Function to make predictions
def predict(input_dict):
    input_df = preprocess_input(input_dict)
    prediction = model.predict(input_df)
    return int(prediction[0])
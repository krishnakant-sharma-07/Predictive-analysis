import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import joblib
from flask import Flask, request, jsonify, render_template
import logging

app = Flask(__name__)

data = None
model = None
scaler = None

# Set up logging
logging.basicConfig(level=logging.INFO)

# Ensure the 'uploads' directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Load model and scaler function
def load_model():
    global model, scaler
    if os.path.exists('model.pkl') and os.path.exists('scaler.pkl'):
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        logging.info("Model and scaler loaded successfully.")
    else:
        logging.error("Model or scaler not found. Please train the model first.")
        model = None
        scaler = None

load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global data
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith('.csv'):
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        data = pd.read_csv(file_path)

        if 'Downtime_Flag' not in data.columns:
            return jsonify({"error": "Target column 'Downtime_Flag' not found in the dataset."}), 400
        
        logging.info("Dataset uploaded successfully.")
        return jsonify({"message": "Dataset uploaded successfully!"}), 200
    else:
        return jsonify({"error": "Invalid file format. Please upload a CSV file."}), 400

def preprocess_data(data):
    # Separate numeric and categorical columns
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    categorical_columns = data.select_dtypes(include=['object']).columns

    # Impute missing values for numeric columns
    if not numeric_columns.empty:
        imputer_numeric = SimpleImputer(strategy='mean')
        data[numeric_columns] = imputer_numeric.fit_transform(data[numeric_columns])

    # Impute missing values for categorical columns
    if not categorical_columns.empty:
        imputer_categorical = SimpleImputer(strategy='most_frequent')
        data[categorical_columns] = imputer_categorical.fit_transform(data[categorical_columns])

    # Encode categorical columns using LabelEncoder
    for col in categorical_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

    # Ensure all columns are numeric
    data = data.apply(pd.to_numeric, errors='coerce')

    # Fill any remaining NaN values
    data.fillna(data.mean(), inplace=True)
    
    return data


@app.route('/train', methods=['POST'])
def train_model():
    global model, scaler
    if data is None:
        return jsonify({"error": "No dataset available. Please upload a dataset first."}), 400

    try:
        processed_data = preprocess_data(data)
        
        if processed_data is None:
            return jsonify({"error": "Error in data preprocessing."}), 400

        # Ensure 'Downtime_Flag' is the target variable and exists
        if 'Downtime_Flag' not in processed_data.columns:
            return jsonify({"error": "'Downtime_Flag' not found in the processed data."}), 400
        
        X = processed_data.drop('Downtime_Flag', axis=1)
        y = processed_data['Downtime_Flag']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)

        # Save feature names
        feature_names = list(X.columns)
        joblib.dump(feature_names, 'feature_names.pkl')

        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        joblib.dump(model, 'model.pkl')
        joblib.dump(scaler, 'scaler.pkl')

        logging.info(f"Model trained successfully with accuracy: {accuracy:.2f}")
        return jsonify({"message": f"Model trained successfully with accuracy: {accuracy:.2f}"}), 200

    except Exception as e:
        logging.error(f"Error during model training: {str(e)}")
        return jsonify({"error": f"Error during model training: {str(e)}"}), 500

@app.route('/predict', methods=['POST'])
def predict():
    global model, scaler
    if model is None or scaler is None:
        return jsonify({"error": "Model is not trained yet. Please train the model first."}), 400

    try:
        feature_names = joblib.load('feature_names.pkl')
        input_data_str = request.form.get('input_data')
        if not input_data_str:
            return jsonify({"error": "No input data provided."}), 400

        input_data = eval(input_data_str)
        input_df = pd.DataFrame([input_data])
        for feature in feature_names:
            if feature not in input_df.columns:
                input_df[feature] = 0

        input_df = input_df[feature_names]
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        confidence = max(model.predict_proba(input_scaled)[0])  # Get the confidence score

        return jsonify({
            "Downtime": "Yes" if prediction == 1 else "No",
            "Confidence": round(confidence, 2)
        }), 200

    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": f"Error during prediction: {str(e)}"}), 500




if __name__ == '__main__':
    app.run(debug=True)

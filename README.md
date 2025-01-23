# Predictive Analysis for Manufacturing Operations

## Overview
This project implements a RESTful API for predictive analysis in manufacturing operations. It uses a machine learning model to predict machine downtime based on input parameters like temperature and runtime. The API includes endpoints for uploading data, training the model, and making predictions.

---

## Setup Instructions

### Prerequisites
- Python 3.7 or later installed on your system.
- Install Postman or cURL for API testing (optional).

### Steps to Set Up and Run the API

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd predictive_analysis
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Flask Application**:
   ```bash
   python app.py
   ```

4. **Access the API**:
   The API will be running at `http://127.0.0.1:5000`.

---

## API Endpoints

### 1. **Upload Dataset**
- **Endpoint**: `/upload`
- **Method**: `POST`
- **Description**: Upload a CSV file containing the dataset for training.
- **Input**: A CSV file with columns such as `Machine_ID`, `Temperature`, `Run_Time`, and `Downtime_Flag`.

#### Example Request (cURL):
```bash
curl -X POST -F "file=@Synthetic_Machine_Data.csv" http://127.0.0.1:5000/upload
```

#### Example Response:
```json
{
  "message": "Dataset uploaded successfully!"
}
```

---

### 2. **Train Model**
- **Endpoint**: `/train`
- **Method**: `POST`
- **Description**: Trains the machine learning model on the uploaded dataset and saves it for future predictions.

#### Example Request (cURL):
```bash
curl -X POST http://127.0.0.1:5000/train
```

#### Example Response:
```json
{
  "accuracy": 0.92,
  "f1_score": 0.89
}
```

---

### 3. **Predict Downtime**
- **Endpoint**: `/predict`
- **Method**: `POST`
- **Description**: Makes predictions based on input data (temperature and runtime).
- **Input**: JSON object with the following keys:
  - `Temperature`
  - `Run_Time`

#### Example Request (cURL):
```bash
curl -X POST -H "Content-Type: application/json" \
-d '{"Temperature": 75, "Run_Time": 120}' http://127.0.0.1:5000/predict
```

#### Example Response:
```json
{
  "Downtime": "Yes",
  "Confidence": 0.87
}
```

---

## File Structure
```
predictive_analysis/
├── app.py                 # Flask application
├── Synthetic_Machine_Data.csv # Sample dataset
├── requirements.txt       # Dependencies
├── README.md              # Documentation
└── uploads/               # Directory for uploaded files (optional)
```

---

## Notes
- Ensure the dataset includes the necessary columns (`Temperature`, `Run_Time`, `Downtime_Flag`) before uploading.
- If errors occur, check the logs in the terminal for debugging information.
- The `requirements.txt` file contains all the necessary Python libraries for running this application.

---

## Contact
For any issues or questions, feel free to contact the developer.


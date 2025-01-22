# Manufacturing Predictor API

A machine learning API that predicts manufacturing equipment downtime using XGBoost.

## Features
- CSV data upload
- Model training with hyperparameter optimization
- Real-time predictions with confidence scores
- Engineered features for improved accuracy
- RESTful API endpoints

## Prerequisites
- Python 3.8+
- pip

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/manufacturing-predictor.git
cd manufacturing-predictor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Generate sample data:
```bash
python utils/data_processor.py
```

4. Start the API:
```bash
uvicorn app:app --reload
```

## API Endpoints

### Upload Data
- **Endpoint:** `POST /upload`
- **Input:** CSV file with columns: Machine_ID, Temperature, Run_Time
```bash
curl -X POST -F "file=@data/sample_data.csv" http://localhost:8000/upload
```
```json
{
    "message": "File uploaded successfully"
}
```

### Train Model
- **Endpoint:** `POST /train`
- **Returns:** Model performance metrics
```bash
curl -X POST http://localhost:8000/train
```
```json
{
    "accuracy": 0.92,
    "f1_score": 0.91
}
```

### Make Prediction
- **Endpoint:** `POST /predict`
- **Input:** JSON with Temperature and Run_Time
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"Temperature": 90, "Run_Time": 250}' \
  http://localhost:8000/predict
```
```json
{
    "Downtime": "Yes",
    "Confidence": 0.89
}
```

## Project Structure
```
manufacturing_predictor/
├── requirements.txt
├── config.py
├── app.py
├── models/
│   ├── __init__.py
│   └── predictor.py
├── utils/
│   ├── __init__.py
│   └── data_processor.py
└── data/
    └── sample_data.csv
```

## Model Details
- XGBoost classifier with GridSearchCV optimization
- Feature engineering including:
  - Time until maintenance
  - Temperature rate
  - Critical zone detection
  - Polynomial feature interactions

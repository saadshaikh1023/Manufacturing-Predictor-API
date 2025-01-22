from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import uvicorn
from models.predictor import ModelTrainer
from utils.data_processor import DataProcessor
import config

app = FastAPI()
model_trainer = ModelTrainer()
data_processor = DataProcessor()

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(400, "Only CSV files are allowed")
    
    df = pd.read_csv(file.file)
    if not all(col in df.columns for col in config.REQUIRED_COLUMNS):
        raise HTTPException(400, f"CSV must contain columns: {config.REQUIRED_COLUMNS}")
    
    await file.seek(0)
    file_path = f"{config.UPLOAD_FOLDER}/{file.filename}"
    with open(file_path, "wb+") as file_object:
        file_object.write(await file.read())
    
    return {"message": "File uploaded successfully"}

@app.post("/train")
async def train_model():
    try:
        metrics = model_trainer.train()
        return JSONResponse(content=metrics)
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/predict")
async def predict(data: dict):
    required_features = ["Temperature", "Run_Time"]
    if not all(feature in data for feature in required_features):
        raise HTTPException(400, f"Request must contain: {required_features}")
    
    try:
        prediction = model_trainer.predict(data)
        return prediction
    except Exception as e:
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
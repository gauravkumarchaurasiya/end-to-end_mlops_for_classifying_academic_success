from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from sklearn.pipeline import Pipeline
import uvicorn
from data_models import PredictionDataset
import pandas as pd
import joblib
from pathlib import Path
from src.models.models_list import best_tuned_model1
from src.logger import logging

app = FastAPI()

# Mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

current_file_path = Path(__file__).parent
model_name = best_tuned_model1.lower() + "_tuned.joblib"
model_path = current_file_path / "models" / "tuned_models" / model_name
preprocessor_path = current_file_path / "models" / "transformers" / "preprocessor.joblib"
output_transformer_path = current_file_path / "models" / "transformers" / "label_encoder.joblib"

try:
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    output_transformer = joblib.load(output_transformer_path)

    model_pipe = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('classification', model)
    ])
except Exception as e:
    logging.error(f"Error loading model or transformers: {e}")
    raise e

@app.get('/')
def home():
    return HTMLResponse(content=open('static/index.html').read(), status_code=200)

@app.post('/predictions')
def do_predictions(test_data: PredictionDataset):
    try:
        data_dict = test_data.dict(by_alias=True)
        X_test = pd.DataFrame([data_dict])
        predictions = model_pipe.predict(X_test)
        predictions = predictions.tolist()
        return {"predicted_academic_success_score": predictions[0]}
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    uvicorn.run(app="app:app", host="0.0.0.0", port=8000)

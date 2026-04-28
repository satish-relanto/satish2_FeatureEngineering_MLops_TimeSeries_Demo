from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import numpy as np
from typing import List
import os
import pandas as pd

class PredictRequest(BaseModel):
    data: List[List[float]]

class PredictResponse(BaseModel):
    predictions: List[float]

app = FastAPI(title="House Price Predictor")

mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
mlflow.set_tracking_uri(mlflow_tracking_uri)

model = None
MODEL_LOAD_ERROR = None
EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "house-prices-local")
MODEL_NAME = os.environ.get("MODEL_NAME", "house_price_model")
FEATURE_COLUMNS = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]

@app.on_event("startup")
async def load_model():
    global model, MODEL_LOAD_ERROR
    MODEL_LOAD_ERROR = None
    try:
        print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
        # Try to load registered model first
        try:
            client = mlflow.tracking.MlflowClient()
            versions = client.search_model_versions(f"name = '{MODEL_NAME}'")
            if versions:
                latest = max(versions, key=lambda version: int(version.version))
                model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{latest.version}")
                print(f"Loaded registered model '{MODEL_NAME}' version {latest.version}")
                return
        except Exception as e:
            print(f"Failed to load registered model: {e}")

        # Fallback to latest run in experiment
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments(filter_string=f"name = '{EXPERIMENT_NAME}'")
        if not experiments:
            print(f"No experiment named '{EXPERIMENT_NAME}' found")
            return
        experiment = experiments[0]
        runs = client.search_runs(experiment_ids=[experiment.experiment_id],
                                  order_by=["start_time DESC"], max_results=1)
        if runs:
            run_id = runs[0].info.run_id
            uri = f"runs:/{run_id}/model"
            model = mlflow.sklearn.load_model(uri)
            print(f"Loaded model from run {run_id} in experiment '{EXPERIMENT_NAME}'")
            print(f"Model type: {type(model)}")
            if hasattr(model, 'named_steps'):
                print(f"Pipeline steps: {list(model.named_steps.keys())}")
        else:
            print(f"No runs found in experiment '{EXPERIMENT_NAME}'")
    except Exception as e:
        MODEL_LOAD_ERROR = str(e)
        print(f"Failed to load model on startup: {e}")

@app.get("/health")
async def health():
    return {
        "status": "ok" if model is not None else "degraded",
        "model_loaded": model is not None,
        "tracking_uri": mlflow.get_tracking_uri(),
        "experiment_name": EXPERIMENT_NAME,
        "model_name": MODEL_NAME,
        "error": MODEL_LOAD_ERROR,
    }

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    global model
    try:
        print("Incoming request data:", req.data)
        if model is None:
            print("Model not loaded")
            raise HTTPException(
                status_code=503,
                detail="Model is not loaded. Run training first with `python -m src.train`.",
            )

        arr = np.array(req.data)
        print("Input array shape:", arr.shape)
        if arr.ndim != 2 or arr.shape[1] != len(FEATURE_COLUMNS):
            raise HTTPException(
                status_code=400,
                detail=f"Expected a 2D array with {len(FEATURE_COLUMNS)} features per row.",
            )

        # Convert to DataFrame with correct column names
        df = pd.DataFrame(arr, columns=FEATURE_COLUMNS)
        print("Input DataFrame shape:", df.shape)
        print("Input DataFrame columns:", df.columns.tolist())

        preds = model.predict(df).tolist()
        print("Predictions:", preds)
        return {"predictions": preds}

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        import traceback
        print(f"Error in /predict: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

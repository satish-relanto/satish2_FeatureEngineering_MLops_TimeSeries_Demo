# MLOps House Price Project Execution Guide

This guide explains how to run the complete project locally and how to showcase it to juniors.

## 1. Open the Project

Open PowerShell and move into the project root:

```powershell
cd C:\MLops-house-price
```

All commands should be run from this folder.

## 2. Activate the Virtual Environment

If the virtual environment already exists:

```powershell
.venv\Scripts\activate
```

If `.venv` does not exist on another machine:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Explanation:

The virtual environment keeps this project's Python packages separate from system Python.

## 3. Run Tests

```powershell
python -m pytest -q
```

Expected result:

```text
2 passed
```

Explanation:

The tests check whether preprocessing and training pipeline components are working correctly.

## 4. Train the Model

```powershell
python -m src.train --experiment-name house-prices-local
```

Expected output will look similar to:

```text
Finished run <run_id> RMSE = 0.50...
```

Explanation:

- `src.data` loads the housing dataset.
- `src.preprocess` handles missing values and feature scaling.
- `src.model` creates the Random Forest model pipeline.
- `src.train` trains the model and logs the result to MLflow.
- RMSE is the model error metric. Lower RMSE is better.

## 5. Start the FastAPI Application

```powershell
uvicorn src.serve:app --host 127.0.0.1 --port 8000
```

Keep this terminal running.

Explanation:

FastAPI exposes the trained machine learning model as a web API.

## 6. Check API Health

Open a second PowerShell terminal:

```powershell
cd C:\MLops-house-price
.venv\Scripts\activate
Invoke-RestMethod http://127.0.0.1:8000/health
```

Expected response should include:

```text
status       : ok
model_loaded : True
```

Explanation:

This confirms that the API is running and the trained model has been loaded successfully.

## 7. Make a Prediction

Run this in the second terminal:

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/predict `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"data": [[8, 41.0, 6.6, 0.0, 3.4, 6.3, 2.0, 1.0]]}'
```

Expected response will look similar to:

```text
predictions
-----------
{3.32...}
```

Explanation:

The API receives 8 housing features and returns a predicted house value.

## 8. Open Interactive API Docs

Open this URL in your browser:

```text
http://127.0.0.1:8000/docs
```

Explanation:

FastAPI automatically creates interactive API documentation. Juniors can test `/health` and `/predict` directly from the browser.

## 9. Optional: Start MLflow UI

Open another terminal:

```powershell
cd C:\MLops-house-price
.venv\Scripts\activate
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri ./mlruns --default-artifact-root ./mlruns
```

Open this URL in your browser:

```text
http://127.0.0.1:5000
```

Explanation:

MLflow tracks training runs, metrics such as RMSE, parameters, and saved model artifacts.

## 10. Suggested Showcase Flow

Use this explanation order while presenting:

1. We load housing data.
2. We preprocess it with median imputation and scaling.
3. We train a Random Forest regression model.
4. We log model results into MLflow.
5. We serve the trained model using FastAPI.
6. We send JSON input and receive predictions.

## 11. Stop Running Servers

To stop FastAPI or MLflow, go to the terminal where the server is running and press:

```text
Ctrl + C
```

## Quick Command Summary

```powershell
cd C:\MLops-house-price
.venv\Scripts\activate
python -m pytest -q
python -m src.train --experiment-name house-prices-local
uvicorn src.serve:app --host 127.0.0.1 --port 8000
```

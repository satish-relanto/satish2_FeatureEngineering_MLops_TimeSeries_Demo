# Feature Engineering and MLOps Time Series Demo

This repository contains learning material and a working MLOps demo for feature engineering, MLflow experiment tracking, model training, and FastAPI model serving.

## Repository Contents

- `day2_case_study.md` - Case study explaining how feature engineering can improve model performance.
- `day2_feature_engineering.ipynb` - Notebook for Day 2 feature engineering exercises.
- `TimeSeries.pdf` - Time series reference material.
- `MLops-house-price/` - End-to-end house price prediction MLOps project.

## MLOps House Price Project

The `MLops-house-price` folder includes:

- Python source code for data loading, preprocessing, model training, and serving.
- MLflow tracking output in `mlruns/`.
- FastAPI prediction service.
- Docker and Docker Compose setup.
- Pytest tests for preprocessing and training behavior.

## Quick Start

From the repository root:

```powershell
cd MLops-house-price
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q
python -m src.train --experiment-name house-prices-local
uvicorn src.serve:app --host 127.0.0.1 --port 8000
```

Open the API docs:

```text
http://127.0.0.1:8000/docs
```

Optional MLflow UI:

```powershell
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri ./mlruns --default-artifact-root ./mlruns
```

Then open:

```text
http://127.0.0.1:5000
```

## Docker

From `MLops-house-price`:

```powershell
docker-compose up --build
```

This starts:

- MLflow UI on port `5000`
- FastAPI service on port `8000`

## MLflow Artifacts

The `mlruns/` folder is ignored because MLflow run artifacts can become large quickly. Train the model locally to regenerate MLflow runs when needed:

```powershell
cd MLops-house-price
python -m src.train --experiment-name house-prices-local
```

## Project Notes

- Run Python commands from inside `MLops-house-price` so imports resolve correctly.
- Use `.env.example` as the template for local environment variables.
- Local virtual environments, caches, IDE files, secrets, and MLflow run artifacts are ignored.

import logging
from dotenv import load_dotenv
import sys
import mlflow
import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager
from mlflow.pyfunc import load_model
import numpy as np
from pydantic import BaseModel, conlist
from typing import Union
import os
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


class Data(BaseModel):
    data: conlist(float, min_length=2, max_length=2)
    run_id: Union[str, None] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, handlers=[
        logging.FileHandler(filename="rest.log", mode="w"),
        logging.StreamHandler(sys.stdout)
    ])
    yield


app = FastAPI(lifespan=lifespan)

mlflow.set_tracking_uri("http://mlflow:5000")


@app.get("/predict")
async def predict(data: Data) -> list[float]:
    if data.run_id is None:
        model = load_model('runs:/' + os.environ["production_model"] + '/sklearn-model')
        predict_result = model.predict(np.array([data.data])).tolist()
        logging.info(f'successful /predict call on model with run_id: {os.environ["production_model"]}')
        return predict_result
    model = load_model('runs:/' + data.run_id + '/sklearn-model')
    predict_result = model.predict(np.array([data.data])).tolist()
    logging.info(f'successful /predict call on model with run_id: {data.run_id}')
    return predict_result


@app.post("/update")
async def update() -> None:
    with mlflow.start_run() as run:
        X, y = make_circles(noise=0.2, factor=0.5, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        clf = DecisionTreeClassifier()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        signature = infer_signature(X_test, y_pred)

        mlflow.log_metrics({"accuracy": accuracy_score(y_test, y_pred)})
        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="sklearn-model",
            signature=signature,
            registered_model_name="sk-learn-decision-tree-reg-model",
        )
        logging.info(f'successful /update call, created model with run_id: {run.info.run_id}')


if __name__ == "__main__":
    load_dotenv('../.env')
    uvicorn.run(app='app:app', host=os.environ.get("REST_SERVICE_HOST"),
                port=int(os.environ.get("REST_SERVICE_DOCKER_PORT")), reload=True)

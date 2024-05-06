import logging
from dotenv import load_dotenv
import mlflow
from mlflow.pyfunc import load_model
import numpy as np
import os
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from celery import Celery


mlflow.set_tracking_uri("http://mlflow:5000")
load_dotenv('../.env')
celery_worker = Celery(__name__, backend=os.environ.get("CELERY_BACKEND"), broker=os.environ.get("CELERY_BROKER"),
                       include=['worker.worker'])


@celery_worker.task
def predict(id: str, data: list[float]):
    if id is None:
        model = load_model('runs:/' + os.environ["production_model"] + '/sklearn-model')
        predict_result = model.predict(np.array([data])).tolist()
        logging.info(f'successful /predict call on model with run_id: {os.environ["production_model"]}')
        return predict_result
    model = load_model('runs:/' + id + '/sklearn-model')
    predict_result = model.predict(np.array([data])).tolist()
    logging.info(f'successful /predict call on model with run_id: {id}')
    return predict_result


@celery_worker.task
def update():
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

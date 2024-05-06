import logging
from dotenv import load_dotenv
from http import HTTPStatus
import sys
import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager
from typing import Any
import os
from starlette.responses import JSONResponse
from worker.worker import predict, update
from models import Data
from celery.result import AsyncResult


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, handlers=[
        logging.FileHandler(filename="rest.log", mode="w"),
        logging.StreamHandler(sys.stdout)
    ])
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/predict")
async def predict_handler(data: Data) -> dict:
    result = predict.delay(data.run_id, data.data)
    return {"task_id": str(result)}


@app.post("/update")
async def update_handler() -> dict:
    result = update.delay()
    return {"task_id": str(result)}


@app.get("/result/{task_id}")
async def result_handler(task_id: str):
    """Fetch result for given task_id"""
    task = AsyncResult(task_id)
    if not task.ready():
        return JSONResponse(status_code=HTTPStatus.ACCEPTED.value, content={'task_id': task_id, 'status': 'Processing'})
    result = task.get()
    return {'task_id': task_id, 'status': 'Success', 'result': str(result)}


if __name__ == "__main__":
    load_dotenv('../.env')
    uvicorn.run(app='app:app', host=os.environ.get("REST_SERVICE_HOST"),
                port=int(os.environ.get("REST_SERVICE_DOCKER_PORT")), reload=True)

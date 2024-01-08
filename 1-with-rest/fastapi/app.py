from fastapi import FastAPI
from mlflow.pyfunc import load_model
import numpy as np
from pydantic import BaseModel, conlist


class Data(BaseModel):
    data: conlist(float, min_length=2, max_length=2)
    run_id: str


app = FastAPI()


@app.post("/predict")
async def predict(data: Data):
    print(data.data)
    print(data.run_id)
    model = load_model('runs:/' + data.run_id + '/sklearn-model')
    return model.predict(np.array([data.data]))

import mlflow
import numpy as np
from mlflow.pyfunc import load_model

mlflow.set_tracking_uri("http://localhost:5000")
model = load_model('runs:/8eb76dc1b5c241e890f3cd28cff09789/sklearn-model')

prediction = model.predict(np.array([[1.0, 1.0]]))
print(prediction)
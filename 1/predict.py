import mlflow
import numpy as np
from mlflow.pyfunc import load_model

mlflow.set_tracking_uri("http://localhost:5000")
model = load_model('runs:/5ddd2075110b4615b1b6e756f74863a9/sklearn-model')

prediction = model.predict(np.array([[1.0, 1.0]]))
print(prediction.tolist())
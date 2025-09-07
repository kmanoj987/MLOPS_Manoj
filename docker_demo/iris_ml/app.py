# iris/app.py
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

# Input schema
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
def predict(data: IrisInput):
    X = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    prediction = model.predict(X)[0]
    return {"predicted_class": int(prediction)}

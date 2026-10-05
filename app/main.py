from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

from fraud_detection.model.predict import predict

app = FastAPI(title="Fraud Detection API")


class Transaction(BaseModel):
    step: int
    type: str
    amount: float
    oldbalanceOrg: float
    oldbalanceDest: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_fraud(tx: Transaction):
    df = pd.DataFrame([tx.model_dump()])
    predictions, probabilities = predict(df)
    return {
        "is_fraud": bool(predictions[0]),
        "fraud_probability": float(probabilities[0]),
    }
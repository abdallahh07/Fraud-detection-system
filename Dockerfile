FROM python:3.10-slim

RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config ./config
COPY fraud_detection ./fraud_detection
COPY app ./app
COPY trained_models ./trained_models

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
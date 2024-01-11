FROM python:3.10
LABEL authors="olegafanasev"
WORKDIR /app
COPY . .
RUN pip install mlflow
RUN pip install "fastapi[all]"
CMD python model.py
CMD uvicorn app:app --host 0.0.0.0 --port 1000
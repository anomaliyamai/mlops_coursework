FROM python:3.10
LABEL authors="olegafanasev"
WORKDIR /mlflow
COPY . .
RUN pip install mlflow
CMD mlflow server --host 0.0.0.0
CMD python model.py
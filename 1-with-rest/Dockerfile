FROM python:3.10-slim
LABEL authors="olegafanasev"
COPY .env .
RUN pip install mlflow-skinny &&  \
    pip install Flask &&  \
    pip install gunicorn &&  \
    pip install querystring-parser && \
    pip install numpy==1.26.1 && \
    pip install scikit-learn==1.3.2 && \
    pip install pandas &&  \
    pip install fastapi && \
    pip install uvicorn && \
    pip3 install python-dotenv \
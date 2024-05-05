FROM service
LABEL authors="olegafanasev"
WORKDIR worker
COPY . .
WORKDIR /
CMD celery -A worker.worker.celery_worker worker --loglevel=INFO
import os
import mlflow.cli
from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv('../.env')
    mlflow.cli.server(["--host", os.environ.get("MLFLOW_HOST"), "--port", os.environ.get("MLFLOW_DOCKER_PORT")])
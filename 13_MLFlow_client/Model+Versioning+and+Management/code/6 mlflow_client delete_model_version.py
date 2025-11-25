import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")

client = MlflowClient()

# model should be in archived stage before deleting 
client.delete_model_version(
    name="linear-regression-model",
    version="1"
)


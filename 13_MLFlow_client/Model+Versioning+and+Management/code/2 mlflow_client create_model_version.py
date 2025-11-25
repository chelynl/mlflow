import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")

client = MlflowClient()

# Add model version to empty registry that was just created
client.create_model_version(
    name="linear-regression-model",
    source="runs:/9feca0d4de424413812aade6809a7704/model", # add run id of model
    tags={
        "framework": "sklearn",
        "hyperparameters": "alpha and l1_ratio"
    },
    description="A second linear regression model trained with alpha and l1_ratio prameters."
)

# client.create_model_version(
#     name="linear-regression-model",
#     source="runs:/e5a1b7eb8fd34b0599620610ee7aec92/model", # add run id of model
# )
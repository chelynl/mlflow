import mlflow
from mlflow import MlflowClient
from mlflow.entities import ViewType

mlflow.set_tracking_uri("http://127.0.0.1:5000")

client = MlflowClient()

experiments = client.search_experiments(view_type=ViewType.ALL,
                                        filter_string="tags.`version` = 'v1' AND tags.`framework` = 'sklearn'",
                                        order_by=["experiment_id ASC"]
                                        )

for exp in experiments:
    print(f"Experiment Name: {exp.name}, Experiment ID: {exp.experiment_id}")

import mlflow
from mlflow import MlflowClient
from mlflow.entities import ViewType

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# initialize mlflow client class object
client = MlflowClient()

# create experiment
experiment_id = client.create_experiment(
    name="Client exp",
    tags={
        "Version": "v1",
        "Priority": "P1"
    }
)

print(f"Experiment id: {experiment_id}")

# manually set experiment tags
client.set_experiment_tag(experiment_id, "framework", "sklearn")

# get experiment by id or name
experiment = client.get_experiment(experiment_id)
print(f"Experiment name: {experiment.name}")

experiment = client.get_experiment_by_name("Client exp")
print(f"Experiment id: {experiment.experiment_id}")

# Print experiment info
print("Name: {}".format(experiment.name))
print("Id: {}".format(experiment.experiment_id))
print("Lifecycle stage: {}".format(experiment.lifecycle_stage))
print("Tags: {}".format(experiment.tags))
print("Artifact location: {}".format(experiment.artifact_location))
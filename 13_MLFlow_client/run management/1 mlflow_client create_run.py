import mlflow
from mlflow import MlflowClient
# get view type value to view type parameter to search_experiments()
from mlflow.entities import ViewType

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# initialize mlflow client class object
client = MlflowClient()

# create an empty run (not active) --> run object which you can extract info from
# prepares run but does not run actual ML code
run = client.create_run(
    experiment_id="1",
    tags={
        "Version": "v1",
        "Priority": "P1"
    },
    run_name="run from client"
)

print(f"Run tags: {run.data.tags}")
print(f"Experiment id: {run.info.experiment_id}")
print(f"Run id: {run.info.run_id}")
print(f"Run name: {run.info.run_name}")
print(f"lifecycle_stage: {run.info.lifecycle_stage}")
print(f"status: {run.info.status}")
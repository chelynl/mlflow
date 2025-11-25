import mlflow
from mlflow import MlflowClient
from mlflow.entities import ViewType

mlflow.set_tracking_uri("http://127.0.0.1:5000")

client = MlflowClient()

# Get list of all experiments
experiments = client.search_experiments(view_type=ViewType.ALL)
# print(experiments)
# print(type(experiments)) # list of exps that you can iterate over

# # get list of experiments in ascending order of experiment id
# experiments = client.search_experiments(view_type=ViewType.ACTIVE_ONLY,
#                                         order_by=["experiment_id ASC"]
#                                         )

# # get list of experiments with filter conditions
# experiments = client.search_experiments(view_type=ViewType.ALL,
#                                         filter_string="tags.`version` = 'v1' AND tags.`framework` = 'sklearn'",
#                                         order_by=["experiment_id ASC"]
#                                         )

# experiments = client.search_experiments(view_type=ViewType.ALL,
#                                         filter_string="name = 'Client exp'",
#                                         order_by=["experiment_id ASC"]
#                                         )

for exp in experiments:
    print(f"Experiment Name: {exp.name}, Experiment ID: {exp.experiment_id}")

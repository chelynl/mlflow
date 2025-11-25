import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")


parameters={
    "alpha":0.3,
    "l1_ratio":0.3
}

experiment_name = "Project exp 1"
entry_point = "ElasticNet"

mlflow.projects.run(
    uri=".",
    entry_point=entry_point,
    parameters=parameters,
    experiment_name=experiment_name
)

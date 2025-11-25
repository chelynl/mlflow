import mlflow
from mlflow import MlflowClient

from mlflow.entities import ViewType

mlflow.set_tracking_uri("http://127.0.0.1:5000")

client = MlflowClient()

import time

# List all experiments
experiments = client.search_experiments(view_type=ViewType.ALL)

print(f"Found {len(experiments)} experiments.")

for exp in experiments:
    if exp.name == "Default":
        print(f"Skipping Default experiment (ID: {exp.experiment_id})")
        continue

    print(f"Processing experiment: {exp.name} (ID: {exp.experiment_id}, State: {exp.lifecycle_stage})")
    
    try:
        if exp.lifecycle_stage == 'active':
            client.delete_experiment(exp.experiment_id)
            print(f"Deleted experiment: {exp.name}")
        elif exp.lifecycle_stage == 'deleted':
            # Rename to free up the name
            new_name = f"{exp.name}_deleted_{int(time.time())}"
            client.rename_experiment(exp.experiment_id, new_name)
            print(f"Renamed deleted experiment to: {new_name}")
            
    except Exception as e:
        print(f"Error processing experiment {exp.name}: {e}")

print("Cleanup complete.")

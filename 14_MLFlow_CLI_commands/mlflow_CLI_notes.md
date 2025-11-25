# MLflow CLI Notes

The MLflow CLI provides a convenient way to perform various operations related to tracking, managing, and deploying ML projects.

## Prerequisites

Before running CLI commands, ensure you point to the correct tracking server.
If you are using a remote or local server (not the default `./mlruns`), set the `MLFLOW_TRACKING_URI` environment variable.

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

Or prepend it to your commands:

```bash
MLFLOW_TRACKING_URI=http://127.0.0.1:5000 mlflow <command>
```

## Debugging

Check the environment and configuration of your MLflow installation.

```bash
# Print useful information
mlflow doctor

# Mask environment variables to prevent leaking sensitive info
mlflow doctor --mask-envs
```

## Artifacts

Manage artifacts in the artifact repository.

```bash
# List artifacts for a specific run
# Example: mlflow artifacts list --run-id e5a1b7eb8fd34b0599620610ee7aec92
mlflow artifacts list --run-id <RUN_ID>

# Download artifacts to a local destination
# Example: mlflow artifacts download --run-id e5a1b7eb8fd34b0599620610ee7aec92 --dst-path cli_artifact
mlflow artifacts download --run-id <RUN_ID> --dst-path <LOCAL_PATH>

# Log (upload) artifacts from a local directory
# Example: mlflow artifacts log-artifacts --local-dir cli_artifact --run-id e5a1b7eb8fd34b0599620610ee7aec92 --artifact-path cli_artifact
mlflow artifacts log-artifacts --local-dir <LOCAL_DIR> --run-id <RUN_ID> --artifact-path <ARTIFACT_PATH>
```

## Database

Manage the MLflow tracking database.

```bash
# Upgrade the database schema
mlflow db upgrade <DATABASE_URI>
# Example:
mlflow db upgrade sqlite:///mlflow.db
```

## Experiments

Manage experiments.

```bash
# Create an experiment
mlflow experiments create --experiment-name <NAME>

# Rename an experiment
mlflow experiments rename --experiment-id <ID> --new-name <NEW_NAME>

# Delete an experiment
mlflow experiments delete --experiment-id <ID>

# Restore a deleted experiment
mlflow experiments restore --experiment-id <ID>

# Search experiments
mlflow experiments search --view "all"

# Export experiment runs to CSV
mlflow experiments csv --experiment-id <ID> --filename <OUTPUT_FILE.csv>
```

## Runs

Manage runs within experiments.

```bash
# List runs for an experiment
mlflow runs list --experiment-id <ID> --view "all"

# Describe a specific run
mlflow runs describe --run-id <RUN_ID>

# Delete a run
mlflow runs delete --run-id <RUN_ID>

# Restore a deleted run
mlflow runs restore --run-id <RUN_ID>
```

import mlflow
import joblib
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import numpy as np
import os

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Set working directory where data is stored
os.chdir("/Users/chelynlee/projects/MLFlow_Udemy/13_MLFlow_client/run management")

client = MlflowClient()

# get run id of previous killed run
run = client.get_run("9feca0d4de424413812aade6809a7704")
# get metric history of previous killed run
metrics = client.get_metric_history(run.info.run_id, 'rmse')

for metric in metrics:
    print(f"Step: {metric.step}, Timestamp: {metric.timestamp}, Value: {metric.value}")
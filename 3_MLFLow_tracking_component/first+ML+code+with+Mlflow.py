import warnings
import argparse
import logging

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

'''
MLFlow Tracking
- Run: single execution of code; records code version, hyperparams, metrics, tags, etc.
- Experiment: can have n num of runs in it; group of runs that you can organize and group
- Experiment/Run gets unique ID and name (used to log/retrieve metadata associated with it)

'''
# import mlflow and its sklearn module
import mlflow
import mlflow.sklearn

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# get arguments from command
'''
Use run commands to quickly execute code with different parameters from terminal:
python first_ML_code_with_Mlflow.py --alpha 0.6 --l1_ratio 0.9
'''
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, required=False, default=0.5)
parser.add_argument("--l1_ratio", type=float, required=False, default=0.8)
args = parser.parse_args()

# evaluation function
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from local
    data = pd.read_csv('/Users/chelynlee/Downloads/MLFlow_Udemy/red-wine-quality.csv')
    data.to_csv("/Users/chelynlee/Downloads/MLFlow_Udemy/red-wine-quality.csv", index=False)

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = args.alpha
    l1_ratio = args.l1_ratio

    # mlflow.set_tracking_uri(uri='/Users/chelynlee/Documents/OFE/test1') # creates folder test1 in specified dir for mlflow tracking
    # mlflow.set_tracking_uri(uri='./mytracks') # creates folder mytracks in current working dir for mlflow tracking
    # mlflow.set_tracking_uri(uri='') # creates default mlruns folder in current working dir for mlflow tracking
    # print("Current tracking uri: {}".format(mlflow.get_tracking_uri())) # see where mlflow is tracking

    # define experiment name
    exp = mlflow.set_experiment(experiment_name="experiment_1")
    # exp = mlflow.set_experiment(experiment_name="exp_for_uri")

    # To associate this run with the experiment, wrap ML code inside mlflow.start_run() with experiment_id
    with mlflow.start_run(experiment_id=exp.experiment_id):
        # Model training inside to record executions metadata (can later retrieve using run ID)
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        # Define all entities to be logged in this run
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        # log model artifacts (trained model; can be used for deployment, scoring, etc.)
        # creates a separate models folder to store model artifacts for corresponding run under experiment folder / usually stored in remote server
        mlflow.sklearn.log_model(lr, "mymodel") # mymodel = name of artifact

# When you run an MLFlow tracking code, a directory named "mlruns" is created in the current working directory
# it is the default local file system store for MLFlow (stores all metadata and artifacts related to your MLFlow runs)
# The experiment ID is the folder name in mlruns and inside it are folders with run IDs
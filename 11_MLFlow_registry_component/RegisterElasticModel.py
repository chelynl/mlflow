import pickle
import mlflow
import mlflow.sklearn

# load the model into memory
filename = '/Users/chelynlee/projects/MLFlow_Udemy/11_MLFlow_registry_component/elastic-net-regression.pkl'
loaded_model = pickle.load(open(filename, "rb"))

# set tracking uri and create experiment
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
exp = mlflow.set_experiment(experiment_name="experiment_register_outside")

# start run
mlflow.start_run()
# within the run, log the model to tracking server and register it in the registry
mlflow.sklearn.log_model(
    loaded_model,
    'model',
    serialization_format='cloudpickle',
    registered_model_name="elastic-net-regression-outside-mlflow"
)
# end run
mlflow.end_run()

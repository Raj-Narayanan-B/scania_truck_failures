from pathlib import Path  # type: ignore
from src.utils import crypter
dagshub_config = crypter(encrypt_or_decrypt='decrypt', file_name='dagshub')
mlflow_config = crypter(encrypt_or_decrypt='decrypt', file_name='mlflow')


CONFIG_PATH = Path('config/config.yaml')
PARAMS_PATH = Path('params.yaml')
SCHEMA_PATH = Path('schema.yaml')
TEST_DATA = "remote_aps_failure_testing_set.csv"
TRAIN_DATA = "remote_aps_failure_training_set.csv"
REPO = dagshub_config['Repo']
BUCKET = dagshub_config['BUCKET']
TOKEN = dagshub_config['Token']
MLFLOW_TRACKING_URI = mlflow_config['MLFLOW_TRACKING_URI']
MLFLOW_TRACKING_PASSWORD = mlflow_config['MLFLOW_TRACKING_PASSWORD']
MLFLOW_TRACKING_USERNAME = mlflow_config['MLFLOW_TRACKING_USERNAME']
# TEMP_PATH = "artifacts/data/temp"

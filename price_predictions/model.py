import pickle
import subprocess

import mlflow
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score


class LGBM:
    def __init__(self, exp_id, params=dict()):
        mlflow.autolog()
        self.experiment_id = exp_id
        self.regressor = LGBMRegressor()
        self.regressor.set_params(**params)

    def fit(self, *args, **kwargs):
        with mlflow.start_run(experiment_id=self.experiment_id) as run:
            self.run_id = run.info.run_id
            mlflow.log_param("GIT_COMMIT_HASH", get_commit_hash())
            return self.regressor.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        with mlflow.start_run(
            experiment_id=self.experiment_id, run_id=self.run_id
        ):  # noqa: E501
            return self.regressor.predict(*args, **kwargs)

    def calc_scores(self, pred, true):
        with mlflow.start_run(
            experiment_id=self.experiment_id, run_id=self.run_id
        ):  # noqa: E501
            r2 = r2_score(true, pred)
            rmse = mean_squared_error(true, pred) ** 0.5
            mae = median_absolute_error(true, pred)

            mlflow.log_metric("r2_score", r2)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def dump(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)


def get_commit_hash():
    s = subprocess.check_output("git rev-parse HEAD", shell=True).decode()
    return s.strip()

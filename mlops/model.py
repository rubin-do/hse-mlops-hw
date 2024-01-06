import pickle

from lightgbm import LGBMRegressor


class LGBM:
    def __init__(self, params=dict(), infer_path=None):
        if infer_path is not None:
            with open(infer_path, "rb") as f:
                self.regressor = pickle.load(f)
        else:
            self.regressor = LGBMRegressor()
            self.regressor.set_params(**params)

    def fit(self, *args, **kwargs):
        return self.regressor.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.regressor.predict(*args, **kwargs)

    def dump(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.regressor, f)

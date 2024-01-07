import logging

import hydra
import mlflow
from omegaconf import DictConfig, OmegaConf

from price_predictions.dvc import load_data
from price_predictions.model import LGBM
from price_predictions.predict import predict_path


@hydra.main(version_base=None, config_path="configs", config_name="infer")
def main(cfg: DictConfig):
    log = logging.getLogger()
    log.setLevel(logging.DEBUG)
    log.debug(OmegaConf.to_yaml(cfg))

    mlflow.set_tracking_uri(cfg.mlflow.address)

    df = load_data(cfg.dvc.path, cfg.dvc.remote)

    model = LGBM.load(cfg.model.infer_path)
    pred = predict_path(model, df, cfg.dataset.features, cfg.dataset.target)

    model.calc_scores(pred[cfg.dataset.target], df[cfg.dataset.target])

    pred.to_csv(cfg.model.predict_path)


if __name__ == "__main__":
    main()

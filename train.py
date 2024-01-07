import logging

import hydra
import mlflow
from omegaconf import DictConfig, OmegaConf

from price_predictions.dvc import load_data
from price_predictions.model import LGBM
from price_predictions.prepare import prepare

config_levels = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):
    log = logging.getLogger()
    log.setLevel(config_levels[cfg.log_level])
    log.debug(OmegaConf.to_yaml(cfg))

    mlflow.set_tracking_uri(cfg.mlflow.address)

    experiment = mlflow.set_experiment(cfg.mlflow.experiment)

    df = load_data(cfg.dvc.path, cfg.dvc.remote)
    X_train, y_train = prepare(df, cfg.dataset.features, cfg.dataset.target)

    params = OmegaConf.to_container(cfg.model.params)
    model = LGBM(experiment.experiment_id, params)

    model.fit(X_train, y_train)

    model.dump(cfg.model.dump_path)


if __name__ == "__main__":
    main()

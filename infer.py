import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from mlops.dvc import load_data
from mlops.model import LGBM
from mlops.predict import predict_path


@hydra.main(version_base=None, config_path="configs", config_name="infer")
def main(cfg: DictConfig):
    log = logging.getLogger()
    log.setLevel(logging.DEBUG)
    log.debug(OmegaConf.to_yaml(cfg))

    df = load_data(cfg.dvc.path, cfg.dvc.remote)

    model = LGBM(params=dict(), infer_path=cfg.model.infer_path)
    pred = predict_path(model, df, cfg.dataset.features, cfg.dataset.target)

    # calculate stats

    pred.to_csv(cfg.model.predict_path)


if __name__ == "__main__":
    main()

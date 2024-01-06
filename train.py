import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from mlops.dvc import load_data
from mlops.model import LGBM
from mlops.prepare import prepare


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):
    log = logging.getLogger()
    log.setLevel(logging.DEBUG)
    log.debug(OmegaConf.to_yaml(cfg))

    df = load_data(cfg.dvc.path, cfg.dvc.remote)
    X_train, y_train = prepare(df, cfg.dataset.features, cfg.dataset.target)

    params = OmegaConf.to_container(cfg.model.params)
    model = LGBM(params)
    model.fit(X_train, y_train)

    log.debug(model.predict(X_train))

    model.dump(cfg.model.dump_path)


if __name__ == "__main__":
    main()

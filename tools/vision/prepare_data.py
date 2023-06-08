import os
import sys
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)
import uuid
import warnings
warnings.filterwarnings("ignore")

import hydra
import pytorch_lightning as pl
from omegaconf import OmegaConf

from python.itl.vision.data import FewShotDataModule


OmegaConf.register_new_resolver(
    "randid", lambda: str(uuid.uuid4())[:6]
)
@hydra.main(config_path="../../python/itl/configs", config_name="config")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    pl.seed_everything(cfg.seed)

    dm = FewShotDataModule(cfg)
    dm.prepare_data()


if __name__ == "__main__":
    main()

#!/usr/bin/env python

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

from crossdomainhsi.models import get_model
from crossdomainhsi.datasets import get_datamodule
from crossdomainhsi.utils import predict_batch

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
import torch

seed_everything(42)
#torch.use_deterministic_algorithms(True, warn_only=True)

device = 'cuda'

@hydra.main(version_base=None, config_path="../run/conf", config_name="pred_conf")
def main(cfg):
    print(OmegaConf.to_yaml(cfg)) # print config

    # init datamodule + model
    datamodule = get_datamodule(cfg.dataset)
    datamodule.setup()
    model = get_model(cfg.model, datamodule).to(device)

    for batch in datamodule.val_dataloader():
        pred = predict_batch(model, batch, 
                      device=device,
                      sub_batch_size=cfg.model.sub_batch_size)
        print(pred.shape)
        break
    
if __name__ == '__main__':
    main()

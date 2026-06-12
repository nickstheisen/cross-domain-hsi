#!/usr/bin/env python

from crossdomainhsi.models import UNet, LiuNet
from crossdomainhsi.models.deeplabv3 import DeeplabV3Plus
from crossdomainhsi.models.hdcminirocket import MiniRocketSeg
from crossdomainhsi.models.mlp import MLP
from crossdomainhsi.models.hypersl import HyperSLClassification

from omegaconf import open_dict

def get_model(cfg, datamodule):
    with open_dict(cfg):
        cfg.model.n_channels = datamodule.n_channels
        cfg.model.n_classes = datamodule.n_classes
        cfg.model.spatial_size = list(datamodule.img_shape)
        cfg.model.ignore_index = datamodule.undef_idx
        cfg.model.label_def = datamodule.cfg.label_def
        cfg.model.wavelengths_file = datamodule.cfg.wavelengths_file

    if cfg.model.name == 'unet':
        model = UNet(cfg)
    elif cfg.model.name == 'deeplabv3plus':
        model = DeeplabV3Plus(cfg)
    elif cfg.model.name == 'liunet':
        model = LiuNet(cfg)
    elif cfg.model.name == 'minirocket':
        model = MiniRocketSeg(cfg)
    elif cfg.model.name == 'mlp':
        model = MLP(cfg)
    elif cfg.model.name == 'hypersl':
        model = HyperSLClassification(cfg)

    else :
        raise NotImplementedError(f"Model '{cfg.name}' does not exist.")
    
    return model

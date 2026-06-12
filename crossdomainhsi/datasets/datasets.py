#!/usr/bin/env python
from omegaconf import open_dict

import crossdomainhsi

from .hsidrive import HSIDrive
from .hyko2 import HyKo2
from .hyperspectralcity import HyperspectralCityV2
from pathlib import Path

label_def_dir = Path(crossdomainhsi.__file__).parent.joinpath("datasets/labeldefs")

def replace_if_exists(paramname, defaultval, argdict):
    return argdict[paramname] if (paramname in argdict) else defaultval

def get_datamodule(cfg):
    with open_dict(cfg):
        label_def = label_def_dir.joinpath(cfg.label_def)
        cfg.label_def = label_def
    if cfg.name == 'hsidrive':
        datamodule = HSIDrive(cfg)
    elif cfg.name == 'hyko2':
        datamodule = HyKo2(cfg)
    elif cfg.name == 'hcv2':
        datamodule = HyperspectralCityV2(cfg)
    else :
        raise NotImplementedError(f"Dataset '{dataset_name}' is not available.")

    return datamodule

            
            
            

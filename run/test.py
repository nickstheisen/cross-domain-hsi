import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import loggers as pl_loggers

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import wandb

from crossdomainhsi.datasets import get_datamodule
from crossdomainhsi.models import get_model

from datetime import datetime
from pathlib import Path
import os
import numpy as np

# set max_split_size_mb to 512 to avoid fragmentation
os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:256"

valid_datasets = ['hsidrive','hyko2', 'hcv2']
valid_models = ['unet', 'agunet', 'spectr', 'deeplabv3plus', 'liunet', 'minirocket', 'mlp', 'hypersl']



def make_reproducible(manual_seed=42):
    seed_everything(manual_seed, workers=True)
    torch.use_deterministic_algorithms(False, warn_only=True)

@hydra.main(version_base=None, config_path="conf", config_name="test_conf")
def test(cfg):
    print(OmegaConf.to_yaml(cfg))

    torch.set_float32_matmul_precision(cfg.training.mat_mul_precision)
    make_reproducible(cfg.training.seed)

    ## Logging
    log_dir = Path(cfg.logging.path+f"{cfg.logging.project_name}")
    log_dir.mkdir(parents=True, exist_ok=True)
    resume_path = cfg.training.resume_path
    loggers = []

    ts = datetime.now().strftime("%Y%m%d_%H-%M-%S")
    pt = "" # full pretrained
    npt = "" # no pretrained backbone
    if cfg.model.name == 'deeplabv3plus':
        if cfg.model.pretrained_weights is not None:
            pt = "-PT"
        if not cfg.model.pretrained_backbone:
            npt = "-NPT"
    logname_run = f"test-{cfg.dataset.log_name}-{cfg.model.log_name}{pt}{npt}-{ts}"

    if cfg.logging.tb_logger:
        loggers.append(pl_loggers.TensorBoardLogger(
                version=logname_run,
                save_dir=log_dir,
        ))

    if cfg.logging.wb_logger:
        wandb.finish()
        loggers.append(pl_loggers.WandbLogger(
                name=logname_run,
                project=f"{cfg.logging.project_name}",
                save_dir=log_dir,
        ))
    

    ## Data Module
    if cfg.dataset.half_precision:
        precision="16-mixed"
    else:
        precision=32

    if cfg.dataset.pca is not None:
        pca_out_path = Path(cfg.dataset.pca_out_dir)
        
        if not pca_out_path.is_dir():
            raise RuntimeError("`pca_out_dir` must be a directory")
        
    datamodule = get_datamodule(cfg.dataset)
    model = get_model(cfg, datamodule)
    ## Trainer
    trainer = Trainer(
            default_root_dir=log_dir,
            accelerator=cfg.training.accelerator,
            devices=cfg.training.devices, 
            precision=precision,
            logger=loggers,
            )
    
    results = trainer.test(model=model,
            datamodule=datamodule,
            ckpt_path=cfg.model.ckpt,
            weights_only=False
    )[0]

    row = [results["Test/accuracy-micro"]*100.,
                results["Test/accuracy-macro"]*100.,
                results["Test/f1-macro"]*100.,
                results["Test/jaccard"]*100.]
    print(f'\n\nOA: {row[0]:.2f}\nAA: {row[1]:.2f}\nF1: {row[2]:.2f}\nmIoU: {row[3]:.2f}\n\n')

    # with open(cfg.model.ckpt, 'r') as f:
    #     ckpt_dirs = f.read().splitlines()

    # ckpt_basepath = Path(cfg.model.ckpt_basepath)
    # results_table = []
    # for ckpt_dir in ckpt_dirs:
    #     ckpt_path = ckpt_basepath.joinpath(ckpt_dir).joinpath('checkpoints')
    #     ckpt_files = list(ckpt_path.glob('*.ckpt'))
    #     if len(ckpt_files) != 1:
    #         print(f"Something went wrong {len(ckpt_files)} files for {ckpt_path}")

    #     results = trainer.test(model=model,
    #             datamodule=datamodule,
    #             ckpt_path=ckpt_files[0],
    #             weights_only=False
    #                            )
    #     results = results[0]
    #     result_row = [results["Test/accuracy-micro"]*100.,
    #                     results["Test/accuracy-macro"]*100.,
    #                     results["Test/f1-macro"]*100.,
    #                     results["Test/jaccard"]*100.]
    #     results_table.append(result_row)
    # for row in results_table:
    #     print(f'{row[0]:.2f}\t{row[1]:.2f}\t{row[2]:.2f}\t{row[3]:.2f}')

    # print(np.array(results_table).mean(axis=0))

if __name__ == '__main__':
    test()

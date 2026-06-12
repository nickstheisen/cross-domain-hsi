#!/usr/bin/env python

from pathlib import Path
import numpy as np
from typing import List, Any, Optional
from imageio import imread
import h5py

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
import torch

from crossdomainhsi.datasets.transforms import ToTensor, PermuteData, ReplaceLabels, SpectralAverage, InsertEmptyChannelDim
from crossdomainhsi.datasets.utils import label_histogram
from .hsdataset import HSDataModule, HSDataset
from crossdomainhsi.datasets.utils import apply_pca

N_DEBUG_SAMPLES = 5

class HyperspectralCityV2(HSDataModule):

    def __init__(self, cfg):
        super().__init__(cfg)


        self.n_classes = 20
        self.undef_idx=19
        self.filepath_train = self.basepath.joinpath('hcv2_train.h5')
        self.filepath_test = self.basepath.joinpath('hcv2_test.h5')
        self.filepath_val = self.basepath.joinpath('hcv2_val.h5')

        self.preprocess = transforms.Compose([
            ToTensor(half_precision=self.cfg.half_precision),
            ReplaceLabels({255:19})
        ])
        self.transform = None

        if self.cfg.spectral_average:
            self.transform = transforms.Compose([
                self.transform,
                SpectralAverage()
            ])

        if self.cfg.pca is not None:
            self.enable_pca()

        dataset = HSDataset(
                self.filepath_test, 
                preprocess=self.preprocess,
                transform=self.transform,
                cfg=self.cfg)
        
        img,_ = dataset[0]
        self.img_shape = img.shape[2:]
        self.n_channels = img.shape[1]

    # def setup(self, stage: Optional[str] = None):

    #     dataset = HSDataset(
    #             self.filepath_train, 
    #             preprocess=self.preprocess,
    #             transform=self.transform,
    #             cfg=self.cfg)

    #     self.dataset_test = HSDataset(
    #             self.filepath_test, 
    #             preprocess=self.preprocess,
    #             transform=self.transform,
    #             cfg=self.cfg)

    #     # Train-test-split is 80/20. From those 80% we use 10% for validation and 90% for training
    #     train_size = round(0.9 * len(dataset))
    #     val_size = len(dataset) - train_size

    #     if self.cfg.manual_seed is not None:
    #         self.dataset_train, self.dataset_val = random_split(
    #                 dataset, 
    #                 [train_size, val_size], 
    #                 generator=torch.Generator().manual_seed(self.cfg.manual_seed))
    #     else :
    #         self.dataset_train, self.dataset_val = random_split(
    #                 dataset, 
    #                 [train_size, val_size])

    #     # calculate data statistics for normalization
    #     if self.cfg.normalize:
    #         self.enable_normalization()

    def enable_pca(self):
        # train
        outpath_train = self.pca_out_dir.joinpath(f'hcv2_train_pca{self.cfg.pca}.h5')
        apply_pca(  self.cfg.pca, self.filepath_train, outpath_train, 
                    debug=self.debug, half_precision=self.half_precision)
        self.filepath_train = outpath_train

        # test
        outpath_test = self.pca_out_dir.joinpath(f'hcv2_test_pca{self.cfg.pca}.h5')
        apply_pca(  self.cfg.pca, self.filepath_test, outpath_test, 
                    debug=self.debug, half_precision=self.half_precision)
        self.filepath_test = outpath_test

        # val 
        outpath_val = self.pca_out_dir.joinpath(f'hcv2_val_pca{self.cfg.pca}.h5')
        apply_pca(  self.cfg.pca, self.filepath_val, outpath_val, 
                    debug=self.cfg.debug, half_precision=False)
        self.filepath_val = outpath_val


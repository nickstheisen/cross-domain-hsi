#!/usr/bin/env python

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from torchvision import transforms


from crossdomainhsi.datasets.analysis.tools import StatCalculator
from crossdomainhsi.datasets.transforms import ToTensor, ReplaceLabels, Normalize, SpectralAverage, InsertEmptyChannelDim, PermuteData
from .hsdataset import HSDataModule, HSDataset
from crossdomainhsi.datasets.utils import apply_pca

from typing import List, Any, Optional
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from imageio import imread

def label_histogram(dataset, n_classes):
    label_hist = torch.zeros(n_classes) # do not count 'unefined'(highest class_id)
    for i, (_, labels) in enumerate(DataLoader(dataset)):
        label_ids, counts = labels.unique(return_counts=True)
        for i in range(len(label_ids)):
            label_id = label_ids[i]
            if not (label_id == n_classes):
                label_hist[label_id] += counts[i]
    return label_hist


class HSIDrive(HSDataModule):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        self.n_classes = 10 if self.cfg.ignore_water else 11
        self.undef_idx = 9 if self.cfg.ignore_water else 10

        self.filepath_train = self.basepath.joinpath('hsidrive_train.h5')
        self.filepath_test = self.basepath.joinpath('hsidrive_test.h5')
        self.filepath_val = self.basepath.joinpath('hsidrive_val.h5')

        if cfg.ignore_water:
            self.preprocess = transforms.Compose([
                            ToTensor(),
                            ReplaceLabels({0:9, 1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:9, 9:7, 10:8}), # replace undefined 0 and water labels 8 with 9 and then shift labels according
                        ])
        else :
            self.preprocess = transforms.Compose([
                            ToTensor(),
                            ReplaceLabels({0:10, 1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:9}), # replace undefined label 0 with 10 and then shift labels by one
                        ])

        self.transform = None

        if self.cfg.spectral_average:
            self.transform = transforms.Compose([
                                self.transform,
                                SpectralAverage()
                             ])

        if self.cfg.pca is not None:
            self.enable_pca()
        
        # read dimensions from image
        dataset = HSDataset(self.filepath_val, 
                            preprocess=self.preprocess, 
                            transform=self.transform,
                            cfg=self.cfg)
        img, _ = dataset[1]
        self.img_shape = img.shape[2:]
        self.n_channels = img.shape[1]
       
    def enable_pca(self):
        # train
        outpath_train = self.pca_out_dir.joinpath(f'hsidrive_train_pca{self.cfg.pca}.h5')
        apply_pca(  self.cfg.pca, self.filepath_train, outpath_train, 
                    debug=self.debug, half_precision=False)
        self.filepath_train = outpath_train

        # test
        outpath_test = self.pca_out_dir.joinpath(f'hsidrive_test_pca{self.cfg.pca}.h5')
        apply_pca(  self.cfg.pca, self.filepath_test, outpath_test, 
                    debug=self.debug, half_precision=False)
        self.filepath_test = outpath_test

        # val 
        outpath_val = self.pca_out_dir.joinpath(f'hsidrive_val_pca{self.cfg.pca}.h5')
        apply_pca(  self.cfg.pca, self.filepath_val, outpath_val, 
                    debug=self.debug, half_precision=False)
        self.filepath_val = outpath_val

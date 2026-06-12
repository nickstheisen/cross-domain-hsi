#!/usr/bin/env python

from .hsdataset import HSDataModule, HSDataset
from torchvision import transforms
from crossdomainhsi.datasets.transforms import ToTensor, PermuteData, ReplaceLabels, SpectralAverage
from crossdomainhsi.datasets.utils import apply_pca

from typing import List, Any, Optional

class HyKo2(HSDataModule):
    def __init__(self,  cfg):
        super().__init__(cfg)

        self.preprocess = transforms.Compose([
            ToTensor()
        ])

        self.transform = None

        self.n_classes = 11
        self.undef_idx=0
        self.filepath_train = self.basepath.joinpath('hyko2_train.h5')
        self.filepath_test = self.basepath.joinpath('hyko2_test.h5')
        self.filepath_val = self.basepath.joinpath('hyko2_val.h5')

        if self.cfg.spectral_average:
            self.transform = transforms.Compose([
                self.transform,
                SpectralAverage()
            ])
        
        if self.cfg.pca is not None:
            self.enable_pca()

        dataset = HSDataset(self.filepath_val, 
                            preprocess=self.preprocess, 
                            transform=self.transform,
                            cfg=self.cfg)
        img,_ = dataset[0]
        self.img_shape = img.shape[2:]
        self.n_channels = img.shape[1]


    def enable_pca(self):
        # train
        outpath_train = self.pca_out_dir.joinpath(f'hyko2_train_pca{self.cfg.pca}.h5')
        apply_pca(  self.cfg.pca, self.filepath_train, outpath_train, 
                    debug=self.cfg.debug, half_precision=False)
        self.filepath_train = outpath_train

        # test
        outpath_test = self.pca_out_dir.joinpath(f'hyko2_test_pca{self.cfg.pca}.h5')
        apply_pca(  self.cfg.pca, self.filepath_test, outpath_test, 
                    debug=self.cfg.debug, half_precision=False)
        self.filepath_test = outpath_test

        # val 
        outpath_val = self.pca_out_dir.joinpath(f'hyko2_val_pca{self.cfg.pca}.h5')
        apply_pca(  self.cfg.pca, self.filepath_val, outpath_val, 
                    debug=self.cfg.debug, half_precision=False)
        self.filepath_val = outpath_val

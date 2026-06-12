#!/usr/bin/env python
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from torchvision import transforms

import h5py
from omegaconf import OmegaConf

from typing import List, Any, Optional
from pathlib import Path
import time

import numpy as np
from random import shuffle
from crossdomainhsi.datasets.analysis.tools import StatCalculator
from crossdomainhsi.datasets.transforms import Normalize
from crossdomainhsi.datasets.utils import label_histogram

def patch_collate(batch):
    images, labels = list(zip(*batch))
    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)
    return (images, labels)

class HSDataModule(pl.LightningDataModule):

    def __init__(
            self,
            cfg: OmegaConf
    ):
        super().__init__()
        
        self.cfg = cfg
        
        self.basepath = Path(cfg.basepath)
        # data preprocessing
        self.pca_out_dir=Path(cfg.pca_out_dir)
        if not self.pca_out_dir.exists():
            self.pca_out_dir.mkdir(parents=True, exist_ok=True)
        if not self.pca_out_dir.is_dir():
            raise RuntimeError("`pca_out_dir` must be a directory!")
      
    def setup(self, stage: Optional[str] = None):
        self.dataset_train = HSDataset(  self.filepath_train, 
                                    preprocess=self.preprocess,
                                    transform=self.transform,
                                    cfg=self.cfg)
        self.dataset_test = HSDataset(   self.filepath_test,
                                    preprocess=self.preprocess,
                                    transform=self.transform,
                                    cfg=self.cfg)
        self.dataset_val = HSDataset(    self.filepath_val,
                                    preprocess=self.preprocess,
                                    transform=self.transform,
                                    cfg=self.cfg)
        # calculate data statistics for normalization
        if self.cfg.normalize:
            self.enable_normalization()


    def enable_normalization(self):
        # enable normalization in whole data set
        self.dataset_train.enable_normalization()
        self.dataset_test.enable_normalization()
        self.dataset_val.enable_normalization()

    def enable_pca(self):
        raise NotImplementedError

    def train_dataloader(self):
        if self.cfg.train_x_percent is not None:
            train_size = round(self.cfg.train_x_percent/100. * len(self.dataset_train))
            if train_size == 0:
                raise RuntimeError(f"Train data has 0 samples. Please set "
                "`dataset.train_x_percent` higher.")
            rest = len(self.dataset_train) - train_size

            if self.cfg.manual_seed is not None:
                self.dataset_train, _ = random_split(
                        self.dataset_train, 
                        [train_size, rest], 
                        generator=torch.Generator().manual_seed(self.cfg.manual_seed))
            else :
                self.dataset_train, _ = random_split(
                        dataset_train, 
                        [train_size, rest])


        return DataLoader(self.dataset_train,
                batch_size=self.cfg.batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=self.cfg.num_workers,
                persistent_workers=True,
                drop_last=self.cfg.drop_last,
                collate_fn=patch_collate)

    def val_dataloader(self):
        return DataLoader(self.dataset_val,
                batch_size=self.cfg.batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=self.cfg.num_workers,
                collate_fn=patch_collate)

    def test_dataloader(self):
        return DataLoader(self.dataset_test,
                batch_size=self.cfg.batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=self.cfg.num_workers,
                collate_fn=patch_collate)

class HSDataset(Dataset):

    def __init__(self, filepath, preprocess, transform, cfg):
        self._filepath = filepath
        self.patch_size = cfg.patch_size
        self.stride = cfg.stride
        # starting from last dim -> (pad_l, pad_r, pad_t, pad_b)
        # TODO we need to make sure that data loaded is (C, H, W)
        self.pad = cfg.pad

        # if h5file is kept open, the object cannot be pickled and in turn 
        # multi-gpu cannot be used
        #t1 = time.time()
        h5file = h5py.File(self._filepath, 'r', libver='latest')
        self._samplelist = list(h5file.keys())
        self.max_val = None
        self.min_val = None
        if 'metadata' in self._samplelist: 
            self._samplelist.remove('metadata')
            self.min_val = h5file['metadata']['min-val'][()]
            self.max_val = h5file['metadata']['max-val'][()]
        self._preprocess = preprocess
        self._transform = transform
        self._normalize = None

        self._n_samples = len(self._samplelist)
        h5file.close()

    def __len__(self):
        return self._n_samples

    def enable_normalization(self):
        assert (self.min_val is not None) and (self.max_val is not None)
        self._normalize = Normalize(self.min_val, self.max_val)

    def __getitem__(self, idx):
        h5file = h5py.File(self._filepath)
        data = np.array(h5file[self._samplelist[idx]]['image'])
        labels = np.array(h5file[self._samplelist[idx]]['labels'])

        if self._preprocess:
            data, labels = self._preprocess((data, labels))
            
        data_shape = data.shape
        labels_shape = labels.shape

        if self.patch_size is not None:
            # use padding
            if self.pad is not None:
                data = F.pad(data, self.pad)
                labels = F.pad(labels, self.pad)

            # sample patches
            data = data.unfold(1, self.patch_size, 
                               self.stride).unfold(2, self.patch_size, self.stride)
            labels = labels.unfold(1, self.patch_size, 
                               self.stride).unfold(2, self.patch_size, self.stride)
            
            # reshape to (N, C, H, W)
            data = data.contiguous().view(data_shape[0], -1, self.patch_size, 
                self.patch_size).permute((1,0,2,3))
            labels = labels.contiguous().view(labels_shape[0], -1, self.patch_size, 
                self.patch_size).permute((1,0,2,3))
        else :
            # add empty batch dimension
            data = data.unsqueeze(dim=0)
            labels = labels.unsqueeze(dim=0)
        
        sample = (data, labels)
        if self._transform:
            sample = self._transform(sample)
        if self._normalize:
            sample = self._normalize(sample)
        
        h5file.close()
        return sample

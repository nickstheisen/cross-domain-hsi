#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .semsegmodule import SemanticSegmentationModule
from typing import List

class LiuBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size_channel, batch_norm):
        super().__init__()
        if batch_norm:
            self.conv3d = nn.Sequential(
                            nn.Conv3d(in_channels, out_channels, 
                                kernel_size = (kernel_size_channel, 1, 1), 
                                padding = 'same'
                            ),
                            nn.BatchNorm3d(out_channels)
            )
                            

        else:
            self.conv3d = nn.Conv3d(in_channels, out_channels, 
                            kernel_size = (kernel_size_channel, 1, 1), 
                            padding = 'same'
            )
        self.pool3d = nn.MaxPool3d((2,1,1))
    
    def forward(self, x):
        x = self.conv3d(x)
        
        x = F.relu(x)
        x = self.pool3d(x)
        return x

class LiuBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size_channel, batch_norm):
        super().__init__()
        if batch_norm:
            self.conv = nn.Sequential(
                            nn.Conv1d(in_channels, out_channels,
                                kernel_size = kernel_size_channel,
                                padding='same'
                            ),
                            nn.BatchNorm1d(out_channels)
            )
                            

        else:
            self.conv = nn.Conv1d(in_channels, out_channels,
                                kernel_size = kernel_size_channel,
                                padding='same'
            )
        self.pool = nn.MaxPool1d(kernel_size=2)
    
    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = self.pool(x)
        return x


class LiuNet2D(SemanticSegmentationModule):
    def __init__(self,
            kernel_size : int = 3,
            n_kernels : List[int] = [32, 32, 64, 64],
            batch_norm : bool = False,
            **kwargs):
        super(LiuNet2D, self).__init__(**kwargs)

        self.save_hyperparameters()

        self.kernel_size = kernel_size
        self.n_kernels = n_kernels
        self.batch_norm = batch_norm

        self.input_layer = LiuBlock2D(1, self.n_kernels[0], self.kernel_size, self.batch_norm)
        convs = []
        for i in range(1,len(n_kernels)):
            convs.append(LiuBlock2D(self.n_kernels[i-1], self.n_kernels[i], self.kernel_size, 
                            self.batch_norm)
            )
        self.feature_extraction = nn.Sequential(*convs)
        
        in_channels_outlayer = self.n_kernels[-1] * (self.n_channels // (2**len(self.n_kernels)))
        self.output_layer = nn.Conv2d(in_channels_outlayer, self.n_classes, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.input_layer(x)
        x = self.feature_extraction(x)

        # flatten along feature dimension
        x = torch.flatten(x, start_dim=1, end_dim=2)

        # fully convolutional
        x = self.output_layer(x)

        return x

class LiuNet(SemanticSegmentationModule):
    def __init__(self, cfg, **kwargs):
        super(LiuNet, self).__init__(cfg, **kwargs)

        self.save_hyperparameters()

        self.input_layer = LiuBlock(1, self.cfg.n_kernels[0], 
            self.cfg.kernel_size, self.cfg.batch_norm)
        convs = []
        for i in range(1,len(self.cfg.n_kernels)):
            convs.append(LiuBlock(self.cfg.n_kernels[i-1], self.cfg.n_kernels[i], 
                self.cfg.kernel_size, self.cfg.batch_norm)
            )
        self.feature_extraction = nn.Sequential(*convs)
        
        in_channels_outlayer = self.cfg.n_kernels[-1] * (self.cfg.n_channels // 
                                                            (2**len(self.cfg.n_kernels)))
        self.output_layer = nn.Linear(in_channels_outlayer, self.cfg.n_classes)

    def forward(self, x):
        x = x.squeeze(dim=[2,3])
        x = x.unsqueeze(1)
        x = self.input_layer(x)
        x = self.feature_extraction(x)

        # flatten along feature dimension
        x = torch.flatten(x, start_dim=1)

        # fully convolutional
        x = self.output_layer(x)

        # restore original shape
        x = x.unsqueeze(2).unsqueeze(3)

        return x



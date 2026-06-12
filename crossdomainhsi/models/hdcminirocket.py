#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .semsegmodule import SemanticSegmentationModule

from .minirocket.minirocket_torch import Minirocket_Encoder
import time

class MiniRocketSeg(SemanticSegmentationModule):
    def __init__(self, cfg, **kwargs):
        super(MiniRocketSeg, self).__init__(cfg, **kwargs)

        self.config = {
            "scale" : self.cfg.scale,
            "n_steps" : self.cfg.n_channels,
            "HDC_dim" : self.cfg.feature_dim,
            "seed" : 42,
        }

        self.save_hyperparameters()
        self.poses_computed = False
        self.biases_set = False

        self.minirocket_enc = Minirocket_Encoder(
            dim=self.cfg.feature_dim,
            n_channels=1, # unimodal (not channels in the sense of spectral channels)
            seq_len=self.cfg.n_channels,
            use_hdc=self.cfg.use_hdc,
            seed=42
            )

        if self.cfg.features_only:
            self.c_head = nn.Sequential(
                    nn.Identity()
            )
        else:
            self.c_head = nn.Sequential(
                nn.Flatten(start_dim=1, end_dim=-1),
                nn.BatchNorm1d(self.cfg.feature_dim),
                nn.Linear(self.cfg.feature_dim, self.cfg.n_classes),
            )
            
    def forward(self, x):
        x = x.squeeze(dim=[2,3])
        x = x.unsqueeze(1)

        if self.cfg.use_hdc and not self.poses_computed:
            # compute hdc positional encodings
            self.minirocket_enc.encoder.compute_poses(config=self.config)
            self.poses_computed = True

        if self.training:
            if not self.biases_set:
                self.minirocket_enc.encoder.fit(x)
                self.biases_set = True
        x = self.minirocket_enc(x)
        logits = self.c_head(x)

        # restore original shape
        logits = logits.unsqueeze(dim=2).unsqueeze(dim=3)
        return logits

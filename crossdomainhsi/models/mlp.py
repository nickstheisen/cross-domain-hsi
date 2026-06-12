#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .semsegmodule import SemanticSegmentationModule


class MLP(SemanticSegmentationModule):
    def __init__(self, cfg, **kwargs):
        super(MLP, self).__init__(cfg, **kwargs)

        self.save_hyperparameters()
        layer_nodes = self.cfg.hidden_nodes
        layer_nodes.insert( 0, self.cfg.n_channels) # input
        layer_nodes.append(self.cfg.n_classes) # output

        self.mlp = nn.Sequential()
        
        n_layers = len(layer_nodes) - 1
        for i in range(0, n_layers):
            self.mlp.add_module(f"linear{i}", nn.Linear(layer_nodes[i], layer_nodes[i+1]))
            if i < (n_layers - 1):
                self.mlp.add_module(f"relu{i}", nn.ReLU())

    def forward(self, x):
        x = x.squeeze(dim=[2,3])

        logits = self.mlp(x)

        # restore original shape
        logits = logits.unsqueeze(dim=2).unsqueeze(dim=3)
        return logits

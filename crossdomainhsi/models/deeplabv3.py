#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .semsegmodule import SemanticSegmentationModule
import crossdomainhsi.models.network as network

class DeeplabV3Plus(SemanticSegmentationModule):
    def __init__(self, cfg, **kwargs):
#            pretrained_weights:str = None,
#            **kwargs):
        super(DeeplabV3Plus, self).__init__(cfg,**kwargs)

        self.save_hyperparameters()
        self.pt_weights = self.cfg.pretrained_weights
        self.model = network.modeling.__dict__[f'deeplabv3plus_{self.cfg.backbone}'](
                        num_classes = self.cfg.n_classes,
                        output_stride=16,
                        pretrained_backbone=self.cfg.pretrained_backbone,
                        num_channels = self.cfg.n_channels)
        
        if not (self.pt_weights is None):
            model_state = torch.load( self.pt_weights )['model_state']
            ## replace output layer to match n_classes
            weight_shape = model_state['classifier.classifier.3.weight'].shape
            bias_shape = model_state['classifier.classifier.3.bias'].shape
            # update classifier weight shapes
            weight_shape = torch.Size([self.cfg.n_classes, weight_shape[1], 
                                            weight_shape[2], weight_shape[3]])
            bias_shape = torch.Size([self.cfg.n_classes])
            # insert random values
            model_state['classifier.classifier.3.weight'] = torch.rand(weight_shape)
            model_state['classifier.classifier.3.bias'] = torch.rand(bias_shape)

            self.model.load_state_dict( model_state )
            
            # freeze everything except classifier
            for param in self.model.parameters():
                param.requires_grad = False
            
            for param in self.model.classifier.parameters():
                param.requires_grad = True

    def forward(self, x):
        print(x.shape)
        x = self.model(x)
        return x


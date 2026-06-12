from collections import OrderedDict

import torch
import torch.nn as nn
from .semsegmodule import SemanticSegmentationModule
from .HyperSL.engine.model import SpectralSharedEncoder

from crossdomainhsi.datasets.utils import read_wavelengths_file

class HyperSLBackbone(nn.Module):
    def __init__(self, model_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if model_size == 'small':
            self.embedding_dim = 256
            self.spectral_encoder =SpectralSharedEncoder(
                embedding_dim = 256,
                encoder_depth=8,
                decoder_depth=4,
                num_heads=8,
            )
        if model_size == 'base':
            self.embedding_dim = 512
            self.spectral_encoder = SpectralSharedEncoder(
                embedding_dim=512,
                max_band=500,
                encoder_depth=24,
                decoder_depth=12,
                num_heads=16,
                mlp_ratio=4.,
                norm_layer=nn.LayerNorm
            )
        if model_size == 'large':
            self.embedding_dim = 1024
            self.spectral_encoder = SpectralSharedEncoder(
                embedding_dim=1024,
                max_band=500,
                encoder_depth=32,
                decoder_depth=16,
                num_heads=32,
                mlp_ratio=4.,
                norm_layer=nn.LayerNorm
            )

        if model_size == 'huge':
            self.embedding_dim = 2048
            self.spectral_encoder = SpectralSharedEncoder(
                embedding_dim=2048,
                max_band=500,
                encoder_depth=48,
                decoder_depth=24,
                num_heads=32,
                mlp_ratio=4.,
                norm_layer=nn.LayerNorm
            )

    def forward(self,x,wavelength):
        x,_,_,_,_,shape = self.spectral_encoder.encoder_forward(x,wavelength,0.)
        B, W, H, C = shape
        x = x.view(B,H,W,-1).permute(0,-1,1,2)
        return x

class HyperSLClassificationHead(nn.Module):
    def __init__(self, class_num , embedding_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding_dim = embedding_dim

        self.convblock = nn.Sequential(
            nn.Conv2d(self.embedding_dim,64,1,1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 512, 3, 2,1), # h,w /2
            nn.BatchNorm2d(512),
            nn.GELU(),
            # nn.Conv2d(128, 256, 2, 2,1), # h,w /4
            # nn.BatchNorm2d(256),
            # nn.GELU(),
            # nn.Conv2d(256, 512, 2, 2, 1), # h,w /8
            # nn.BatchNorm2d(512),
            # nn.GELU(),
            ## === Exp 1 ==
            # nn.Conv2d(self.embedding_dim, 64, 1, 1),
            # nn.BatchNorm2d(64),
            # nn.GELU(),
            # nn.Conv2d(64, 128, 3, 2, 1),
            # nn.BatchNorm2d(128),
            # nn.GELU(),
            # nn.Conv2d(128, 256, 2, 2, 1),
            # nn.BatchNorm2d(256),
            # nn.GELU(),
            ## == Exp 2 ==
            # nn.BatchNorm2d(self.embedding_dim),
            # nn.GELU(),
            # == Exp 3 ==
            # nn.Conv2d(self.embedding_dim, 64, 1, 1),
            # nn.BatchNorm2d(64),
            # nn.GELU(),
            )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(512,1024),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.Linear(1024,1024),
            nn.Dropout(0.4),
            nn.GELU(),
            nn.Linear(1024, class_num),
            nn.GELU()
            ## == Exp 1 ==
            # nn.Linear(256, 128),
            # nn.Dropout(0.2),
            # nn.GELU(),
            # nn.Linear(128, class_num),
            # nn.GELU()
            ## == Exp 2 ==
            # nn.Linear(self.embedding_dim, 128),
            # nn.Dropout(0.4),
            # nn.GELU(),
            # nn.Linear(128, class_num),
            # nn.GELU(),
            ## == Exp 3 ==
            # nn.Linear(64, class_num),
            # nn.GELU(),
        )

    def forward(self,x):
        B, C, H, W = x.shape
        x = self.convblock(x)
        x = self.pool(x).view(B,-1)
        x = self.classifier(x)
        return x

class MLPClassificationHead(nn.Module):
    def __init__(self, class_num , embedding_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding_dim = embedding_dim
        self.pool = nn.AdaptiveAvgPool2d(1) # use or not?
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.embedding_dim),
            nn.Linear(self.embedding_dim, class_num)
        )

    def forward(self,x):
        B, C, H, W = x.shape
        x = self.pool(x).view(B,-1)
        x = self.classifier(x)
        return x

class HyperSLClassification(SemanticSegmentationModule):
    def __init__(self, cfg, **kwargs):
        super(HyperSLClassification, self).__init__(cfg, **kwargs)

        self.save_hyperparameters()

        self.wavelengths_file = self.cfg.wavelengths_file
        if self.wavelengths_file is not None:
            self.wavelengths = torch.from_numpy(read_wavelengths_file(self.wavelengths_file))
        else:
            raise RuntimeError('No wavelengths file provided')

        if self.cfg.use_backbone:
            self.backbone = HyperSLBackbone(model_size=self.cfg.bb_model_size)
            self.embedding_dim = self.backbone.embedding_dim
            if self.cfg.bb_weights_file is not None:
                ckpt = torch.load(self.cfg.bb_weights_file)
                weights = OrderedDict()
                for k, v in ckpt['model'].items():
                    name = k[7:]
                    weights[name] = v
                self.backbone.spectral_encoder.load_state_dict(weights)
                if self.cfg.freeze_backbone:
                    for param in self.backbone.spectral_encoder.parameters():
                        param.requires_grad = False
        else:
            self.backbone = None
            self.embedding_dim = self.cfg.n_channels

        if self.cfg.mlp_classifier:
            self.classifier = MLPClassificationHead(class_num=self.cfg.n_classes,
                                                    embedding_dim=self.embedding_dim)
        else:
            self.classifier = HyperSLClassificationHead(class_num=self.cfg.n_classes,
                                                    embedding_dim=self.embedding_dim)

        # self.model = ClassificationModel(
        #     class_num=self.cfg.n_classes,
        #     model_size=self.cfg.bb_model_size,
        # )

    def forward(self, x):
        if self.wavelengths.device != self.device:
            self.wavelengths = self.wavelengths.to(self.device)
        n_samples = x.shape[0]
        w = self.wavelengths.repeat(n_samples, 1)
        if self.backbone is not None:
            x = x.permute(0, 2, 3, 1)
            x = self.backbone(x, w)
        logits = self.classifier(x)
        # #x = x[:, None, None, :] # insert empty dimensions
        # logits = self.model.forward(x, w)
        logits = logits.unsqueeze(2).unsqueeze(3)
        return logits

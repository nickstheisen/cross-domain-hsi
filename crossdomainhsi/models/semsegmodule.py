#!/usr/bin/env python

import torch
from torch import nn
import torchmetrics

import torchvision.transforms as T

import pytorch_lightning as pl
from pytorch_lightning.utilities import grad_norm

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
plt.switch_backend('agg')

from typing import Optional

import kornia as K
from kornia.augmentation import AugmentationSequential, RandomHorizontalFlip

def inv_num_of_samples(histogram):
    return histogram.sum()/histogram

def inv_square_num_of_samples(histogram):
    return histogram.sum()/torch.sqrt(histogram)

class DataAugmentation(nn.Module):
    def __init__(self, 
                p_hflip: float,
                ):
        super().__init__()
        # if p(hflip) is basically 0, don't do it
        self.apply_hflip = True if p_hflip > 1e-5 else False

        self.augment = AugmentationSequential(
                                RandomHorizontalFlip(p=p_hflip),
                                data_keys=['input', 'mask'])
    
    @torch.no_grad() # disable gradients for efficiency
    def forward(self, inputs, labels):
        if self.apply_hflip:
            labels = torch.unsqueeze(labels, dim=1)
            inputs, labels = self.augment(inputs, labels.float())
            labels = labels.squeeze(dim=1).long()
        return inputs, labels

class SemanticSegmentationModule(pl.LightningModule):
    
    def __init__(self, cfg, **kwargs):
        super(SemanticSegmentationModule, self).__init__(**kwargs)

        self.cfg = cfg.model

        if self.cfg.sub_batch_size is not None:
            self.automatic_optimization = False
        
        # class definitions
        self.label_names, self.label_colors = self._load_label_def(self.cfg.label_def)


        # Augmentation
        self.augment = DataAugmentation(p_hflip=self.cfg.da_hflip)

        # logging
        self.log_grad_norm = self.cfg.log_grad_norm
        self.rich_train_log = self.cfg.rich_train_log

        ## Attention! Unfortunately, metrics must be created as members instead of directly storing
        ## them in dictionaries otherwise they are not identified as child modules
        ## and in turn not moved to the correct device
        ## TODO This is only a workaround, I should find a better solution in the future

        ## train metrics
        self.train_metrics = {}
        self.acc_train_micro = torchmetrics.Accuracy(
                task=self.cfg.classification_task,
                num_classes=self.cfg.n_classes, 
                ignore_index=self.cfg.ignore_index, 
                average='micro')
        self.train_metrics["Train/accuracy-micro"] = self.acc_train_micro
        
        if self.cfg.rich_train_log:
            self.acc_train_macro = torchmetrics.Accuracy(
                    task=self.cfg.classification_task,
                    num_classes=self.cfg.n_classes,
                    ignore_index=self.cfg.ignore_index, 
                    average='macro')
            self.train_metrics["Train/accuracy-macro"] = self.acc_train_macro
            self.acc_train_class = torchmetrics.Accuracy(
                    task=self.cfg.classification_task,
                    num_classes=self.cfg.n_classes,
                    ignore_index=self.cfg.ignore_index, 
                    average='none')
            self.train_metrics["Train/accuracy-class"] = self.acc_train_class
            self.f1_train_macro = torchmetrics.F1Score(
                    task=self.cfg.classification_task,
                    num_classes=self.cfg.n_classes,
                    ignore_index=self.cfg.ignore_index, 
                    average='macro')
            self.train_metrics["Train/f1-macro"] = self.f1_train_macro
            self.f1_train_class = torchmetrics.F1Score(
                    task=self.cfg.classification_task,
                     num_classes=self.cfg.n_classes,
                    ignore_index=self.cfg.ignore_index, 
                    average='none')
            self.train_metrics["Train/f1-class"] = self.f1_train_class
            self.jaccard_train = torchmetrics.JaccardIndex(
                    task=self.cfg.classification_task,
                    ignore_index=self.cfg.ignore_index, 
                    num_classes=self.cfg.n_classes)
            self.train_metrics["Train/jaccard"] = self.jaccard_train
            self.jaccard_train_class = torchmetrics.JaccardIndex(
                    task=self.cfg.classification_task,
                    average='none',
                    ignore_index=self.cfg.ignore_index,
                    num_classes=self.cfg.n_classes)
            self.train_metrics["Train/jaccard-class"] = self.jaccard_train_class
            self.confmat_train = torchmetrics.ConfusionMatrix(
                    task=self.cfg.classification_task,
                    num_classes=self.cfg.n_classes,
                    ignore_index=self.cfg.ignore_index)
            self.train_metrics["Train/conf_mat"] = self.confmat_train

        ## val metrics
        self.val_metrics = {}
        self.acc_val_micro = torchmetrics.Accuracy(
                task=self.cfg.classification_task,
                num_classes=self.cfg.n_classes,
                ignore_index=self.cfg.ignore_index, 
                average='micro')
        self.val_metrics["Validation/accuracy-micro"] = self.acc_val_micro
        self.acc_val_macro = torchmetrics.Accuracy(
                task=self.cfg.classification_task,
                num_classes=self.cfg.n_classes,
                ignore_index=self.cfg.ignore_index, 
                average='macro')
        self.val_metrics["Validation/accuracy-macro"] = self.acc_val_macro
        self.acc_val_class = torchmetrics.Accuracy(
                task=self.cfg.classification_task,
                num_classes=self.cfg.n_classes,
                ignore_index=self.cfg.ignore_index, 
                average='none')
        self.val_metrics["Validation/accuracy-class"] = self.acc_val_class
        self.f1_val_macro = torchmetrics.F1Score(
                task=self.cfg.classification_task,
                num_classes=self.cfg.n_classes,
                ignore_index=self.cfg.ignore_index, 
                average='macro')
        self.val_metrics["Validation/f1-macro"] = self.f1_val_macro
        self.f1_val_class = torchmetrics.F1Score(
                task=self.cfg.classification_task,
                num_classes=self.cfg.n_classes,
                ignore_index=self.cfg.ignore_index, 
                average='none')
        self.val_metrics["Validation/f1-class"] = self.f1_val_class
        self.jaccard_val = torchmetrics.JaccardIndex(
                task=self.cfg.classification_task,
                ignore_index=self.cfg.ignore_index, 
                num_classes=self.cfg.n_classes)
        self.val_metrics["Validation/jaccard"] = self.jaccard_val
        self.jaccard_val_class = torchmetrics.JaccardIndex(
                task=self.cfg.classification_task,
                average='none',
                ignore_index=self.cfg.ignore_index,
                num_classes=self.cfg.n_classes)
        self.val_metrics["Validation/jaccard-class"] = self.jaccard_val_class
        self.confmat_val = torchmetrics.ConfusionMatrix(
                task=self.cfg.classification_task,
                num_classes=self.cfg.n_classes,
                ignore_index=self.cfg.ignore_index)
        self.val_metrics["Validation/conf_mat"] = self.confmat_val

        ## test metrics
        self.test_metrics = {}
        self.acc_test_micro = torchmetrics.Accuracy(
                task=self.cfg.classification_task,
                num_classes=self.cfg.n_classes,
                ignore_index=self.cfg.ignore_index, 
                average='micro')
        self.test_metrics["Test/accuracy-micro"] = self.acc_test_micro
        self.acc_test_macro = torchmetrics.Accuracy(
                task=self.cfg.classification_task,
                num_classes=self.cfg.n_classes,
                ignore_index=self.cfg.ignore_index, 
                average='macro')
        self.test_metrics["Test/accuracy-macro"] = self.acc_test_macro
        self.acc_test_class = torchmetrics.Accuracy(
                task=self.cfg.classification_task,
                num_classes=self.cfg.n_classes,
                ignore_index=self.cfg.ignore_index, 
                average='none')
        self.test_metrics["Test/accuracy-class"] = self.acc_test_class
        self.f1_test_macro = torchmetrics.F1Score(
                task=self.cfg.classification_task,
                num_classes=self.cfg.n_classes,
                ignore_index=self.cfg.ignore_index, 
                average='macro')
        self.test_metrics["Test/f1-macro"] = self.f1_test_macro
        self.f1_test_class = torchmetrics.F1Score(
                task=self.cfg.classification_task,
                num_classes=self.cfg.n_classes,
                ignore_index=self.cfg.ignore_index, 
                average='none')
        self.test_metrics["Test/f1-class"] = self.f1_test_class
        self.jaccard_test = torchmetrics.JaccardIndex(
                task=self.cfg.classification_task,
                ignore_index=self.cfg.ignore_index, 
                num_classes=self.cfg.n_classes)
        self.test_metrics["Test/jaccard"] = self.jaccard_test
        self.jaccard_test_class = torchmetrics.JaccardIndex(
                task=self.cfg.classification_task,
                average='none',
                ignore_index=self.cfg.ignore_index,
                num_classes=self.cfg.n_classes)
        self.test_metrics["Test/jaccard-class"] = self.jaccard_test_class
        self.confmat_test = torchmetrics.ConfusionMatrix(
                task=self.cfg.classification_task,
                num_classes=self.cfg.n_classes,
                ignore_index=self.cfg.ignore_index)
        self.test_metrics["Test/conf_mat"] = self.confmat_test

    def setup(self, stage: Optional[str]=None):
        if self.cfg.class_weighting is not None:
            train_c_hist, _, _ = self.trainer.datamodule.class_histograms() 
            if self.cfg.class_weighting == 'INS':
                self.class_weights = inv_num_of_samples(train_c_hist)
            elif self.cfg.class_weighting == 'ISNS':
                self.class_weights = inv_square_num_of_samples(train_c_hist)
            else :
                raise RuntimeError(f'Class weighting strategy "{self.cfg.class_weighting}" does'
                        ' not exist or is not implemented yet!')
        else:
            self.class_weights = torch.ones(self.cfg.n_classes)
        
        if self.cfg.loss_name == 'cross_entropy':
            ignore_index = -100 if self.cfg.ignore_index is None else self.cfg.ignore_index
            self.criterion = nn.CrossEntropyLoss(
                    ignore_index=ignore_index,
                    weight=self.class_weights)
                    #reduction='none')
        else : 
            raise RuntimeError(f'Loss function "{self.cfg.loss_name}" is not available '
                    'or is not implemented yet')

        # logging
        if self.cfg.export_metrics:
            # confusion matrices
            self.confmat_log_dir = Path(self.logger.log_dir).joinpath('confmats')
            self.confmat_log_dir.mkdir(parents=True, exist_ok=True)

            # class-wise metrics
            self.classmetric_log_dir = Path(self.logger.log_dir).joinpath('class-metrics')
            self.classmetric_log_dir.mkdir(parents=True, exist_ok=True)
            
            # predictions
            self.pred_export_log_dir = Path(self.logger.log_dir).joinpath('predictions')
            self.pred_export_log_dir.mkdir(parents=True, exist_ok=True)

    def configure_optimizers(self):
        if self.cfg.optimizer_name == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), 
                    lr=self.cfg.learning_rate, 
                    momentum=self.cfg.momentum,
                    weight_decay=self.cfg.weight_decay)
        elif self.cfg.optimizer_name == 'RMSprop':
            optimizer = torch.optim.RMSprop(self.parameters(),
                    lr=self.cfg.learning_rate,
                    momentum=self.cfg.momentum,
                    weight_decay=self.cfg.weight_decay,
                    eps=self.cfg.optimizer_eps)
        elif self.cfg.optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(),
                    lr=self.cfg.learning_rate,
                    weight_decay=self.cfg.weight_decay,
                    eps=self.cfg.optimizer_eps)
        elif self.cfg.optimizer_name == 'AdamW':
            # TODO see if I need to provide ignore_index here?
            optimizer = torch.optim.AdamW(self.parameters(),
                    lr=self.cfg.learning_rate,
                    weight_decay=self.cfg.weight_decay,
                    eps=self.cfg.optimizer_eps)
        else :
            raise RuntimeError(f'Optimizer {self.cfg.optimizer_name} unknown!')
        return optimizer

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        
        if self.cfg.sub_batch_size is None :
            # regular training step
            inputs, labels = self.augment(inputs, labels)
            prediction = self.forward(inputs)
        
            loss = self.criterion(prediction, labels.squeeze(1)) 

            # NOTE I removed this as it does not seem to make a huge difference between two runs
            # if reduction is used in both and keep the code a little simpler
            # built-in reduction='mean' with CrossEntropyLoss is non-deterministic
            # on GPU so we do it manually
            # see: https://discuss.pytorch.org/t/pytorchs-non-deterministic-cross-entropy-loss-and-the-problem-of-reproducibility/172180/8
            #loss = loss.mean()         

            if self.cfg.export_metrics:
                prediction = prediction.argmax(dim=1, keepdims=True)
                for name, metric in self.train_metrics.items():
                    metric(prediction, labels)
        else :
            # iterative training step (useful e.g. when batch does not fit into gpu-memory)
            opt = self.optimizers()
            #opt.zero_grad()
            total_loss = 0
            n_sub_batches = 0
            for i in range(0, inputs.shape[0], self.cfg.sub_batch_size):
                sub_inputs = inputs[i:i+self.cfg.sub_batch_size]
                sub_labels = labels[i:i+self.cfg.sub_batch_size]

                opt.zero_grad()
                
                sub_input, sub_labels = self.augment(sub_inputs, sub_labels)
                prediction = self.forward(sub_inputs)

                loss = self.criterion(prediction, sub_labels.squeeze(1))
                if not loss.isnan():
                    # if using automatic optimization sub_batch_size does not influence
                    # memory consumption on gpu. I think because backprop is only triggered
                    # at the end of each epoch some intermediate results must be kept, resulting
                    # in basically the same memory usage as with regular batches
                    self.manual_backward(loss)
                    opt.step()
                    total_loss += loss
                    n_sub_batches += 1

                if self.cfg.export_metrics:
                    prediction = prediction.argmax(dim=1, keepdims=True)
                    for name, metric in self.train_metrics.items():
                        metric(prediction, sub_labels)
            
            #opt.step()
            loss = total_loss/n_sub_batches
            # TODO make it configurable if updated after batch size or after sub batch size
        self.log('train_loss_step', loss)
        return loss


    def on_before_optimizer_step(self, optimizer):
        if self.log_grad_norm:
            norms = grad_norm(self, norm_type=2)
            self.log_dict(norms)

    def on_train_epoch_end(self):
        if self.cfg.export_metrics:
            for name, metric in self.train_metrics.items():
                if "conf_mat" in name:
                    confmat_epoch = metric.compute()

                    # plot confusion_matrix
                    confmat_epoch = confmat_epoch.detach().cpu().numpy().astype(int)
                    confmat_logpath = self.confmat_log_dir.joinpath(
                            f'confmat_train_epoch{self.current_epoch}.csv')
                    self._export_confmat(confmat_logpath, confmat_epoch)
                    self.logger.experiment.add_figure("Train/confusion-matrix", 
                            self._plot_confmat(confmat_epoch),
                            self.current_epoch)
                    self.logger.experiment.add_figure("Train/log-confusion-matrix",
                            self._plot_log_confmat(confmat_epoch),
                            self.current_epoch)
                elif "class" in name:
                    # actually compute score from logs
                    score_epoch = metric.compute()
                    
                    # detach from graph, retrieve from gpu-memory, convert to numpy array
                    score_epoch = score_epoch.detach().cpu().numpy()

                    # extract metric name (remove 'Validation/','Train/','Test/' and replace spaces)
                    metric_name = name.split('/')[1].replace(' ', '_')

                    # export and plot classwise scores
                    score_logpath = self.classmetric_log_dir.joinpath(
                            f'{metric_name}_train_epoch{self.current_epoch}.csv')
                    self._export_classmetric(score_logpath, score_epoch)
                    self.logger.experiment.add_figure(f"Train/{metric_name}",
                            self._plot_classmetric(score_epoch),
                            self.current_epoch)
                else:
                    self.log(name, metric)

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        if self.cfg.sub_batch_size is None :
            prediction = self.forward(inputs)
            
            if self.cfg.export_metrics:
                prediction = prediction.argmax(dim=1, keepdims=True)
                
                for name, metric in self.val_metrics.items():
                    metric(prediction, labels)
                
                # Visualize prediction of first batch in each epoch
                if self.cfg.plot_batch and batch_idx == 0:
                    preds = prediction.detach().cpu().numpy().astype(int)
                    labels = labels.detach().cpu().numpy().astype(int)
                    self.logger.experiment.add_figure(
                            "Sample-prediction/validation-batch0:",
                            self._plot_batch_prediction(preds,labels),
                            self.current_epoch)
                    self.logger.experiment.add_figure(
                            "Sample-prediction/validation-batch0-undef-masked:",
                            self._plot_batch_prediction(preds, labels, undef_mask=True),
                            self.current_epoch)
        else : 
            # iterative training step (useful e.g. when batch does not fit into gpu-memory)
            for i in range(0, inputs.shape[0], self.cfg.sub_batch_size):
                sub_inputs = inputs[i:i+self.cfg.sub_batch_size]
                sub_labels = labels[i:i+self.cfg.sub_batch_size]
                
                prediction = self.forward(sub_inputs)


                if self.cfg.export_metrics:
                    prediction = prediction.argmax(dim=1, keepdims=True)
                    
                    for name, metric in self.val_metrics.items():
                        metric(prediction, sub_labels)
                    
                    # Visualize prediction of first sub batch in each epoch
                    if self.cfg.plot_batch and batch_idx == 0 and i == 0 :
                        preds = prediction.detach().cpu().numpy().astype(int)
                        sub_labels = sub_labels.detach().cpu().numpy().astype(int)
                        self.logger.experiment.add_figure(
                                "Sample-prediction/validation-batch0:",
                                self._plot_batch_prediction(preds,sub_labels),
                                self.current_epoch)
                        self.logger.experiment.add_figure(
                                "Sample-prediction/validation-batch0-undef-masked:",
                                self._plot_batch_prediction(preds, sub_labels, undef_mask=True),
                                self.current_epoch)
    def predict_step(self, batch, batch_idx):
        inputs, _ = batch
        if self.cfg.sub_batch_size is None :
            prediction = self.forward(inputs)
        else : 
            sub_batch_predictions = []
            # iterative training step (useful e.g. when batch does not fit into gpu-memory)
            for i in range(0, inputs.shape[0], self.cfg.sub_batch_size):
                sub_inputs = inputs[i:i+self.cfg.sub_batch_size]
                
                sub_batch_predictions.append(self.forward(sub_inputs).detach().cpu())
            prediction = torch.vstack(sub_batch_predictions)
        return prediction

    def on_validation_epoch_end(self):
        if self.cfg.export_metrics:
            for name, metric in self.val_metrics.items():
                if "conf_mat" in name:
                    confmat_epoch = metric.compute()

                    # plot confusion_matrix
                    confmat_epoch = confmat_epoch.detach().cpu().numpy().astype(int)
                    confmat_logpath = self.confmat_log_dir.joinpath(
                            f'confmat_val_epoch{self.current_epoch}.csv')
                    self._export_confmat(confmat_logpath, confmat_epoch)
                    self.logger.experiment.add_figure("Validation/confusion-matrix", 
                            self._plot_confmat(confmat_epoch),
                            self.current_epoch)
                    self.logger.experiment.add_figure("Validation/log-confusion-matrix",
                            self._plot_log_confmat(confmat_epoch),
                            self.current_epoch)

                elif "class" in name:
                    # actually compute score from logs
                    score_epoch = metric.compute()
                    
                    # detach from graph, retrieve from gpu-memory, convert to numpy array
                    score_epoch = score_epoch.detach().cpu().numpy()

                    # extract metric name (remove 'Validation/','Train/','Test/' and replace spaces)
                    metric_name = name.split('/')[1].replace(' ', '_')

                    # export and plot classwise scores
                    score_logpath = self.classmetric_log_dir.joinpath(
                            f'{metric_name}_val_epoch{self.current_epoch}.csv')
                    self._export_classmetric(score_logpath, score_epoch)
                    self.logger.experiment.add_figure(f"Validation/{metric_name}",
                            self._plot_classmetric(score_epoch),
                            self.current_epoch)
                else :
                    self.log(name, metric)

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        if self.cfg.sub_batch_size is None:
            prediction = self.forward(inputs)
            loss = self.criterion(prediction, labels.squeeze(1))
            
            if self.cfg.export_metrics:
                prediction = prediction.argmax(dim=1, keepdims=True)

                for _, metric in self.test_metrics.items():
                    metric(prediction, labels)
        else : 
            # iterative training step (useful e.g. when batch does not fit into gpu-memory)
            for i in range(0, inputs.shape[0], self.cfg.sub_batch_size):
                sub_inputs = inputs[i:i+self.cfg.sub_batch_size]
                sub_labels = labels[i:i+self.cfg.sub_batch_size]
                
                prediction = self.forward(sub_inputs)

                if self.cfg.export_metrics:
                    prediction = prediction.argmax(dim=1, keepdims=True)

                    for _, metric in self.test_metrics.items():
                        metric(prediction, sub_labels)

    def on_test_epoch_end(self):
        if self.cfg.export_metrics:
            for name, metric in self.test_metrics.items():
                if "conf_mat" in name:
                    confmat_epoch = metric.compute()

                    # plot confusion_matrix
                    confmat_epoch = confmat_epoch.detach().cpu().numpy().astype(int)
                    confmat_logpath = self.confmat_log_dir.joinpath(
                            f'confmat_val_epoch{self.current_epoch}.csv')
                    self._export_confmat(confmat_logpath, confmat_epoch)
                    self.logger.experiment.add_figure("Test/confusion-matrix", 
                            self._plot_confmat(confmat_epoch),
                            self.current_epoch)
                    self.logger.experiment.add_figure("Test/log-confusion-matrix",
                            self._plot_log_confmat(confmat_epoch),
                            self.current_epoch)

                elif "class" in name:
                    # actually compute score from logs
                    score_epoch = metric.compute()

                    for i in range(score_epoch.size(dim=0)):
                        self.log(f"{name}/{i}", score_epoch[i])
                    
                    # detach from graph, retrieve from gpu-memory, convert to numpy array
                    score_epoch = score_epoch.detach().cpu().numpy()

                    # extract metric name (remove 'Validation/','Train/','Test/' and replace spaces)
                    metric_name = name.split('/')[1].replace(' ', '_')

                    # export and plot classwise scores
                    score_logpath = self.classmetric_log_dir.joinpath(
                            f'{metric_name}_test_epoch{self.current_epoch}.csv')
                    self._export_classmetric(score_logpath, score_epoch)
                    self.logger.experiment.add_figure(f"Test/{metric_name}",
                            self._plot_classmetric(score_epoch),
                            self.current_epoch)
                else :
                    self.log(name, metric)

    def _export_confmat(self, path, confmat):
        np.savetxt(path, confmat)

    def _plot_log_confmat(self,confmat):
        # avoid zero division
        epsilon = 1
        confmat = confmat + epsilon

        return self._plot_confmat(np.log(confmat))

    def _plot_confmat(self, confmat):
        fig, ax = plt.subplots()
        label_ids = np.arange(confmat.shape[0]-1)
        ax.matshow(confmat[:-1,:-1])
        ax.set_xticks(label_ids)
        ax.set_xticklabels([self.label_names[i] for i in label_ids], rotation=90)
        ax.set_yticks(label_ids)
        ax.set_yticklabels([self.label_names[i] for i in label_ids])
        plt.tight_layout()

        return fig

    def _export_classmetric(self, path, scores):
        np.savetxt(path, scores)

    def _plot_classmetric(self, scores):
        fig, ax = plt.subplots()
        label_ids = np.arange(scores.shape[0])
        ax.bar(label_ids, scores)
        ax.set_xticks(label_ids)
        ax.set_xticklabels([self.label_names[i] for i in label_ids], rotation=90)
        plt.tight_layout()
        return fig

    def _load_label_def(self, label_def):
        label_defs = np.loadtxt(label_def, delimiter=',', dtype=str)
        label_names = np.array(label_defs[:,1])
        label_colors = np.array(label_defs[:, 2:], dtype='int')
        return label_names, label_colors
    
    # plot ground truth and predictions
    # `max_imgs` limits number of images to avoid images being very small in the plot
    def _plot_batch_prediction(self, pred, label, undef_mask=False, max_imgs=8):
        batch_size = min(label.shape[0], max_imgs)
        if undef_mask:
            pred[label == self.cfg.ignore_index] = self.cfg.ignore_index
        # four predictions per row, followed by four labelimages
        n_cols = min(batch_size, 4)
        n_rows = 2 * int((batch_size + n_cols - 1)/n_cols)
        figure, axes = plt.subplots(nrows=n_rows, ncols=n_cols, squeeze=False, figsize=(12,12))
        for i in range(batch_size):
            r = i % n_cols
            c = 2 * int(i / n_cols)
            
            axes[c, r].imshow(self.label_colors[label[i]].squeeze())
            axes[c+1, r].imshow(self.label_colors[pred[i]].squeeze())

            # remove axes for cleaner image
            axes[c, r].axis('off')
            axes[c+1, r].axis('off')

        # add legend that shows label color class mapping
        handles = []
        for i, color in enumerate(self.label_colors):
            handles.append(mpatches.Patch(color=color*(1./255), label=self.label_names[i]))

        figure.legend(handles=handles, loc='lower left', ncol=4, mode='expand')
        figure.tight_layout()
        return figure

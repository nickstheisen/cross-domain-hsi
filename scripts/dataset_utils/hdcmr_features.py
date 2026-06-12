#!/usr/bin/env python

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

from hyperseg.models import get_model
from hyperseg.datasets import get_datamodule
from hyperseg.utils import predict_batch

from pytorch_lightning import seed_everything
import torch

from sklearn.decomposition import IncrementalPCA, TruncatedSVD
import time
import gc
from tqdm import tqdm
import h5py
from pathlib import Path
import numpy as np
import pickle as pk
from torchvision import transforms

seed_everything(42)
device = 'cuda'

chunk_size = 2000
n_components = 512

image_h = 956
image_w = 684
step_width=2

source_data_dir = Path('/mnt/data/datasets/hypso1/')
target_data_dir = Path(f'/mnt/data/datasets/hypso1/hdcmr-pca{n_components}')
input_files = ['hypso1_train.h5', 'hypso1_val.h5', 'hypso1_test.h5']

rem_channels = [0,1,2,3,106,107,108,109]

def prepare_hdc_model(cfg, device):
    datamodule = get_datamodule(cfg.dataset)
    datamodule.setup()
    model = get_model(cfg.model, datamodule).to(device)
    return model

def fit_pca(pca, model, infile, sub_batch_size, load_path=None, export_path=None):
    if load_path is not None:
        ipca =  pk.load(open(load_path,'rb'))
        return ipca

    model.eval()
    with h5py.File(infile, 'r', libver='latest') as src:
        min_val = src['metadata']['min-val'][()]
        max_val = src['metadata']['max-val'][()]
        for key in tqdm(list(src.keys())):
            if key == 'metadata':
                continue
            im_shape = src[key]['image'].shape
            for i in tqdm(range(0, im_shape[1], step_width)):
                chunk = src[key]['image'][:,i:i+step_width, :]
                chunk = prepare_sample(chunk, rem_channels, max_val, min_val)
                chunk = predict_batch(model, (chunk, None), 
                              device=device,
                              sub_batch_size=sub_batch_size)
                assert chunk.shape[1] == 9996

                pca.partial_fit(chunk.squeeze(dim=[2,3]))

    if export_path is not None:
        pk.dump(ipca, open(export_path,"wb"))

    print(f' Explained Variance:  {ipca.explained_variance_ratio_.sum()}')
    print(f'IPCA Mat: {ipca.components_.nbytes*1e-9}')
    
    return ipca

def get_dataloader(cfg, mode='train'):
    datamodule = get_datamodule(cfg.dataset)
    datamodule.setup()
    if mode == 'train':
        return datamodule.train_dataloader()
    elif mode == 'val':
        return datamodule.val_dataloader()
    else:
        return datamodule.test_dataloader()

def prepare_sample(chunk, rem_channels, min_val, max_val):
    # remove channels
    chunk = np.delete(chunk, rem_channels, axis=-1)
    chunk = transforms.functional.to_tensor(chunk).type(torch.float)
    chunk = (chunk - min_val) / ( max_val - min_val )
    in_channels = chunk.shape[0]
    # bring in correct shape
    chunk = chunk.reshape(in_channels, -1)
    chunk = chunk.permute([1,0]).unsqueeze(dim=2).unsqueeze(dim=3)
    return chunk

def transform_dataset(infile, outfile, pca, model, sub_batch_size):
    model.eval()
    with h5py.File(infile, 'r', libver='latest') as src:
        min_val = src['metadata']['min-val'][()]
        max_val = src['metadata']['max-val'][()]
        for key in tqdm(list(src.keys())):
            if key == 'metadata':
                continue
            im_shape = src[key]['image'].shape
            for i in range(0, im_shape[1], step_width):
                chunk = src[key]['image'][:,i:i+step_width, :]
                chunk = prepare_sample(chunk, rem_channels, max_val, min_val)
                chunk = predict_batch(model, (chunk, None), 
                              device=device,
                              sub_batch_size=sub_batch_size)
                chunk = pca.transform(chunk.squeeze(dim=[2,3]))
                print(chunk.shape)

@hydra.main(version_base=None, config_path="../../run/conf", config_name="pred_conf")
def main(cfg):
    print(OmegaConf.to_yaml(cfg)) # print config
    '''
    ## prepare HDC-MR Model
    model = prepare_hdc_model(cfg, device)
    model.eval()

    ## load training data and fit PCA parameters
    train_dataset_path = source_data_dir.joinpath('hypso1_train.h5')
    ipca = IncrementalPCA(n_components=n_components, batch_size=None)
    
    with h5py.File(train_dataset_path, 'r', libver='latest') as file:
        for key in tqdm(list(file.keys())):
            if key == 'metadata':
                continue
            # process image column by column
            for i in range(0, image_w, step_width):
                data = file[key][:,i+step_width,:].reshape(-1, 

    '''
    model = prepare_hdc_model(cfg, device)
    ipca = IncrementalPCA(n_components=n_components, batch_size=None)
    infile = source_data_dir.joinpath('hypso1_train.h5')
    ipca = fit_pca(ipca, model, infile, sub_batch_size=cfg.model.sub_batch_size, export_path='pca.pkl')

    for filename in input_files:
        inpath = source_data_dir.joinpath(filename)
        outpath = target_data_dir.joinpath(filename)
        transform_dataset(inpath, outpath, ipca, model, cfg.model.sub_batch_size)
        break
 
    '''
    dataloader = get_dataloader(cfg, 'train')
    for image, _ in tqdm(dataloader):
        features = predict_batch(model, (image, _), device=device, sub_batch_size=cfg.model.sub_batch_size)
        assert features.shape[1] == 9996

        features = features.squeeze(dim=[2,3])
        features = ipca.transform(features)
        print(features.shape)
    '''
    '''
    for batch in tqdm(datamodule.train_dataloader()):
        # calculate HDC-MiniROCKET Features
        features = predict_batch(model, batch, 
                      device=device,
                      sub_batch_size=cfg.model.sub_batch_size)
        assert features.shape[1] == 9996

        features = features.squeeze(dim=[2,3]) # remove empty channel dimensions
        for i in tqdm(range(0, 2000, chunk_size)):
            ipca.partial_fit(features[i:i+chunk_size])
        break

    print(f' Explained Variance:  {ipca.explained_variance_ratio_.sum()}')
    print(ipca.components_.shape)
    print(f'IPCA Mat: {ipca.components_.nbytes*1e-9}')

    batch = next(iter(datamodule.val_dataloader()))
    print('test1')
    features = predict_batch(model, batch, device=device, sub_batch_size=cfg.model.sub_batch_size)
    print('test2')
    print(f'Features: {features.nbytes*1e-9}')


    features = features.squeeze(dim=[2,3]) # remove empty channel dimensions
    chunk_size2=10
    for i in tqdm(range(0, features.shape[0], chunk_size2)):
        dr = ipca.transform(np.random.rand((5,9996)))
        print(dr.shape)
    '''
       
if __name__ == '__main__':
    main()

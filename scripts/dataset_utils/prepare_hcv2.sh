#!/bin/bash

#1. Download HCV2 raw dataset

## Test data
mkdir -p data/raw/hcv2/test
wget -nc https://isis-data.science.uva.nl/cv/HyperspectralCityV2.0/test/V2_test_image.rar \
    -O data/raw/hcv2/test/V2_test_image.rar
unrar x data/raw/hcv2/test/V2_test_image.rar -opdata/raw/hcv2/test/ -o-
wget -nc https://isis-data.science.uva.nl/cv/HyperspectralCityV2.0/test/V2_fixed_test_label.zip \
    -O data/raw/hcv2/test/V2_fixed_test_label.zip
unzip -n data/raw/hcv2/test/V2_fixed_test_label.zip -d data/raw/hcv2/test

## Train & Validation data

mkdir -p data/raw/hcv2/train
wget -nc https://isis-data.science.uva.nl/cv/HyperspectralCityV2.0/train/V2_train_image.rar \
    -O data/raw/hcv2/train/V2_train_image.rar
unrar x data/raw/hcv2/train/V2_train_image.rar -opdata/raw/hcv2/train/ -o-
wget -nc https://isis-data.science.uva.nl/cv/HyperspectralCityV2.0/train/V2_fixed_train_label.zip \
    -O data/raw/hcv2/train/V2_fixed_train_label.zip
unzip -n data/raw/hcv2/train/V2_fixed_train_label.zip -d data/raw/hcv2/train

# 2. Split datasets and onvert raw data to hdf5 format
mkdir -p data/datasets/hcv2
python scripts/dataset_utils/prep_hcv2.py \
    --input_dir data/raw/hcv2/ --output_dir data/datasets/hcv2 \
    --splits_dir scripts/dataset_utils/datasplits/hs3bench/hcv2
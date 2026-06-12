#!/bin/bash

# 1. Download HSIDrive V2.0 raw dataset
mkdir -p data/raw/hsidrive
wget -nc https://ipaccess.ehu.eus/HSI-Drive/files/HSI_Drive_v2_0.zip -O data/raw/hsidrive/HSI_Drive_v2_0.zip
7za x -pehu_drive#v20 data/raw/hsidrive/HSI_Drive_v2_0.zip -odata/raw/hsidrive/ -aos

# 2. Split datasets and convert to hdf5 format
mkdir -p data/datasets/hsidrive
python scripts/dataset_utils/prep_hsidrive.py \
    --input_dir data/raw/hsidrive/Image_dataset --output_dir data/datasets/hsidrive \
    --splits_dir scripts/dataset_utils/datasplits/hs3bench/hsidrive/
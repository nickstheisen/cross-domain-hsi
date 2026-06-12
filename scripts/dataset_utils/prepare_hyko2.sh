#!/bin/bash

#1. Download HyKo2 raw dataset
mkdir -p data/raw/hyko2
wget -nc https://hyko-proxy.uni-koblenz.de/hyko-dataset/HyKo2/vis/vis_annotated.zip -O data/raw/hyko2/vis_annotated.zip
unzip -n data/raw/hyko2/vis_annotated.zip -d data/raw/hyko2

#2. Split datasets and convert to hdf5 format
mkdir -p data/datasets/hyko2
python scripts/dataset_utils/prep_hyko2.py \
    --input_dir data/raw/hyko2/ --output_dir data/datasets/hyko2 \
    --splits_dir scripts/dataset_utils/datasplits/hs3bench/hyko2/
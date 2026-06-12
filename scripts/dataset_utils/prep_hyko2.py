from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import time
import h5py
import numpy as np
from imageio import imread
from scipy.io import loadmat
import csv

parser = ArgumentParser(description="")
parser.add_argument("--input_dir", help="input directory.", required=True)
parser.add_argument("--output_dir", help="output directory", required=True)
parser.add_argument("--splits_dir", help="directory with train/test/val splits in csv format.", 
                    required=True)

def read_split_file(split_file):
    if not split_file.exists():
        print(f"Split file {split_file} does not exist.")
        exit(1)

    with open(split_file, 'r') as f:
        reader = csv.reader(f)
        sample_names = [row[0] for row in reader]
    return set(sample_names)


def convert_dir(dataset_name, inputdir, outdir, split, split_file):
    filenames = read_split_file(split_file)

    sample_files = [inputdir.joinpath(f'{filename}') for filename in filenames]
    outfile = outdir.joinpath(f'{dataset_name}_{split}.h5')

    print(f"{len(sample_files)} samples in {split} split")
    print(f"Start conversion of {split} images!")
    print(f"Writing to {outfile}")

    t1 = time.time()
    with h5py.File(outfile, "a") as hdf_file:
        for sample in tqdm(sample_files):
            if sample.stem in hdf_file:
                continue
            mat = loadmat(sample)
            data = mat.get('data')
            labels = mat.get('label_Semantic Classes for Urban Scenes')
            
            if (data is not None and labels is not None):
                group = hdf_file.create_group(sample.stem)
                group.create_dataset("image", data=data)
                group.create_dataset("labels", data=labels)
            else:
                print(f"{sample} is missing or empty.")
    t2 = time.time()
    print(f"Converted {len(sample_files)} in {t2-t1}s!")
    print()
   

if __name__ == '__main__':
    DATASET_NAME = 'hyko2'
    args = parser.parse_args()
    inputdir = Path(args.input_dir).expanduser().resolve()
    outdir = Path(args.output_dir).expanduser().resolve()
    splitdir = Path(args.splits_dir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    
    for split in ['train', 'test', 'val']:
        split_file = splitdir.joinpath(f'{split}_samples.csv')
        convert_dir(DATASET_NAME, inputdir, outdir, split, split_file)

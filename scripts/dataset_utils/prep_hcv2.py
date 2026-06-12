from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import time
import h5py
import numpy as np
from imageio import imread
from scipy.io import loadmat
import csv
import cv2
import struct

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


def load_data_from_hsd(filename):

    with open(str(filename), 'rb') as f:

        # load meta infos
        height = struct.unpack('i', f.read(4))[0]
        width = struct.unpack('i', f.read(4))[0]
        bands = struct.unpack('i', f.read(4))[0]
        D = struct.unpack('i', f.read(4))[0]
        startw = struct.unpack('i', f.read(4))[0]
        stepw = struct.unpack('f', f.read(4))[0]
        endw = struct.unpack('i', f.read(4))[0]

        # load average values per band?
        averages = np.zeros((bands), dtype="float32")
        for i in range(bands):
            averages[i] = struct.unpack('f', f.read(4))[0]
            
        # load coefficients for dimensionality reduction matrix
        coeffs = np.zeros((D*bands), dtype="float32")
        for i in range(D*bands):
            coeffs[i] = struct.unpack('f', f.read(4))[0]
        
        # load dimensionality reduced data
        scoredata = np.zeros((height*width*D), dtype="float32")
        for i in range(height*width*D):
            scoredata[i] = struct.unpack('f', f.read(4))[0]

        # reconstruct data
        coeffs = coeffs.reshape((D, bands), order='C')
        scoredata = scoredata.reshape((height*width, D), order='C')
        temp = scoredata @ coeffs
        data1 = temp + averages
        
        # reconstruct original structure of hyperspectral cube
        data = data1.reshape(height, width, bands, order='C')
        return data

def load_labels_from_png(filename):
    label_img = cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE)
    return label_img


def convert_dir(dataset_name, inputdir, outdir, split, split_file, subsample=2):
    filenames = read_split_file(split_file)
    subdir = "train" if (split == "train" or split == "val") else "test"
    inputdir = inputdir.joinpath(subdir)


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
            
            data = load_data_from_hsd(sample)
            labels = load_labels_from_png(inputdir.joinpath(f'gt/rgb{sample.stem}_gray.png'))
            
            if (data is not None and labels is not None):
                if subsample is not None:
                    scaling_factor = 1.0 / subsample
                    labels = cv2.resize(labels, dsize=None, fx=scaling_factor, fy=scaling_factor, 
                            interpolation=cv2.INTER_NEAREST)
                    data = cv2.resize(data, dsize=None, fx=scaling_factor, fy=scaling_factor, 
                            interpolation=cv2.INTER_NEAREST)

                group = hdf_file.create_group(sample.stem)
                group.create_dataset("image", data=data)
                group.create_dataset("labels", data=labels)
            else:
                print(f"{sample} is missing or empty.")
    t2 = time.time()
    print(f"Converted {len(sample_files)} in {t2-t1}s!")
    print()
   

if __name__ == '__main__':
    DATASET_NAME = 'hcv2'
    args = parser.parse_args()
    inputdir = Path(args.input_dir).expanduser().resolve()
    outdir = Path(args.output_dir).expanduser().resolve()
    splitdir = Path(args.splits_dir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    
    for split in ['train', 'test', 'val']:
        split_file = splitdir.joinpath(f'{split}_samples.csv')
        convert_dir(DATASET_NAME, inputdir, outdir, split, split_file)
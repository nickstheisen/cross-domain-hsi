from crossdomainhsi.models.hypersl import HyperSLBackbone
from crossdomainhsi.datasets.utils import read_wavelengths_file
import crossdomainhsi
from pathlib import Path
import h5py
import torch
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from argparse import ArgumentParser

parser = ArgumentParser(description="")
parser.add_argument("--inputdir", 
                    help="input directory. Output will be written to \{input directory\}/\{backbone\}", 
                    required=True)
parser.add_argument("--backbone", help="backbone model", required=True, choices=["hypersl"])
parser.add_argument("--dataset", help="dataset to be processed", required=True, 
                    choices=["hcv2","hyko2","hsidrive"])
parser.add_argument("--remove_channels", help="remove these channls", default=None)

batchsizes = {
    "hcv2": 5000,
    "hsidrive": 5000,
    "hyko2": 50000
}

def convert(in_filepath, out_filepath, model, batchsize, remove_channels=None):
    with h5py.File(in_filepath, "r") as in_file, h5py.File(out_filepath, "a") as out_file:
        for key in tqdm(in_file.keys()):
            if ((f"{key}/image" in out_file) or
                (f"{key}/labels" in out_file)):
                continue

            if ((f"{key}/image" not in in_file) or
                (f"{key}/labels" not in in_file)):
                continue
            # copy each group to out_file
            in_file.copy(key, out_file)
            # get image as numpy array
            img = out_file[key]['image'][:]
            if remove_channels is not None:
                img = np.delete(img, remove_channels, -1)
            img = torch.Tensor(img).to('cuda')
            h,w,c = img.shape
            img = img.reshape(-1, c)

            with torch.no_grad():
                features = torch.zeros((img.shape[0], model.embedding_dim))
                for i in range((img.shape[0]//batchsize) +1):
                    lower = i*batchsize
                    upper = (i+1)*batchsize
                    if upper > img.shape[0]:
                        upper = img.shape[0]
                    sub_img = img[lower:upper].unsqueeze(1).unsqueeze(2)
                    wls = wavelengths.repeat(upper - lower, 1)
                    result = model(sub_img, wls)
                    features[lower:upper] = result.squeeze()

                del out_file[key]['image']
                features = features.view(h,w,-1)
                out_file[key].create_dataset('image', data=features.cpu().numpy(), dtype=features.cpu().numpy().dtype)

if __name__ == '__main__':

    args = parser.parse_args()

    inputdir = Path(args.inputdir).expanduser().resolve()
    outdir = inputdir.joinpath(args.backbone)

    splits = ['train', 'val', 'test']
    modelname = args.backbone
    batchsize = batchsizes[args.dataset]

    wavelength_filepath = Path(crossdomainhsi.__file__).parent.joinpath(f"datasets/spectralbands/wave_{args.dataset}.csv")

    if args.backbone == "hypersl":
        model = HyperSLBackbone(model_size='small')
        weights_path = Path(crossdomainhsi.__file__).parent.joinpath(
        "../modelweights/hypersl_backbone/5_base_mask95_checkpoint.pt")
    else:
        raise RuntimeError(f"model {args.model} unknown.")

    # HyperSL
    ckpt = torch.load(weights_path)
    weights = OrderedDict()
    for k, v in ckpt['model'].items():
        name = k[7:]
        weights[name] = v
    model.spectral_encoder.load_state_dict(weights)
    model.to('cuda')
    wavelengths = torch.tensor(read_wavelengths_file(wavelength_filepath)).to('cuda').unsqueeze(0)
    #remove_channels = [0,1,2,3,106,107,108,109]

    #################################################################################

    for split in splits:
        outdir.mkdir(parents=True, exist_ok=True)
        in_filepath = inputdir.joinpath(f"{args.dataset}_{split}.h5")
        out_filepath = outdir.joinpath(f"{args.dataset}_{split}.h5")
        print(in_filepath, out_filepath)
        convert(in_filepath, out_filepath, model, batchsize, args.remove_channels)

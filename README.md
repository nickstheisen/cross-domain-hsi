**Disclaimer: This repository is incomplete. Code and the rest of documentation will follow soon.**

# Cross-Domain Transfer of Hyperspectral Foundation Models
This repository is the official implementation of the paper:

**Cross-Domain Transfer of Hyperspectral Foundation Models** <br/>
Nick Theisen, Peer Neubert </br>
*International Conference on Pattern Recognition (ICPR), 2026*

## Abstract

Hyperspectral imaging (HSI) semantic segmentation typically relies on in-domain training, but limited data availability often restricts model performance in real-world applications. Current approaches to leverage foundation models in proximal sensing use cross-modality techniques, bridging RGB and HSI to exploit vision foundation models. However, these methods either discard spectral information or introduce architectural complexity. We propose cross-domain transfer as an alternative, reusing HSI foundation models – originally trained in remote sensing – for proximal sensing applications. By eliminating the need to bridge modality gaps, our approach preserves spectral information while maintaining a simple architecture. Using the HS3-Bench benchmark, we systematically evaluate and compare conventional in-domain, in-modality training, cross-modality transfer and cross-domain transfer strategies. Our results demonstrate that cross-domain transfer achieves large performance improvements over in-domain, in-modality training,reduces the performance gap to cross-modality approaches and maintains strong performance in limited data settings. Thus, this work advances more effective HSI semantic segmentation in diverse applications.

## Setup

### Installation
1. Setup cronda environment:
```bash
conda create -n cross-domain-hsi python=3.11 -y
conda activate cross-domain-hsi
```

2. Install the python package from this repo. This also installs required dependencies. For reference we have also exported the packages in our environment into a `requirements.txt`.
```
pip install -e .
```

3. Verify installation. Open python console. The following should not throw errors.
```python
import crossdomainhsi
import torch
```

### Dataset Preparation
- The raw datasets need to be converted to [HDF5](https://www.hdfgroup.org/solutions/hdf5/) file format to be compatible with our Base-Class HSDataModule. 
- We provide scripts to download and process (i.e. apply dataset splits and convert to HDF5) the datasets mentioned below. After processing only the HDF5 files need to be kept and raw data could be deleted.

|**Dataset**|**Size (raw + extracted data)**|**Size (processed data)**|
|-----------|-------------------|--------------------|
|HyKo2  |5.2GB  | 2.4|
|HSI-Drive|28GB|8.7GB|
|HCV2|337GB|419GB|

#### HyKo2
Download and prepare [HyKo2](https://hyko-proxy.uni-koblenz.de/hyko-dataset/HyKo2/vis).
```bash
conda activate cross-domain-hsi
bash scripts/dataset_utils/prepare_hyko2.sh
```

#### HSI-DriveV2
7zip is required to extract password-protected .zip-file.
```
sudo apt install p7zip-full
```
Download and prepare [HSI-Drive V2.0](https://ipaccess.ehu.eus/HSI-Drive/#download)
```bash
conda activate cross-domain-hsi
bash scripts/dataset_utils/prepare_hyko2.sh
```

#### Hyperspectral City V2 (HCV2)
_Note: Dataset is huge. Download as well as processing might take a long time. Consider running this script over night._

`unrar` is required to extract .rar-File
```bash
sudo apt install unrar
```
Download and prepare [HCV2](https://isis-data.science.uva.nl/cv/HyperspectralCityV2.0/)
```bash
conda activate cross-domain-hsi
bash scripts/dataset_utils/prepare_hcv2.sh
```

## Reproduction
The dataset splits, model weights and the HDF5-Converted HyKo2 dataset for reference can be downloaded [here](https://drive.google.com/drive/folders/16YTyiVuKEXhxgUFmwF9Uy-f86Oct1FyS?usp=sharing).

- In some experiments we used features from a HyperSL-Backbone as input. We did not finetune the backbone. Hence the features can be calculated once for each dataset and then be processed with different models. TODO upload script ..

We provide different scripts for training, testing and inference. The model and dataset configurations can be found in `run/conf/`. 

To perform evaluation on the test data, you can donwload the checkpoints [here](https://drive.google.com/drive/folders/16YTyiVuKEXhxgUFmwF9Uy-f86Oct1FyS?usp=sharing) and call the according script, e.g. for evaluation of unet on hyko2 call:

```
python run/test.py dataset=hyko2 model=unet model.ckpt=./modelweights/hyko2/hyko2-unet.ckpt
```


## Acknowledgement
This repository reuses code from the HS3-Bench repository https://github.com/nickstheisen/hyperseg and HyperSL repository https://github.com/kkweil/HyperSL. We would like to thank the authors.

This work was partially funded by Wehrtechnische Dienststelle 41 (WTD), Koblenz, Germany.
